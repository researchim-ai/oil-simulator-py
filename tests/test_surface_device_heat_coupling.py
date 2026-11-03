"""Accepted-step streaming preserves journals, transactions and device ownership."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from surface import HydraulicConvergenceError, ThermalLiquidStation
from surface.torch_backend import liquid_backend_config
from surface.torch_heat_session import TorchHeatWindow

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def station(device):
    config = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    model = ThermalLiquidStation.from_dict(liquid_backend_config(config["system"], "torch", device))
    return model, model.initialize(**config["initial"]), config


@pytest.mark.parametrize("device", DEVICES)
def test_direct_coupling_matches_all_logs_and_eliminates_vector_upload(device):
    model, initial, config = station(device)
    commands = {k: v for k, v in config["periods"][0].items() if k != "duration_s"}
    host = model.advance(initial, 0.12, hydraulic_heat_transfer="host", **commands)
    bridge = model.heat._torch_bridge
    before = bridge.transfer_stats()
    actual = model.advance(initial, 0.12, hydraulic_heat_transfer="device", **commands)
    after = bridge.transfer_stats()
    assert actual == host
    assert after["device_hydraulic_records"] - before["device_hydraulic_records"] == len(
        actual["steps"]
    )
    # Public output must remain JSON-safe; tensors are strictly short-lived.
    json.dumps(actual, allow_nan=False)
    assert after["device_hydraulic_bytes"] > before["device_hydraulic_bytes"]
    before_host = bridge.transfer_stats()
    model.advance(initial, 0.12, hydraulic_heat_transfer="host", **commands)
    host_bytes = bridge.transfer_stats()["upload_bytes"] - before_host["upload_bytes"]
    assert after["upload_bytes"] - before["upload_bytes"] < host_bytes


@pytest.mark.parametrize("device", DEVICES)
def test_stream_window_failure_keeps_session_retryable(device):
    model, initial, _ = station(device)
    session = model.heat.open_session(initial.heat)
    window = TorchHeatWindow(session, None, None, {}, 10000)
    physics = model.recycling.physics
    records, payloads = [], []

    def collect(record, payload):
        records.append(record)
        payloads.append(payload)

    physical = initial.recycling.protected.controlled.physical
    physics.advance(physical, 0.002, _on_accepted_step=collect)
    assert len(records) >= 2 and payloads[0] is not None
    before = session.snapshot().to_dict()
    window.push(records[0], payloads[0])
    with pytest.raises(ValueError, match="model mismatch"):
        window.push(records[1], replace(payloads[1], model_id="wrong"))
    assert session.snapshot().to_dict() == before
    with pytest.raises(RuntimeError, match="closed"):
        window.finish()
    retry = TorchHeatWindow(session, None, None, {}, 10000)
    for record, payload in zip(records, payloads):
        retry.push(record, payload)
    retry.finish()
    expected = model.heat.advance_records(initial.heat, records)
    assert session.close().to_dict() == expected["state"]


@pytest.mark.parametrize("device", DEVICES)
def test_wrong_device_coordinates_rejected_before_thermal_commit(device):
    model, initial, _ = station(device)
    physics = model.recycling.physics
    captured = []
    physics.advance(
        initial.recycling.protected.controlled.physical,
        0.001,
        _on_accepted_step=lambda r, p: captured.append((r, p)),
    )
    record, payload = captured[0]
    data = list(payload.data)
    data[0] = data[0].to(torch.float32)
    session = model.heat.open_session(initial.heat)
    window = TorchHeatWindow(session, None, None, {}, 10000)
    with pytest.raises(ValueError, match="coordinates/device"):
        window.push(record, replace(payload, data=tuple(data)))
    assert session.snapshot().to_dict() == initial.heat.to_dict()


def test_explicit_modes_validate_and_callback_error_is_not_hydraulic_retry():
    model, initial, _ = station("cpu")
    with pytest.raises(ValueError, match="resident Torch"):
        model.advance(
            initial, 0.001, hydraulic_heat_transfer="device", hydraulic_state_transfer="solve"
        )
    with pytest.raises(ValueError, match="hydraulic_heat_transfer"):
        model.advance(initial, 0.001, hydraulic_heat_transfer="invalid")
    physical = initial.recycling.protected.controlled.physical
    original = copy.deepcopy(physical.to_dict())
    calls = []

    def fail(record, payload):
        calls.append(record)
        raise HydraulicConvergenceError("thermal callback sentinel")

    with pytest.raises(HydraulicConvergenceError, match="sentinel"):
        model.recycling.physics.advance(physical, 0.01, _on_accepted_step=fail)
    assert len(calls) == 1
    assert physical.to_dict() == original


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("direction", [-1, 1])
def test_direct_transport_finite_reservoir_and_reversal(device, direction):
    # Manufactured transport isolates interface ordering; it is not a
    # momentum-equation oracle. Physical coupled cases are checked above.
    from check_thermal_liquid import manufactured_record, reference_pipe

    from surface import LiquidHeatNetwork, LiquidTransientNetwork
    from surface.torch_liquid_interval import DeviceHydraulicStep

    physics = LiquidTransientNetwork.from_dict(
        {**reference_pipe().physics.to_dict(), "device": device}
    )
    model = LiquidHeatNetwork(
        physics,
        1000,
        {"in": 310},
        boundary_reservoirs_kg={"out": 2},
        reference_temperature_k=300,
        pipe_cells={"line": 8},
        maximum_dt_s=0.001,
        temporal_error_control=False,
        store_step_temperatures=True,
    )
    initial = model.initialize(physics.initialize())
    record = manufactured_record(physics, 0.004)
    rate = direction * 0.001
    record["midpoint_branch_volume_rates_m3_s"] = {b.name: rate for b in physics._branches}
    record["boundary_mass_transfers_in_kg"] = {
        "in": rate * 1000 * 0.004,
        "out": -rate * 1000 * 0.004,
    }
    tensor = lambda v: torch.tensor(v, dtype=torch.float64, device=device)
    data = [None] * 11
    data[0] = tensor([record["final_pressures_pa"][n] for n in physics.free])
    data[4] = tensor([rate for _ in physics._branches])
    ledger = {
        "friction_unit_heat_j": tensor(
            [0.0] * (len(physics.network.pipes) + len(physics.network.valves))
        ),
        **{
            k: tensor([0.0] * len(physics.names))
            for k in ("pump_unit_heat_j", "drag_unit_heat_j", "motor_unit_heat_j")
        },
        **{
            k: tensor(0.0)
            for k in ("friction_heat_j", "pump_heat_j", "shaft_drag_heat_j", "motor_heat_j")
        },
        "boundary_mass_transfers_in_kg": tensor(
            [record["boundary_mass_transfers_in_kg"].get(n, 0.0) for n in physics._nodes]
        ),
    }
    expected = model.advance_records(initial, [record])
    session = model.open_session(initial)
    window = TorchHeatWindow(session, None, None, {}, 10000)
    window.push(record, DeviceHydraulicStep(physics.model_id, tuple(data), ledger))
    result = window.finish()
    assert result["steps"] == expected["steps"]
    assert result["audits"] == expected["audits"]
    assert session.close().to_dict() == expected["state"]
    assert dict(expected["state"]["masses_kg"])["out"] == pytest.approx(2 + direction * 0.004)


@pytest.mark.parametrize("device", DEVICES)
def test_stream_releases_device_payloads_and_commits_only_after_sensor_export(monkeypatch, device):
    import weakref

    from surface.torch_heat_bridge import TorchHeatBridge

    model, initial, config = station(device)
    observed = []
    original_push = TorchHeatWindow.push
    original_download = TorchHeatBridge.download

    def track(window, record, payload):
        observed.append(weakref.ref(payload))
        assert sum(ref() is not None for ref in observed) <= 2
        return original_push(window, record, payload)

    monkeypatch.setattr(TorchHeatWindow, "push", track)
    commands = {k: v for k, v in config["periods"][0].items() if k != "duration_s"}
    model.advance(initial, 0.03, hydraulic_heat_transfer="device", **commands)
    assert observed and not any(ref() is not None for ref in observed)
    monkeypatch.setattr(TorchHeatWindow, "push", original_push)
    session = model.heat.open_session(initial.heat, fluid_sensors=(model.heat.names[0],))
    window = TorchHeatWindow(session, None, None, {}, 10000)
    captured = []
    model.recycling.physics.advance(
        initial.recycling.protected.controlled.physical,
        0.001,
        _on_accepted_step=lambda r, p: captured.append((r, p)),
    )
    for record, payload in captured:
        window.push(record, payload)

    def fail_download(bridge, values):
        raise RuntimeError("sensor export sentinel")

    monkeypatch.setattr(TorchHeatBridge, "download", fail_download)
    with pytest.raises(RuntimeError, match="sentinel"):
        window.finish()
    monkeypatch.setattr(TorchHeatBridge, "download", original_download)
    assert session.snapshot().to_dict() == initial.heat.to_dict()
    window.finish()
    assert session.time_s == pytest.approx(0.001)


@pytest.mark.parametrize("device", DEVICES)
def test_direct_schedule_restart_replays_every_record_and_binds_transfer_mode(device):
    from surface import run_thermal_liquid_schedule

    model, initial, config = station(device)
    periods = [
        {**config["periods"][0], "duration_s": 0.01, "hydraulic_heat_transfer": "device"}
        for _ in range(3)
    ]
    full = run_thermal_liquid_schedule(model, initial, periods)
    prefix = run_thermal_liquid_schedule(model, initial, periods, stop_after=1)
    resumed = run_thermal_liquid_schedule(model, initial, periods, checkpoint=prefix)
    assert resumed == full
    wrong = [{**p, "hydraulic_heat_transfer": "host"} for p in periods]
    with pytest.raises(ValueError, match="fingerprint"):
        run_thermal_liquid_schedule(model, initial, wrong, checkpoint=prefix)


@pytest.mark.parametrize("device", DEVICES)
def test_model_change_during_stream_is_rejected_before_window_publication(device):
    model, initial, _ = station(device)
    session = model.heat.open_session(initial.heat)
    cursor = session._cursor
    window = TorchHeatWindow(session, None, None, {}, 10000)
    model.recycling.physics.advance(
        initial.recycling.protected.controlled.physical, 0.001, _on_accepted_step=window.push
    )
    cp = model.heat.specific_heat_j_kg_k
    model.heat.specific_heat_j_kg_k += 1
    try:
        with pytest.raises(ValueError, match="model changed"):
            window.finish()
        assert session._cursor is cursor
    finally:
        model.heat.specific_heat_j_kg_k = cp
    assert session.snapshot().to_dict() == initial.heat.to_dict()
