"""Compact export is an exact projection; every numerical guard still runs."""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_liquid_transient import wave_model, wave_reference

from surface import HydraulicConvergenceError, LiquidTransientNetwork, ThermalLiquidStation
from surface.liquid_journal import compact_record
from surface.torch_backend import liquid_backend_config

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def projection(result, branches):
    return {**result, "steps": [compact_record(r, branches) for r in result["steps"]]}


@pytest.mark.parametrize("device", DEVICES)
def test_compact_wave_preserves_state_and_each_scalar_audit(device):
    model = LiquidTransientNetwork.from_dict(
        {**wave_model(32).to_dict(), "device": device, "temporal_error_control": True}
    )
    initial = model.initialize(dict(zip(model.free, wave_reference(32, 0))))
    full = model.advance(initial, 0.0002)
    bridge = model._torch_bridge
    before = bridge.transfer_stats()
    compact = model.advance(
        initial, 0.0002, hydraulic_journal="compact", hydraulic_journal_branches=["p0"]
    )
    after = bridge.transfer_stats()
    assert compact == projection(full, ("p0",))
    assert after["state_download_batches"] - before["state_download_batches"] == 1
    assert after["endpoint_download_batches"] - before["endpoint_download_batches"] == 1
    assert after["journal_download_bytes"] - before["journal_download_bytes"] < 8 * 200 * len(
        compact["steps"]
    )
    json.dumps(compact, allow_nan=False)


def station(device):
    config = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    config["system"]["temperature_trips"][0]["threshold_k"] = 300.0001
    config["system"]["temperature_trips"][0]["delay_s"] = 0
    model = ThermalLiquidStation.from_dict(liquid_backend_config(config["system"], "torch", device))
    return model, model.initialize(**config["initial"]), config


@pytest.mark.parametrize("device", DEVICES)
def test_compact_full_stack_preserves_trips_controls_recycle_and_heat(device):
    model, initial, config = station(device)
    commands = {k: v for k, v in config["periods"][0].items() if k != "duration_s"}
    full = model.advance(initial, 0.12, **commands)
    compact = model.advance(initial, 0.12, hydraulic_journal="compact", **commands)
    branches = tuple(x.valve for x in model.recycling.loops)
    assert compact == projection(full, branches)
    assert compact["temperature_events"]
    assert compact["recycled_volume_m3"] == full["recycled_volume_m3"]


@pytest.mark.parametrize("device", DEVICES)
def test_compact_failure_is_not_partial_state_and_retry_is_exact(monkeypatch, device):
    model, initial, config = station(device)
    original = initial.to_dict()
    step = LiquidTransientNetwork._step
    calls = []

    def failing(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise HydraulicConvergenceError("injected compact retry")
        return step(self, *args, **kwargs)

    monkeypatch.setattr(LiquidTransientNetwork, "_step", failing)
    commands = {k: v for k, v in config["periods"][0].items() if k != "duration_s"}
    result = model.advance(initial, 0.01, hydraulic_journal="compact", **commands)
    assert result["failed_steps"] >= 1
    assert initial.to_dict() == original
    assert result["state"]["heat"]["time_s"] == pytest.approx(0.01)


def test_compact_validation_has_no_hidden_fallback():
    model = wave_model(8)
    initial = model.initialize()
    for kwargs in (
        {"hydraulic_journal": "bad"},
        {"hydraulic_journal": "compact", "hydraulic_state_transfer": "solve"},
        {"hydraulic_journal_branches": ["unknown"]},
        {"hydraulic_journal_branches": ["p0", "p0"]},
        {"hydraulic_journal_branches": "p0"},
    ):
        with pytest.raises(ValueError):
            model.advance(initial, 0.001, **kwargs)
    data = model.to_dict()
    data["backend"] = "scipy_reference"
    reference = LiquidTransientNetwork.from_dict(data)
    with pytest.raises(ValueError, match="resident Torch"):
        reference.advance(reference.initialize(), 0.001, hydraulic_journal="compact")
    thermal, state, _ = station("cpu")
    for kwargs in ({"thermal_state_transfer": "window"}, {"hydraulic_heat_transfer": "host"}):
        with pytest.raises(ValueError, match="direct resident"):
            thermal.advance(state, 0.001, hydraulic_journal="compact", **kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_each_accepted_device_field_is_identical_even_when_not_exported(device):
    model = LiquidTransientNetwork.from_dict(
        {**wave_model(32).to_dict(), "device": device, "temporal_error_control": True}
    )
    initial = model.initialize(dict(zip(model.free, wave_reference(32, 0))))
    paths = []
    for mode in ("full", "compact"):
        samples = []

        def capture(record, payload, mode=mode, samples=samples):
            assert payload.journal == mode
            samples.append(tuple(v.detach().clone() for v in payload.data))

        model.advance(initial, 0.0002, hydraulic_journal=mode, _on_accepted_step=capture)
        paths.append(samples)
    assert len(paths[0]) == len(paths[1]) > 1
    for full, compact in zip(*paths):
        assert all(torch.equal(a, b) for a, b in zip(full, compact))


@pytest.mark.parametrize("device", DEVICES)
def test_compact_keeps_two_stops_regeneration_and_relief_opening(device):
    from check_liquid_protection import relief_model
    from check_liquid_transient import coupled_model

    data = coupled_model().to_dict()
    data["device"] = device
    for drive in data["drives"].values():
        drive["viscous_drag_nm_s"] = 0
    data["no_flow_power_w_at_nominal"] = {"a": 0, "b": 0}
    model = LiquidTransientNetwork.from_dict(data)
    initial = model.initialize({"h": 500000, "t": 480000}, speeds_rad_s={"a": 0.01, "b": 0.02})
    commands = {"motor_torques_nm": {"a": -10, "b": -10}}
    full = model.advance(initial, 0.004, **commands)
    compact = model.advance(initial, 0.004, hydraulic_journal="compact", **commands)
    assert compact == projection(full, ())
    assert sum(r["electrical_energy_j"] for r in compact["steps"]) < 0
    assert compact["state"]["speeds_rad_s"] == {"a": 0, "b": 0}
    model = LiquidTransientNetwork.from_dict({**relief_model(0.001).to_dict(), "device": device})
    initial = model.initialize({"v": 515000})
    full = model.advance(initial, 0.003)
    compact = model.advance(initial, 0.003, hydraulic_journal="compact")
    assert compact == projection(full, ())
    assert any(r["relief_openings"] for r in compact["steps"])


@pytest.mark.parametrize("device", DEVICES)
def test_compact_thermal_failure_does_not_publish_and_can_retry(monkeypatch, device):
    from surface import ConservationError
    from surface.torch_heat_session import TorchHeatWindow

    model, initial, config = station(device)
    saved = initial.to_dict()
    push = TorchHeatWindow.push
    calls = []

    def fail(window, record, payload):
        calls.append(1)
        if len(calls) == 2:
            raise ConservationError("compact thermal sentinel")
        return push(window, record, payload)

    commands = {k: v for k, v in config["periods"][0].items() if k != "duration_s"}
    with monkeypatch.context() as patch:
        patch.setattr(TorchHeatWindow, "push", fail)
        with pytest.raises(ConservationError, match="sentinel"):
            model.advance(initial, 0.01, hydraulic_journal="compact", **commands)
    assert len(calls) == 2 and initial.to_dict() == saved
    full = model.advance(initial, 0.01, **commands)
    compact = model.advance(initial, 0.01, hydraulic_journal="compact", **commands)
    assert compact == projection(full, tuple(x.valve for x in model.recycling.loops))


@pytest.mark.parametrize("device", DEVICES)
def test_compact_checkpoint_replays_and_binds_export_policy(device):
    from surface import run_thermal_liquid_schedule

    model, initial, config = station(device)
    periods = [
        {**config["periods"][0], "duration_s": 0.01, "hydraulic_journal": "compact"}
        for _ in range(3)
    ]
    full = run_thermal_liquid_schedule(model, initial, periods)
    prefix = run_thermal_liquid_schedule(model, initial, periods, stop_after=1)
    resumed = run_thermal_liquid_schedule(model, initial, periods, checkpoint=prefix)
    assert resumed == full
    with pytest.raises(ValueError, match="fingerprint"):
        run_thermal_liquid_schedule(
            model, initial, [{**p, "hydraulic_journal": "full"} for p in periods], checkpoint=prefix
        )


@pytest.mark.parametrize("device", DEVICES)
def test_user_branches_merge_with_recycle_and_selector_cache_is_bounded(device):
    model, initial, config = station(device)
    commands = {k: v for k, v in config["periods"][0].items() if k != "duration_s"}
    full = model.advance(initial, 0.01, **commands)
    selected = ("line", "p1")
    compact = model.advance(
        initial,
        0.01,
        hydraulic_journal="compact",
        hydraulic_journal_branches=list(selected),
        **commands,
    )
    retained = (*selected, *(x.valve for x in model.recycling.loops))
    assert compact == projection(full, retained)
    wave = LiquidTransientNetwork.from_dict({**wave_model(8).to_dict(), "device": device})
    state = wave.initialize()
    for i in range(6):
        result = wave.advance(
            state, 0.00001, hydraulic_journal="compact", hydraulic_journal_branches=[f"p{i}"]
        )
        assert list(result["steps"][0]["midpoint_branch_volume_rates_m3_s"]) == [f"p{i}"]
    assert len(wave._torch_bridge._compact_layouts) == 4


@pytest.mark.parametrize("device", DEVICES)
def test_compact_pressure_retries_and_failed_budget_keep_committed_cache(device):
    from check_liquid_transient import oscillator

    model = LiquidTransientNetwork.from_dict(
        {**oscillator().to_dict(), "device": device, "max_pressure_change_pa": 5}
    )
    initial = model.initialize({"a": 601000, "b": 599000})
    full = model.advance(initial, 0.005)
    assert full["failed_steps"] > 0
    cache = model._torch_bridge._interval_cache
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        model.advance(initial, 0.005, hydraulic_journal="compact", max_substeps=1)
    assert model._torch_bridge._interval_cache is cache
    compact = model.advance(initial, 0.005, hydraulic_journal="compact")
    assert compact == projection(full, ())
