"""Persistent thermal fields across real control windows, with unchanged audits."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_thermal_liquid import pipe_probe

from surface import (
    ConservationError,
    HydraulicConvergenceError,
    LiquidHeatNetwork,
    ThermalLiquidStation,
)
from surface.torch_backend import liquid_backend_config

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def transport(device, history=True, count=6, cells=80):
    m, _, base = pipe_probe(
        cells=cells, dt=0.25 / cells, duration=0.25 / cells, compact=not history
    )
    data = m.to_dict()
    data["physics"]["device"] = data["device"] = device
    data["temporal_error_control"] = True
    m = LiquidHeatNetwork.from_dict(data)
    state = m.initialize(m.physics.initialize())
    records = []
    for i in range(count):
        rec = copy.deepcopy(base)
        rec["time_s"] = i * rec["dt_s"]
        q = 0.001 if i % 3 else -0.001
        rec["midpoint_branch_volume_rates_m3_s"] = {"line": q, "v": q}
        rec["boundary_mass_transfers_in_kg"] = {
            "in": 1000 * q * rec["dt_s"],
            "out": -1000 * q * rec["dt_s"],
        }
        records.append(rec)
    return m, state, records


def physical(result):
    return {k: v for k, v in result.items() if k != "transfer_stats"}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("history", [False, True])
def test_window_preserves_each_audit_and_moves_state_once(device, history):
    m, state, records = transport(device, history)
    old = m.advance_records(state, records, resident=False)
    new = m.advance_records(state, records)
    assert physical(new) == physical(old)
    a, b = old["transfer_stats"], new["transfer_stats"]
    assert a["state_upload_batches"] == a["state_download_batches"] == len(records)
    assert b["state_upload_batches"] == b["state_download_batches"] == 1
    size = (2 * len(m.names) + len(m.wall_names)) * 8
    assert b["state_upload_bytes"] == b["state_download_bytes"] == size
    assert a["upload_bytes"] - b["upload_bytes"] == (len(records) - 1) * size
    assert a["download_bytes"] - b["download_bytes"] == (len(records) - 1) * size
    expected_history = len(new["steps"]) * (len(m.names) + len(m.wall_names)) * 8 if history else 0
    assert a["history_download_bytes"] == b["history_download_bytes"] == expected_history
    json.dumps(physical(new), allow_nan=False)


@pytest.mark.parametrize("device", DEVICES)
def test_bounded_history_flush_does_not_change_results(device):
    m, state, records = transport(device, True, count=3, cells=16)
    expected = m.advance_records(state, records)
    m._torch_bridge.history_buffer_bytes = 2 * (len(m.names) + len(m.wall_names)) * 8
    actual = m.advance_records(state, records)
    assert physical(actual) == physical(expected)
    assert actual["transfer_stats"]["history_flushes"] > 0
    assert (
        actual["transfer_stats"]["history_download_bytes"]
        == expected["transfer_stats"]["history_download_bytes"]
    )


@pytest.mark.parametrize("device", DEVICES)
def test_failed_window_does_not_commit_state_and_can_retry(device):
    m, state, records = transport(device, False, count=3)
    before = state.to_dict()
    bad = copy.deepcopy(records)
    bad[-1]["pump_heat_j"] = 1
    with pytest.raises(ConservationError, match="ledger"):
        m.advance_records(state, bad)
    assert state.to_dict() == before
    expected = m.advance_records(state, records, resident=False)
    assert physical(m.advance_records(state, records)) == physical(expected)
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        m.advance_records(state, records, max_substeps=1)
    assert state.to_dict() == before


@pytest.mark.parametrize("device", DEVICES)
def test_control_window_keeps_temperature_trip_and_pid_clock(device):
    c = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    c["system"]["temperature_trips"][0]["threshold_k"] = 300.0001
    c["system"]["temperature_trips"][0]["delay_s"] = 0
    m = ThermalLiquidStation.from_dict(liquid_backend_config(c["system"], "torch", device))
    state = m.initialize(**c["initial"])
    commands = {k: v for k, v in c["periods"][0].items() if k != "duration_s"}
    old = m.advance(state, 0.12, thermal_state_transfer="record", **commands)
    new = m.advance(state, 0.12, **commands)
    assert new == old
    trips = [e for e in new["temperature_events"] if e["type"] == "temperature_trip"]
    assert trips and trips[0]["time_s"] == pytest.approx(0.05, abs=1e-14)
    assert new["latched_temperature_trips"]


def test_window_validates_timeline_modes_and_model_ownership():
    m, state, records = transport("cpu", False, count=2)
    gap = copy.deepcopy(records)
    gap[1]["time_s"] += 1
    with pytest.raises(ValueError, match="time mismatch"):
        m.advance_records(state, gap)
    with pytest.raises(TypeError, match="boolean"):
        m.advance_records(state, records, resident=1)
    with pytest.raises(ValueError, match="nonempty"):
        m.advance_records(state, [])
    m.advance_records(state, records)
    m.specific_heat_j_kg_k += 1
    valid_new_state = m.initialize(m.physics.initialize())
    with pytest.raises(ValueError, match="model changed"):
        m.advance_records(valid_new_state, records)


@pytest.mark.parametrize("device", DEVICES)
def test_session_keeps_fields_across_windows_and_owns_transfer_counts(device):
    from surface import LiquidHeatState

    m, initial, records = transport(device, False)
    sensors = ("a", "line::0")
    session = m.open_session(initial, fluid_sensors=sensors)
    old = initial
    all_steps, all_audits = [], []
    for index in range(0, len(records), 2):
        window = records[index : index + 2]
        # Interleave another caller of the same kernel: session accounting and
        # state must not accidentally include its transfers or cached forcing.
        reference = m.advance_records(old, window, resident=False)
        old = LiquidHeatState(**reference["state"])
        result = session.advance_records(window)
        assert dict(result["state"].temperatures_k) == {
            n: dict(old.temperatures_k)[n] for n in sensors
        }
        assert result["state"].wall_temperatures_k == ()
        assert result["steps"] == reference["steps"]
        assert result["audits"] == reference["audits"]
        assert result["transfer_stats"]["state_upload_batches"] == 0
        assert result["transfer_stats"]["state_download_batches"] == 0
        all_steps.extend(result["steps"])
        all_audits.extend(result["audits"])
    assert session.close() == old
    stats = session.transfer_stats()
    assert stats["state_upload_batches"] == stats["state_download_batches"] == 1
    assert stats["sensor_download_batches"] == len(records) // 2
    assert stats["sensor_download_bytes"] == len(records) // 2 * len(sensors) * 8
    for action in (
        session.observe,
        session.snapshot,
        session.close,
        lambda: session.advance_records(records),
        lambda: session.align_time(0),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            action()


@pytest.mark.parametrize("device", DEVICES)
def test_session_rolls_back_rejected_window_and_retries(device):
    m, initial, records = transport(device, False)
    session = m.open_session(initial, fluid_sensors=("a",))
    session.advance_records(records[:2])
    before = session.snapshot()
    bad = copy.deepcopy(records[2:4])
    bad[-1]["pump_heat_j"] = 1
    with pytest.raises(ConservationError):
        session.advance_records(bad)
    assert session.snapshot() == before
    with pytest.raises(ValueError, match="unintegrated"):
        session.align_time(before.time_s + 1)
    assert session.time_s == before.time_s
    session.advance_records(records[2:])
    expected = m.advance_records(initial, records, resident=False)
    assert session.close().to_dict() == expected["state"]
    assert initial.time_s == 0


@pytest.mark.parametrize("device", DEVICES)
def test_report_window_and_record_modes_keep_wall_trip_identical(device):
    c = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    c["system"]["temperature_trips"][0].update(
        metric="wall_temperature_k", location="h", threshold_k=300.00000001, delay_s=0
    )
    m = ThermalLiquidStation.from_dict(liquid_backend_config(c["system"], "torch", device))
    state = m.initialize(**c["initial"])
    commands = {k: v for k, v in c["periods"][0].items() if k != "duration_s"}
    old = m.advance(state, 0.12, thermal_state_transfer="record", **commands)
    assert m.advance(state, 0.12, thermal_state_transfer="window", **commands) == old
    before = m.heat._torch_bridge.transfer_stats()
    new = m.advance(state, 0.12, **commands)
    after = m.heat._torch_bridge.transfer_stats()
    assert new == old
    assert new["latched_temperature_trips"]
    assert after["state_upload_batches"] - before["state_upload_batches"] == 1
    assert after["state_download_batches"] - before["state_download_batches"] == 1
    with pytest.raises(ValueError, match="thermal_state_transfer"):
        m.advance(state, 0.12, thermal_state_transfer="unknown", **commands)


def test_sensor_validation_empty_observations_and_model_mutation():
    m, state, records = transport("cpu", False, count=2)
    for fluid, wall in [(("missing",), ()), (("a", "a"), ()), ((), ("a",))]:
        with pytest.raises(ValueError, match="sensor"):
            m.open_session(state, fluid, wall)
    session = m.open_session(state)
    observations = session.advance_records(records)["state"]
    assert observations.temperatures_k == observations.wall_temperatures_k == ()
    assert session.transfer_stats()["sensor_download_batches"] == 0
    assert session.close().to_dict() == m.advance_records(state, records)["state"]
    session = m.open_session(state)
    m.specific_heat_j_kg_k += 1
    with pytest.raises(ValueError, match="model changed"):
        session.advance_records(records)


def test_existing_positional_station_budgets_keep_their_meaning():
    c = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    model = ThermalLiquidStation.from_dict(c["system"])
    initial = model.initialize(**c["initial"])
    old_positional = model.advance(
        initial,
        0.001,
        "off",
        None,
        False,
        (),
        None,
        None,
        None,
        None,
        None,
        None,
        100000,
        100000,
        100000,
        100000,
        100000,
        100000,
    )
    assert old_positional == model.advance(initial, 0.001, mode="off")
