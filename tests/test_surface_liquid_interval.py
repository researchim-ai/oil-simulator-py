"""Resident mechanical state, rejected trials and exact legacy event parity."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_liquid_protection import relief_model
from check_liquid_transient import coupled_model, oscillator, wave_model, wave_reference

from surface import (
    HydraulicConvergenceError,
    LiquidTransientNetwork,
    LiquidTransientState,
    ThermalLiquidStation,
)
from surface.torch_backend import liquid_backend_config

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def device_model(model, device, **kwargs):
    return LiquidTransientNetwork.from_dict({**model.to_dict(), "device": device, **kwargs})


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("adaptive", [False, True])
def test_same_steps_and_audits_without_root_vector_downloads(device, adaptive):
    m = device_model(coupled_model(), device, temporal_error_control=adaptive)
    initial = m.initialize({"h": 550000, "t": 500000}, {"line": 0.002}, {"a": 100, "b": 110})
    old = m.advance(
        initial, 0.006, motor_torques_nm={"a": 5, "b": 7}, hydraulic_state_transfer="solve"
    )
    b = m._torch_bridge
    before = b.transfer_stats()
    new = m.advance(initial, 0.006, motor_torques_nm={"a": 5, "b": 7})
    assert new == old
    after = b.transfer_stats()
    stats = {k: after[k] - before[k] for k in before}
    assert stats["state_upload_batches"] == 1
    assert stats["root_vector_download_bytes"] == stats["root_vector_upload_bytes"] == 0
    assert stats["journal_download_batches"] == len(new["steps"])
    assert stats["state_download_batches"] == len(new["steps"])
    assert stats["cache_misses"] == 1
    json.dumps(new, allow_nan=False)


@pytest.mark.parametrize("device", DEVICES)
def test_braking_two_stops_and_regeneration_keep_legacy_times(device):
    data = coupled_model().to_dict()
    data["device"] = device
    for d in data["drives"].values():
        d["viscous_drag_nm_s"] = 0
    data["no_flow_power_w_at_nominal"] = {"a": 0, "b": 0}
    m = LiquidTransientNetwork.from_dict(data)
    initial = m.initialize({"h": 500000, "t": 480000}, speeds_rad_s={"a": 0.01, "b": 0.02})
    commands = {"a": -10, "b": -10}
    old = m.advance(initial, 0.004, motor_torques_nm=commands, hydraulic_state_transfer="solve")
    new = m.advance(initial, 0.004, motor_torques_nm=commands)
    assert new == old
    assert new["state"]["speeds_rad_s"] == {"a": 0, "b": 0}
    assert sum(s["electrical_energy_j"] for s in new["steps"]) < 0


@pytest.mark.parametrize("device", DEVICES)
def test_rejected_pressure_trials_do_not_export_full_history(device):
    m = device_model(oscillator(), device, max_pressure_change_pa=5)
    initial = m.initialize({"a": 601000, "b": 599000})
    old = m.advance(initial, 0.005, hydraulic_state_transfer="solve")
    b = m._torch_bridge
    before = b.transfer_stats()
    new = m.advance(initial, 0.005)
    stats = {k: v - before[k] for k, v in b.transfer_stats().items()}
    assert new == old and new["failed_steps"] > 0
    assert stats["journal_download_batches"] == len(new["steps"])
    assert stats["state_upload_batches"] == 1
    assert stats["root_vector_download_bytes"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_cache_survives_only_committed_intervals_and_checks_input_bits(device):
    m = device_model(coupled_model(), device)
    state = m.initialize({"h": 550000, "t": 500000}, {"line": 0.002}, {"a": 100, "b": 110})
    first = m.advance(state, 0.002, motor_torques_nm={"a": 5, "b": 7})
    state = LiquidTransientState(**first["state"])
    before = m._torch_bridge.transfer_stats()
    second = m.advance(
        state, 0.002, motor_torques_nm={"a": 6, "b": 4}, boundary_pressures_pa={"out": 479000}
    )
    after = m._torch_bridge.transfer_stats()
    assert after["cache_hits"] - before["cache_hits"] == 1
    assert after["state_upload_batches"] == before["state_upload_batches"]
    state = LiquidTransientState(**second["state"])
    cache = m._torch_bridge._interval_cache
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        m.advance(state, 0.006, max_substeps=1)
    assert m._torch_bridge._interval_cache is cache
    # Same clock is insufficient: editing a pressure must invalidate reuse.
    pressures = dict(state.pressures_pa)
    pressures["h"] += 0.01
    edited = replace(state, pressures_pa=pressures)
    before = m._torch_bridge.transfer_stats()
    m.advance(edited, 0.001)
    after = m._torch_bridge.transfer_stats()
    assert after["cache_misses"] - before["cache_misses"] == 1
    assert after["state_upload_batches"] - before["state_upload_batches"] == 1


@pytest.mark.parametrize("device", DEVICES)
def test_relief_and_tiny_positive_torque_keep_domain_semantics(device):
    m = device_model(relief_model(0.001), device)
    initial = m.initialize({"v": 515000})
    assert m.advance(initial, 0.003) == m.advance(initial, 0.003, hydraulic_state_transfer="solve")
    m = device_model(coupled_model(), device)
    initial = m.initialize({"h": 500000, "t": 480000})
    old = m.advance(initial, 0.002, motor_torques_nm={"a": 2e-8}, hydraulic_state_transfer="solve")
    new = m.advance(initial, 0.002, motor_torques_nm={"a": 2e-8})
    assert new == old and 0 < new["state"]["speeds_rad_s"]["a"] < 1e-8


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_physical_wave_matches_legacy_and_independent_mode(device):
    m = device_model(wave_model(80), device)
    initial = m.initialize(dict(zip(m.free, wave_reference(80, 0))))
    old = m.advance(initial, 0.00005, hydraulic_state_transfer="solve")
    new = m.advance(initial, 0.00005)
    assert new == old
    values = np.asarray([new["state"]["pressures_pa"][n] for n in m.free])
    assert max(abs(values - wave_reference(80, 0.00005))) < 1e-4
    assert m._torch_bridge.equations.newton.matrix.size == 161


@pytest.mark.parametrize("device", DEVICES)
def test_short_coupled_station_keeps_all_pid_and_thermal_records(device):
    c = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    m = ThermalLiquidStation.from_dict(liquid_backend_config(c["system"], "torch", device))
    state = m.initialize(**c["initial"])
    commands = {k: v for k, v in c["periods"][0].items() if k != "duration_s"}
    old = m.advance(state, 0.02, hydraulic_state_transfer="solve", **commands)
    new = m.advance(state, 0.02, **commands)
    assert new == old


def test_invalid_transfer_mode_and_stale_kernel_are_rejected():
    m = coupled_model()
    state = m.initialize()
    with pytest.raises(ValueError, match="hydraulic_state_transfer"):
        m.advance(state, 0.001, hydraulic_state_transfer="unknown")
    m.advance(state, 0.001)
    m.max_pressure_change_pa += 1
    state = m.initialize()
    with pytest.raises(ValueError, match="model changed"):
        m.advance(state, 0.001)
