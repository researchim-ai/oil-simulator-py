"""Thermal gravity vessel: analytic limits, ledgers, restart and atomic failure."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.liquid import HydraulicConvergenceError, PropertyModelError
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
from surface.stratified_pr_thermal import StratifiedPRThermal, StratifiedThermalState
from surface.vessel_wall import VesselWall

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def model(device, **kwargs):
    cfg = json.loads((ROOT / "configs/surface_stratified_pr_gas.json").read_text())
    cfg["system"]["geometry"]["length_m"] = 2.0
    c = StratifiedPRColumn.from_dict(cfg["system"], device=device)
    s = StratifiedPRState.from_dict(c.initialize(**cfg["initial"])["state"])
    w = VesselWall(1000.0, 10.0, 4.0, 350.0)
    m = StratifiedPRThermal(c, w, initial_step_s=0.1, maximum_step_s=2.0, **kwargs)
    return m, m.initialize(s, 340.0)


@pytest.mark.parametrize("device", DEVICES)
def test_energy_entropy_restart_and_exact_interval(device):
    m, s = model(device)
    original = s.to_dict()
    r = m.advance(s, 0.2)
    f = StratifiedThermalState.from_dict(r["state"])
    assert f.time_s == 0.2
    assert f.column.component_moles == s.column.component_moles
    assert f.column.temperature_k > s.column.temperature_k
    assert abs(r["energy_residual_j"]) < 1e-6
    assert r["entropy_production_j_k"] >= -1e-7
    assert s.to_dict() == original
    m.inspect(f)
    resumed = StratifiedThermalState.from_dict(json.loads(json.dumps(f.to_dict())))
    assert m.advance(resumed, 0.1)["state"] == m.advance(f, 0.1)["state"]


@pytest.mark.parametrize("device", DEVICES)
def test_decoupled_wall_matches_backward_euler(device):
    m, s = model(device)
    m = StratifiedPRThermal(m.column, VesselWall(1000.0, 0.0, 4.0, 350.0), initial_step_s=0.1)
    s = m.initialize(s.column, 340.0)
    f, report = m._step(s, 1.0)
    assert f.wall_temperature_k == pytest.approx((1000 * 340 + 4 * 350) / 1004, abs=1e-10)
    assert f.column.temperature_k == pytest.approx(s.column.temperature_k, abs=1e-7)
    assert f.column.total_energy_j == s.column.total_energy_j
    assert abs(report["energy_residual_j"]) < 1e-7


def test_atomic_budget_and_domain_rejections():
    m, s = model("cpu", maximum_attempts=1)
    original = s.to_dict()
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        m.advance(s, 1.0)
    assert s.to_dict() == original
    with pytest.raises(PropertyModelError):
        m.advance(replace(s, model_id="0" * 64), 0.1)
    with pytest.raises(ValueError):
        m.advance(s, float("nan"))
    with pytest.raises(ValueError):
        m.inspect(replace(s, wall_temperature_k=float("nan")))


@pytest.mark.parametrize("device", DEVICES)
def test_dilute_gas_implicit_step_matches_analytic_two_capacity_system(device):
    cfg = json.loads((ROOT / "configs/surface_stratified_pr_gas.json").read_text())
    cfg["system"]["geometry"]["length_m"] = 1.0
    cfg["system"]["gravity_m_s2"] = 0.0
    cfg["system"]["fluid"]["min_pressure_pa"] = 1e-7
    cfg["initial"]["pressure_pa"] = 0.01
    c = StratifiedPRColumn.from_dict(cfg["system"], device=device)
    s = StratifiedPRState.from_dict(c.initialize(**cfg["initial"])["state"])
    # Heat capacity of this dilute inventory is very small; scale wall/UA to it.
    capacity = sum(
        n * (component.ideal_cp_j_mol_k - 8.31446261815324)
        for n, component in zip(s.component_moles, c.fluid.components)
    )
    wall = VesselWall(capacity * 2, capacity * 0.1, capacity * 0.03, 360.0)
    m = StratifiedPRThermal(c, wall)
    initial = m.initialize(s, 340.0)
    dt = 0.5
    actual, report = m._step(initial, dt)
    import numpy as np

    a = np.array(
        [
            [capacity + dt * wall.fluid_ua_w_k, -dt * wall.fluid_ua_w_k],
            [
                -dt * wall.fluid_ua_w_k,
                wall.heat_capacity_j_k + dt * (wall.fluid_ua_w_k + wall.ambient_ua_w_k),
            ],
        ]
    )
    b = np.array(
        [capacity * s.temperature_k, wall.heat_capacity_j_k * 340 + dt * wall.ambient_ua_w_k * 360]
    )
    expected = np.linalg.solve(a, b)
    assert actual.column.temperature_k == pytest.approx(expected[0], abs=2e-6)
    assert actual.wall_temperature_k == pytest.approx(expected[1], abs=2e-6)
    assert report["entropy_production_j_k"] >= -1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_two_phase_heating_conserves_inventory_and_moves_interface(device):
    cfg = json.loads((ROOT / "configs/surface_stratified_pr.json").read_text())
    c = StratifiedPRColumn.from_dict(cfg["system"], device=device)
    s = StratifiedPRState.from_dict(c.initialize(**cfg["initial"])["state"])
    assert len(s.phase_roots) == 2
    m = StratifiedPRThermal(c, VesselWall(100.0, 1.0, max_temperature_change_k=10.0))
    initial = m.initialize(s, s.temperature_k + 2.0)
    result, report = m._step(initial, 0.05)
    assert len(result.column.phase_roots) == 2
    assert result.column.component_moles == s.component_moles
    assert result.column.temperature_k > s.temperature_k
    assert result.column.interface_heights_m != s.interface_heights_m
    assert abs(report["energy_residual_j"]) < 1e-6


def test_isolated_stationary_solution_and_model_reconstruction():
    m, s = model("cpu")
    m = StratifiedPRThermal(m.column, VesselWall(1000.0, 0.0, 0.0))
    s = m.initialize(s.column, 330.0)
    reconstructed = StratifiedPRThermal.from_dict(json.loads(json.dumps(m.to_dict())))
    assert reconstructed.model_id == m.model_id
    result = reconstructed.advance(s, 0.1)
    final = StratifiedThermalState.from_dict(result["state"])
    assert final.column.temperature_k == pytest.approx(s.column.temperature_k, abs=1e-7)
    assert final.wall_temperature_k == s.wall_temperature_k
    assert final.column.total_energy_j == s.column.total_energy_j
    assert result["ambient_heat_j"] == 0.0
    assert abs(result["energy_residual_j"]) < 1e-7


def test_cooling_and_elevation_gauge_invariance():
    m, s = model("cpu")
    cold = m.initialize(s.column, 300.0)
    first, report = m._step(cold, 0.1)
    assert first.column.temperature_k < s.column.temperature_k
    assert report["entropy_production_j_k"] >= -1e-7
    cfg = m.column.to_dict()
    cfg["elevation_m"] += 10000.0
    column = StratifiedPRColumn.from_dict(cfg)
    mass = sum(
        n * component.molar_mass_kg_mol
        for n, component in zip(s.column.component_moles, column.fluid.components)
    )
    shifted = replace(
        s.column,
        total_energy_j=s.column.total_energy_j + mass * column.gravity_m_s2 * 10000.0,
        model_id=column.model_id,
    )
    equivalent = StratifiedPRThermal(column, m.wall)
    second, other = equivalent._step(equivalent.initialize(shifted, 300.0), 0.1)
    assert second.column.temperature_k == pytest.approx(first.column.temperature_k, abs=1e-7)
    assert second.wall_temperature_k == pytest.approx(first.wall_temperature_k, abs=1e-7)
    assert other["fluid_heat_j"] == pytest.approx(report["fluid_heat_j"], abs=1e-6)
