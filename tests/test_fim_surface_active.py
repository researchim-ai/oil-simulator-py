"""Actual FIM feedback and exact reduced Jacobian with active surface equipment."""

import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_fim_surface_coupling import assemble_context, solver, total_rate

from surface import EquipmentOperatingError

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/fim_water_surface_active.json").read_text())


@pytest.mark.parametrize("target", [None, 6e6, 9e6, 2e6])
def test_active_equipment_full_well_blocks_match_finite_differences(target):
    c = config()
    surface = c["simulation"]["physical_surface_network"]
    if target is None:
        surface.pop("pressure_control")
    else:
        surface["pressure_control"]["target_pressure_pa"] = target
    c["fluid"]["s_w"] = 0.85  # central differences stay inside the water-only mobility domain
    s = solver(c)
    p, sw, pb, assemble = assemble_context(s)
    _, _, ws = assemble()
    B, D = ws["B_full"].clone(), ws["D_full"].clone()
    for j in range(2):
        a, b = pb.clone(), pb.clone()
        a[j] += 10
        b[j] -= 10
        plus = assemble(pw=a)[2]["r_w"].clone()
        minus = assemble(pw=b)[2]["r_w"].clone()
        torch.testing.assert_close((plus - minus) / 20, D[:, j], rtol=2e-6, atol=2e-6)
    for cell in (0, 2):
        for k, base, h in ((0, p, 10.0), (1, sw, 1e-6)):
            a, b = base.clone(), base.clone()
            a[cell] += h
            b[cell] -= h
            plus = assemble(**({"pp": a} if k == 0 else {"ss": a}))[2]["r_w"].clone()
            minus = assemble(**({"pp": b} if k == 0 else {"ss": b}))[2]["r_w"].clone()
            expected = B[:, s.wells.conn_cell == cell, k].sum(1)
            torch.testing.assert_close((plus - minus) / (2 * h), expected, rtol=3e-5, atol=3e-5)
    if target == 6e6:
        assert abs(D[0, 1]) < 1e-8  # fixed header pressure decouples the two well pressures
    else:
        assert abs(D[0, 1]) > 0.001


def test_pressure_target_and_speed_change_reservoir_solution_and_balance():
    rows = []
    for target in (5.5e6, 6e6, 7e6):
        c = config()
        c["simulation"]["physical_surface_network"]["pressure_control"]["target_pressure_pa"] = (
            target
        )
        s = solver(c)
        initial = sum(
            float(m.sum()) for m in s.calc_masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten())
        )
        assert s.step(86400)
        report = s.wells.surface_report
        reg = report["pressure_control"]
        r = report["network"]
        assert reg["status"] == "target" and reg["evaluations"] < 12
        assert abs(r["pressures_pa"]["header"] - target) < 1e-4
        final = sum(
            float(m.sum()) for m in s.calc_masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten())
        )
        assert initial - final == pytest.approx(total_rate(s) * 86400, abs=0.01)
        assert -r["boundary_mass_rates_kg_s"]["outlet"] == pytest.approx(total_rate(s), abs=1e-9)
        assert abs(r["power_balance_error_w"]) < 1e-5
        rows.append(total_rate(s))
    assert rows[0] > rows[1] > rows[2] > 0


def test_control_changes_preserve_equipment_and_validate_atomically():
    s = solver(config())
    w = s.wells
    w.set_surface_boundary_pressures({"outlet": 9e6})
    assert len(w.surface_network.pumps) == len(w.surface_network.valves) == 1
    before = w.surface_network
    with pytest.raises(ValueError, match="owns"):
        w.set_surface_controls(pump_speed_ratios={"booster": 0.8})
    with pytest.raises(ValueError):
        w.set_surface_controls(valve_openings={"export_valve": 2.0})
    assert w.surface_network is before
    w.set_surface_pressure_control(None)
    w.set_surface_controls(pump_speed_ratios={"booster": 0.8}, valve_openings={"export_valve": 0.5})
    assert w.surface_network.pumps[0].speed_ratio == 0.8
    assert w.surface_network.valves[0].opening == 0.5
    assert s.step(86400)


def test_equipment_safety_error_rolls_back_reservoir_and_wells():
    c = config()
    c["simulation"]["physical_surface_network"]["network"]["pumps"][0]["max_shaft_power_w"] = 1.0
    s = solver(c)
    p = s.fluid.pressure.clone()
    assert s.wells.p_bh is None
    with pytest.raises(EquipmentOperatingError, match="shaft power"):
        s.step(86400)
    assert torch.equal(s.fluid.pressure, p) and s.wells.p_bh is None
    assert s.wells.surface_report is None


def test_regulator_numerical_failure_rolls_back_and_restart_keeps_controls():
    s = solver(config())
    assert s.step(86400)
    restored = pickle.loads(pickle.dumps(s))
    assert restored.wells.surface_pressure_control == s.wells.surface_pressure_control
    assert restored.step(86400, 86400) and s.step(86400, 86400)
    assert torch.equal(restored.fluid.pressure, s.fluid.pressure)
    p = s.fluid.pressure.clone()
    report = copy.deepcopy(s.wells.surface_report)
    s.wells.set_surface_pressure_control(
        replace(s.wells.surface_pressure_control, max_evaluations=3)
    )
    assert not s.step(86400, 172800)
    assert torch.equal(s.fluid.pressure, p) and s.wells.surface_report == report
    assert "regulator" in s.last_surface_failure


def test_regulator_moves_to_speed_limit_and_returns_to_target():
    s = solver(config())
    assert s.step(86400)
    assert s.wells.surface_report["pressure_control"]["status"] == "target"
    s.wells.set_surface_boundary_pressures({"outlet": 12e6})
    assert s.step(86400, 86400)
    assert s.wells.surface_report["pressure_control"]["status"] == "max_speed"
    s.wells.set_surface_boundary_pressures({"outlet": 8e6})
    assert s.step(86400, 172800)
    assert s.wells.surface_report["pressure_control"]["status"] == "target"


def test_station_recipe_retains_limits_and_standby_availability_in_fim():
    from surface import PumpStationControl

    c = config()
    spec = c["simulation"]["physical_surface_network"]
    spec.pop("pressure_control")
    pump = spec["network"]["pumps"][0]
    spec["network"]["pumps"] = [{**copy.deepcopy(pump), "name": f"unit_{i}"} for i in (1, 2, 3)]
    s = solver(c)
    w = s.wells
    policy = PumpStationControl(
        ("unit_1", "unit_2", "unit_3"),
        "header",
        6e6,
        0.5,
        1.2,
        switch_saving_w=20.0,
        pressure_tolerance_pa=0.01,
    )
    options = {"source_mass_rates_kg_s": {"bottom_1": 0.4, "bottom_2": 0.6}}
    chosen = policy.solve(w.station_network, **options)
    w.set_surface_station(chosen)
    assert s.step(86400)
    assert set(w.surface_report["station"]["running_pumps"]) == set(chosen.running_pumps)
    disabled = chosen.running_pumps[0]
    chosen2 = policy.solve(
        w.station_network, unavailable=[disabled], previous_running=chosen.running_pumps, **options
    )
    w.set_surface_station(chosen2)
    assert s.step(86400, 86400)
    assert disabled not in w.surface_report["station"]["running_pumps"]
    with pytest.raises(ValueError, match="owns"):
        w.set_surface_controls(pump_status={disabled: True})
    w.set_surface_boundary_pressures({"outlet": 9e6})
    with pytest.raises(ValueError, match="configuration"):
        w.set_surface_station(chosen2)
    w.set_surface_station(None)
    assert all(p.open for p in w.surface_network.pumps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("regulated", [False, True])
def test_active_surface_cuda_matches_cpu(regulated):
    from simulator.fluid import Fluid
    from simulator.manual_assembly_solver import ManualFIMSolver
    from simulator.reservoir import Reservoir
    from simulator.well import WellManager

    c = config()
    if not regulated:
        c["simulation"]["physical_surface_network"].pop("pressure_control")
    results = []
    for name in ("cpu", "cuda"):
        dev = torch.device(name)
        r = Reservoir(c["reservoir"], device=dev)
        f = Fluid(c["fluid"], r, device=dev)
        s = ManualFIMSolver(r, f, WellManager(c["wells"], r), c["simulation"])
        assert s.step(86400)
        assert s.device.type == name and s._last_well_sys["D_full"].device.type == name
        results.append(s)
    cpu, gpu = results
    torch.testing.assert_close(gpu.fluid.pressure.cpu(), cpu.fluid.pressure, rtol=0, atol=0.001)
    torch.testing.assert_close(gpu.fluid.s_w.cpu(), cpu.fluid.s_w, rtol=0, atol=1e-10)
    assert total_rate(gpu) == pytest.approx(total_rate(cpu), rel=1e-8)
    assert gpu._last_well_sys["B_full"].is_cuda and gpu._last_well_sys["D_full"].is_cuda
    if regulated:
        assert gpu.wells.surface_report["pressure_control"]["status"] == "target"
