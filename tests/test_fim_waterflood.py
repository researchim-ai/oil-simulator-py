"""Actual mixed FIM step, network tangents, resources and transactional failure."""

import copy
import json
from pathlib import Path

import pytest
import torch

from simulator.fluid import Fluid
from simulator.reservoir import Reservoir
from simulator.simulation import Simulator
from simulator.well import WellManager
from surface.liquid import PropertyModelError
from surface.water_injection import WaterInjectionTree

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    c = json.loads((ROOT / "configs/fim_blackoil_waterflood.json").read_text())
    c["fluid"]["pvt_path"] = str(ROOT / c["fluid"]["pvt_path"])
    return c


def simulation(config=None):
    c = configuration() if config is None else config
    r = Reservoir(c["reservoir"], device=torch.device("cpu"))
    f = Fluid(c["fluid"], r, device=torch.device("cpu"))
    return Simulator(r, f, WellManager(c["wells"], r), c["simulation"], device=torch.device("cpu"))


def test_tree_pressure_derivative_and_shared_pump_against_finite_difference():
    spec = copy.deepcopy(
        configuration()["simulation"]["physical_surface_network"]["injection_network"]["network"]
    )
    spec["nodes"].append({"name": "injection_2", "elevation_m": -30})
    spec["pipes"].append({**spec["pipes"][0], "name": "injection_line_2", "end": "injection_2"})
    tree = WaterInjectionTree(spec)
    base = {"injection_1": 0.4, "injection_2": 0.3}
    result = tree.evaluate(base)
    for j, name in enumerate(tree.leaves):
        plus, minus = base.copy(), base.copy()
        plus[name] += 1e-5
        minus[name] -= 1e-5
        a, b = tree.evaluate(plus), tree.evaluate(minus)
        expected = torch.tensor(
            [(a["node_pressures_pa"][n] - b["node_pressures_pa"][n]) / 2e-5 for n in tree.leaves],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            result["pressure_derivatives_pa_per_kg_s"][:, j], expected, rtol=2e-8, atol=1e-3
        )
    assert result["pressure_derivatives_pa_per_kg_s"][0, 1] < 0
    assert result["feasible"]


def test_mixed_global_well_jacobian_blocks_match_finite_difference():
    sim = simulation()
    s = sim.fim_solver
    s.wells.injection.speed_ratios = {"injection_booster": 0.6}
    s.wells.injection.wells[0].control_value = 10000
    p = s.fluid.pressure.flatten().clone()
    sw = s.fluid.s_w.flatten().clone()
    x = s.x.clone()
    old = s.masses(p, sw, x, s.sat, s.ug)
    s.wells.begin_step(*s.cell_state(p, sw, x, s.sat, s.ug), 0)
    pb = p.new_tensor([2.4e6, 3.2e6, 2.4e6])

    def assemble(pp=p, ww=sw, xx=x, bhp=pb):
        return s.assemble_bo(pp, ww, xx, s.sat, old, 864, bhp, ug=s.ug)

    _, _, sys = assemble()
    B = sys["B_full"].clone()
    D = sys["D_full"].clone()
    cv = sys["Cv"].clone()
    assert s.wells.step_mode[1] == "surface"
    for j in range(3):
        a, b = pb.clone(), pb.clone()
        a[j] += 2
        b[j] -= 2
        ra, _, plus = assemble(bhp=a)
        rb, _, minus = assemble(bhp=b)
        torch.testing.assert_close((plus["r_w"] - minus["r_w"]) / 4, D[:, j], rtol=3e-5, atol=1e-5)
        expected = cv.new_zeros((s.n_cells, 3)).index_add_(
            0, s.wells.conn_cell[s.wells.conn_well == j], cv[s.wells.conn_well == j]
        )
        torch.testing.assert_close((ra - rb).view(-1, 3) / 4, expected, rtol=4e-5, atol=1e-9)
    for cell in range(3):
        for k, h in enumerate((2.0, 1e-6, 1e-6)):
            a = [p.clone(), sw.clone(), x.clone()]
            b = [v.clone() for v in a]
            a[k][cell] += h
            b[k][cell] -= h
            plus = assemble(*a)[2]
            minus = assemble(*b)[2]
            expected = B[:, s.wells.conn_cell == cell, k].sum(1)
            torch.testing.assert_close(
                (plus["r_w"] - minus["r_w"]) / (2 * h), expected, rtol=4e-5, atol=0.003
            )


def test_water_budget_is_closed_by_a_new_reservoir_solution():
    sim = simulation()
    driver = sim.facility_coupling
    s = sim.fim_solver
    assert driver.step(864, current_time=0)
    r = driver.report
    water = r["water_rates_kg_s"]
    assert water["injected"] > water["produced"] > 0
    assert 0.049 < water["makeup"] <= 0.05 + 1e-12
    assert water["disposed"] == 0
    injected = -s.wells._Q[1, 1].item() * s.fluid.rho_water_ref
    assert injected == pytest.approx(water["injected"], abs=1e-12)
    assert max(map(abs, r["material_account"]["residual_kg"])) < 0.01
    assert len(r["trace"]) > 2
    assert [w.control_value for w in s.wells.wells] == [100, 100, 100]


def test_pump_shutdown_and_restart_and_no_makeup():
    sim = simulation()
    driver = sim.facility_coupling
    driver.set_controls(injection_enabled=False)
    assert driver.step(864, current_time=0)
    water = driver.report["water_rates_kg_s"]
    assert water["injected"] == water["makeup"] == 0
    assert water["disposed"] == water["produced"]
    assert driver.report["injection_network"]["shaft_power_w"] == 0
    driver.set_controls(injection_enabled=True, makeup_water_max_kg_s=0)
    assert driver.step(864, current_time=864)
    water = driver.report["water_rates_kg_s"]
    assert 0 < water["injected"] <= water["produced"] + 1e-12
    assert water["makeup"] == 0
    assert max(map(abs, driver.report["material_account"]["residual_kg"])) < 0.01


def test_pump_power_curtails_injection_and_disposes_excess_water():
    c = configuration()
    c["simulation"]["physical_surface_network"]["injection_network"]["network"]["pumps"][0][
        "max_shaft_power_w"
    ] = 500
    sim = simulation(c)
    driver = sim.facility_coupling
    assert driver.step(864, current_time=0)
    assert driver.report["injection_network"]["shaft_power_w"] <= 500
    water = driver.report["water_rates_kg_s"]
    assert water["disposed"] > 0 and water["makeup"] == 0


def test_post_reservoir_exception_restores_parent_children_and_caps(monkeypatch):
    sim = simulation()
    driver = sim.facility_coupling
    s = sim.fim_solver
    assert driver.step(864, current_time=0)
    previous = driver.report
    fields = [s.fluid.pressure.clone(), s.fluid.s_w.clone(), s.x.clone()]
    children = [
        (w.p_bh.clone(), w.conn_sched.clone(), copy.deepcopy(w.step_mode))
        for w in s.wells.subwell_models
    ]

    def fail(*args, **kwargs):
        raise PropertyModelError("forced after FIM commit")

    monkeypatch.setattr(type(driver.facility), "run", fail)
    with pytest.raises(PropertyModelError, match="after FIM"):
        driver.step(864, current_time=864)
    assert driver.report is previous
    for a, b in zip(fields, [s.fluid.pressure, s.fluid.s_w, s.x]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for child, (pb, schedule, mode) in zip(s.wells.subwell_models, children):
        torch.testing.assert_close(child.p_bh, pb, rtol=0, atol=0)
        torch.testing.assert_close(child.conn_sched, schedule, rtol=0, atol=0)
        assert child.step_mode == mode
    assert [w.control_value for w in s.wells.wells] == [100, 100, 100]


def test_two_injectors_share_a_pump_in_one_full_well_block():
    c = configuration()
    c["reservoir"]["dimensions"] = [4, 1, 1]
    c["wells"][2]["i"] = 3
    c["wells"].append({**c["wells"][1], "name": "I2", "i": 2})
    spec = c["simulation"]["physical_surface_network"]["injection_network"]
    spec["well_nodes"]["I2"] = "injection_2"
    spec["network"]["nodes"].append({"name": "injection_2", "elevation_m": -40})
    spec["network"]["pipes"].append(
        {**spec["network"]["pipes"][0], "name": "injection_line_2", "end": "injection_2"}
    )
    sim = simulation(c)
    s = sim.fim_solver
    s.wells.injection.speed_ratios = {"injection_booster": 0.6}
    for well in s.wells.injection.wells:
        well.control_value = 10000
    p = s.fluid.pressure.flatten().clone()
    sw = s.fluid.s_w.flatten().clone()
    x = s.x.clone()
    old = s.masses(p, sw, x, s.sat, s.ug)
    s.wells.begin_step(*s.cell_state(p, sw, x, s.sat, s.ug), 0)
    pb = p.new_tensor([2.4e6, 3.2e6, 2.4e6, 3.2e6])

    def assemble(bhp):
        return s.assemble_bo(p, sw, x, s.sat, old, 864, bhp, ug=s.ug)[2]

    sys = assemble(pb)
    D = sys["D_full"].clone()
    assert abs(float(D[1, 3])) > 1e-4 and abs(float(D[3, 1])) > 1e-4
    for j in range(4):
        a, b = pb.clone(), pb.clone()
        a[j] += 2
        b[j] -= 2
        torch.testing.assert_close(
            (assemble(a)["r_w"] - assemble(b)["r_w"]) / 4, D[:, j], rtol=4e-5, atol=1e-5
        )


def test_insufficient_pump_head_blocks_injection_without_reverse_sources():
    sim = simulation()
    driver = sim.facility_coupling
    driver.set_controls(pump_speed_ratios={"injection_booster": 0.1})
    assert driver.step(864, current_time=0)
    assert abs(driver.report["water_rates_kg_s"]["injected"]) < 1e-9
    assert driver.report["water_rates_kg_s"]["makeup"] == 0
    assert max(map(abs, driver.report["material_account"]["residual_kg"])) < 0.01


def test_nonreturn_completion_blocks_only_the_reverse_perforation():
    c = configuration()
    c["wells"][1]["perforations"] = [[1, 0, 0], [2, 0, 0]]
    sim = simulation(c)
    s = sim.fim_solver
    p = s.fluid.pressure.flatten().clone()
    p[2] = 3.4e6
    v, d = s.cell_state(p, s.fluid.s_w.flatten(), s.x, s.sat, s.ug)
    s.wells.begin_step(v, d, 0)
    child = s.wells.injection
    q, dq, cv = child._conn_flows(v, d, p.new_tensor([3.2e6]))
    assert q[0, 1] < 0 and torch.count_nonzero(q[0, [0, 2]]) == 0
    assert (
        torch.count_nonzero(q[1]) == torch.count_nonzero(dq[1]) == torch.count_nonzero(cv[1]) == 0
    )
    assert not child.wellbore_mixing


def test_runner_preserves_event_clocks_and_independent_audit_rejects_forgery():
    from check_fim_waterflood import audit
    from run_fim_waterflood import run

    c = configuration()
    events = [
        {"time_days": 0, "controls": {}},
        {"time_days": 0.01, "controls": {"injection_enabled": False}},
        {"time_days": 0.02, "controls": {"injection_enabled": True}},
    ]
    result = run(c, 0.03, 0.01, events, save_fields=True)
    certificate = audit(c, result)
    assert certificate["accepted"] and certificate["operating_events"] == 3
    assert len(result["reservoir_snapshots"]) == 3
    saved = json.loads(json.dumps(result, allow_nan=False))
    saved["accepted_substeps"][-1]["coupling"]["water_rates_kg_s"]["makeup"] += 0.1
    with pytest.raises(ValueError):
        audit(c, saved)


def test_shared_facility_and_injection_power_budget():
    c = configuration()
    c["simulation"]["coupled_waterflood"]["max_total_drive_power_w"] = 5000
    sim = simulation(c)
    driver = sim.facility_coupling
    assert driver.step(864, current_time=0)
    assert driver.report["total_drive_power_w"] <= 5000
    assert driver.report["water_rates_kg_s"]["disposed"] > 0
    assert max(map(abs, driver.report["material_account"]["residual_kg"])) < 0.01


def test_npsh_map_limits_actual_injection():
    c = configuration()
    c["simulation"]["physical_surface_network"]["injection_network"]["network"]["pumps"][0][
        "npsh_curve"
    ]["values"] = [40.5, 60, 80]
    sim = simulation(c)
    driver = sim.facility_coupling
    assert driver.step(864, current_time=0)
    assert 0 < driver.report["water_rates_kg_s"]["injected"] < 0.01
    assert driver.report["water_rates_kg_s"]["makeup"] == 0
    assert driver.report["injection_network"]["feasible"]


def test_bhp_limit_releases_back_to_rate_when_capacity_is_available():
    c = configuration()
    c["wells"][1]["bhp_max"] = 3.05
    sim = simulation(c)
    driver = sim.facility_coupling
    s = sim.fim_solver
    assert driver.step(864, current_time=0)
    row = s.wells.surface_report["wells"]["I1"]
    assert row["mode"] == "bhp_limit"
    assert row["bhp_pa"] == pytest.approx(3.05e6, abs=10)
    s.wells.wells[1].bhp_max = 4
    assert driver.step(864, current_time=864)
    assert s.wells.surface_report["wells"]["I1"]["mode"] == "rate"


def test_dispatch_pressure_accuracy_refines_a_coarse_fraction_budget():
    c = configuration()
    c["simulation"]["coupled_waterflood"]["fraction_tolerance"] = 0.05
    c["simulation"]["coupled_waterflood"]["dispatch_bhp_tolerance_pa"] = 1
    sim = simulation(c)
    driver = sim.facility_coupling
    assert driver.step(864, current_time=0)
    bracket = driver.report["dispatch_resolution"]["injection"]
    assert bracket["maximum_common_bhp_gap_pa"] <= 1
    assert bracket["high_fraction"] - bracket["low_fraction"] < 1e-5
    assert driver.report["water_rates_kg_s"]["makeup"] <= 0.05 + 1e-12


def test_joint_dispatch_can_curtail_production_before_allocating_injection():
    c = configuration()
    c["simulation"]["coupled_waterflood"]["facility"]["gas_export"]["export_valve"][
        "throat_area_m2"
    ] = 2e-7
    sim = simulation(c)
    driver = sim.facility_coupling
    assert driver.step(864, current_time=0)
    assert 0 < driver.report["production_fraction"] < 1
    assert driver.report["facility"]["feasible"]
    assert driver.report["dispatch_resolution"]["production"]["maximum_common_bhp_gap_pa"] <= 10
    assert max(map(abs, driver.report["material_account"]["residual_kg"])) < 0.01
