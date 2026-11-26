"""Actual waterflood half-stage accounting, transactional rejection and error identity."""

import copy

import pytest
import torch
from test_fim_waterflood import simulation
from test_waterflood_multidepth import configuration_3d


def configuration(**options):
    c = configuration_3d()
    c["simulation"]["coupled_waterflood"]["temporal_control"] = {
        "startup_step_days": .01, "well_bhp_tolerance_pa": 1000., **options,
    }
    c["simulation"].update(newton_tol_cnv=1e-9, newton_tol_mb=1e-13)
    return c


def test_two_actual_half_stages_match_explicit_coupled_reference():
    c = configuration()
    controlled = simulation(c).facility_coupling
    del c["simulation"]["coupled_waterflood"]["temporal_control"]
    reference = simulation(c).facility_coupling
    assert controlled.step(864, current_time=0)
    assert reference.step(432, current_time=0)
    assert reference.step(432, current_time=432)
    a, b = controlled.solver, reference.solver
    for x, y in [(a.fluid.pressure, b.fluid.pressure), (a.fluid.s_w, b.fluid.s_w),
                 (a.fluid.s_g, b.fluid.s_g), (a.wells.p_bh, b.wells.p_bh)]:
        # Temporal mode also tightens the algebraic solve. States agree to
        # floating-point precision rather than requiring identical Krylov paths.
        torch.testing.assert_close(x, y, rtol=2e-13, atol=2e-12)
    assert len(controlled.accepted_stages) == 2
    assert [s["dt_s"] for s in controlled.accepted_stages] == [432, 432]
    assert controlled.report["temporal_control"]["indicator"] <= 1
    assert controlled.temporal_work["coupled_solves"] == 3
    for left, right in zip(a.wells.wells, b.wells.wells):
        for attr in ["cum_surface", "cum_prod", "cum_inj"]:
            assert getattr(left, attr) == pytest.approx(getattr(right, attr), rel=2e-13, abs=2e-12)
        for j, phase in enumerate(["oil", "water", "gas"]):
            actual = sum(stage["surface"]["wells"][left.name]["component_mass_rates_kg_s"][j]
                         * stage["dt_s"] / float(a.rho_sc[j]) for stage in controlled.accepted_stages)
            assert left.cum_surface[phase] == pytest.approx(actual, rel=2e-13, abs=2e-12)


def test_temporal_rejection_restores_parent_children_reports_caps_and_tolerances():
    driver = simulation(configuration(pressure_tolerance_pa=1e-12)).facility_coupling
    s = driver.solver
    fields = [s.fluid.pressure.clone(), s.fluid.s_w.clone(), s.fluid.s_g.clone()]
    children = [(w.p_bh, w.conn_sched.clone(), w.injection_report if hasattr(w, "injection_report") else w.surface_report)
                for w in s.wells.subwell_models]
    names = ("tol_cnv", "tol_mb", "min_newton", "lin_tol_max", "lin_tol_min",
             "lin_accept", "relaxed_max_pv_fraction", "tol_well_bhp")
    settings = {name: getattr(s, name) for name in names}
    assert not driver.step(864, current_time=0)
    assert driver.last_attempt["temporal_errors"]["pressure"] > 1
    assert driver.report is None and driver.accepted_stages == ()
    for old, new in zip(fields, [s.fluid.pressure, s.fluid.s_w, s.fluid.s_g]):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    for w, (pb, schedule, report) in zip(s.wells.subwell_models, children):
        assert w.p_bh is pb
        torch.testing.assert_close(w.conn_sched, schedule, rtol=0, atol=0)
        assert (w.injection_report if hasattr(w, "injection_report") else w.surface_report) is report
    assert [w.control_value for w in s.wells.wells] == [100, 100, 100]
    assert settings == {name: getattr(s, name) for name in names}


def test_exception_on_second_fine_stage_publishes_nothing(monkeypatch):
    driver = simulation(configuration()).facility_coupling
    original = driver._step_once
    count = 0
    p = driver.solver.fluid.pressure.clone()
    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise ValueError("bad equipment data")
        return original(*args, **kwargs)
    monkeypatch.setattr(driver, "_step_once", fail)
    with pytest.raises(ValueError, match="equipment"):
        driver.step(864, current_time=0)
    torch.testing.assert_close(p, driver.solver.fluid.pressure, rtol=0, atol=0)
    assert driver.report is None and not driver.accepted_stages


def test_internal_recycle_error_cannot_cancel_between_wells(monkeypatch):
    driver = simulation(configuration()).facility_coupling
    monkeypatch.setattr(driver, "_transfers", lambda stage: torch.zeros(11, dtype=torch.float64))
    def stage(value, dt):
        return {
            "dt_s": dt, "well_order": ["P", "I"],
            "withdrawn_component_mass_kg": [[0, value, 0], [0, -value, 0]],
            "surface": {"wells": {n: {"bhp_pa": 3e6} for n in ["P", "I"]}},
            "facility_coupling": {"total_drive_power_w": 1},
        }
    driver.report = {}
    fields = {"pressure": torch.ones(1), "s_w": torch.ones(1), "s_g": torch.zeros(1),
              "component_mass": torch.ones(3, 1)}
    errors = driver._temporal_defect(fields, fields, stage(10, 2), stage(2, 1), stage(3, 1))
    assert errors["withdrawal"] > 1
    assert errors["resource_transfers"] == 0


def test_runner_emits_fine_stages_hits_event_and_audit_rejects_forged_evidence():
    from check_fim_waterflood import audit
    from run_fim_waterflood import run
    c = configuration()
    events = [{"time_days": .01, "controls": {"injection_enabled": False}}]
    result = run(c, .02, .01, events, save_fields=True)
    assert audit(c, result)["accepted"]
    steps = result["accepted_substeps"]
    assert len(steps) >= 4 and len(steps) % 2 == 0
    assert sum(s["coupling"]["dt_s"] for s in steps) == pytest.approx(.02 * 86400)
    assert any(s["time_days"] == .01 for s in steps)
    for change in ["missing", "error", "work", "probe", "nan_probe"]:
        forged = copy.deepcopy(result)
        if change == "missing":
            forged["accepted_substeps"].pop(0)
        elif change == "error":
            forged["accepted_substeps"][1]["coupling"]["temporal_control"]["errors"]["pressure"] = 2
        elif change == "work":
            forged["accepted_substeps"][1]["coupling"]["temporal_control"]["total_capacity_evaluations"] = 0
        elif change == "probe":
            forged["accepted_substeps"][1]["coupling"]["temporal_probe"]["coarse_transfers_kg"][1] += 1
        else:
            forged["accepted_substeps"][1]["coupling"]["temporal_probe"]["coarse_transfers_kg"][1] = float("nan")
        with pytest.raises(ValueError):
            audit(c, forged)


def test_startup_event_restart_and_dispatch_resolution_validation():
    driver = simulation(configuration()).facility_coupling
    assert driver.limit_initial_time_step_s(86400) == 864
    driver.report = {"operator_controls": copy.deepcopy(driver.controls),
                     "temporal_control": {"macro_dt_s": 100, "indicator": .25}}
    assert driver.limit_next_time_step_s(1000) == pytest.approx(180)
    driver.set_controls(injection_enabled=False)
    assert driver.limit_next_time_step_s(86400) == 864
    driver.last_attempt = {"temporal_errors": {"pressure": 100}}
    assert driver.limit_rejected_time_step_s(500, failed_dt_s=1000) == 100
    with pytest.raises(ValueError, match="Dispatch BHP"):
        simulation(configuration(well_bhp_tolerance_pa=1))


def test_refinement_checker_rejects_changed_physics_and_relaxed_time_policy():
    from compare_fim_waterflood import check_time_refinement
    a = configuration()
    b = copy.deepcopy(a)
    b["simulation"]["coupled_waterflood"]["temporal_control"]["well_bhp_tolerance_pa"] = 500
    assert check_time_refinement([a, b]) == "temporal_tolerances_and_maximum_step"
    b["reservoir"]["permeability"] = 101
    with pytest.raises(ValueError, match="Only stricter"):
        check_time_refinement([a, b])
    b = copy.deepcopy(a)
    b["simulation"]["coupled_waterflood"]["temporal_control"]["well_bhp_tolerance_pa"] = 2000
    with pytest.raises(ValueError, match="must not be relaxed"):
        check_time_refinement([a, b])


def test_restart_preserves_temporal_history_and_next_waterflood_solution():
    import pickle
    driver = simulation(configuration()).facility_coupling
    assert driver.step(432, current_time=0)
    restored = pickle.loads(pickle.dumps(driver))
    assert restored.limit_next_time_step_s(864) == driver.limit_next_time_step_s(864)
    assert driver.step(432, current_time=432)
    assert restored.step(432, current_time=432)
    for name in ["pressure", "s_w", "s_g"]:
        torch.testing.assert_close(getattr(driver.solver.fluid, name),
                                   getattr(restored.solver.fluid, name), rtol=0, atol=0)
    assert restored.report == driver.report
    assert restored.temporal_work == driver.temporal_work


def test_local_component_redistribution_cannot_hide_in_total_inventory(monkeypatch):
    driver = simulation(configuration()).facility_coupling
    monkeypatch.setattr(driver, "_transfers", lambda stage: torch.zeros(11, dtype=torch.float64))
    def stage(dt):
        return {"dt_s": dt, "well_order": ["P", "I"],
                "withdrawn_component_mass_kg": [[0, 0, 0], [0, 0, 0]],
                "surface": {"wells": {n: {"bhp_pa": 3e6} for n in ["P", "I"]}},
                "facility_coupling": {"total_drive_power_w": 1}}
    driver.report = {}
    coarse = {"pressure": torch.ones(2), "s_w": torch.ones(2), "s_g": torch.zeros(2),
              "component_mass": torch.ones(3, 2, dtype=torch.float64)}
    fine = copy.deepcopy(coarse)
    fine["component_mass"][2] = torch.tensor([.5, 1.5])
    torch.testing.assert_close(coarse["component_mass"].sum(1), fine["component_mass"].sum(1))
    errors = driver._temporal_defect(coarse, fine, stage(2), stage(1), stage(1))
    assert errors["component_inventory"] > 1
    assert errors["withdrawal"] == errors["resource_transfers"] == 0
