"""Rate-change proposals and atomic rejection on the actual coupled FIM."""

import copy

import pytest
import torch
from test_fim_facility_coupling import build

from simulator.facility_time_control import FacilityTimeControl


def test_rate_indicator_and_short_clipped_tail():
    c = FacilityTimeControl(rate_relative_tolerance=0.01, rate_absolute_tolerance_kg_s=1e-8)
    assert c.indicator([1, 2, 0], [1, 2, 0]) == 0
    assert c.indicator([1], [1.1]) > 1
    assert c.limit(1e6) == 86400
    assert c.limit(86400, 1e-4, 1e-10, clipped=True) == 86400
    assert c.limit(86400, 1e-4, 1e-10, clipped=False) == 2e-4
    with pytest.raises(ValueError):
        c.indicator([1, float("nan")], [1, 2])
    with pytest.raises(ValueError):
        c.limit(0)


@pytest.mark.parametrize(
    "kw",
    [
        {"rate_relative_tolerance": 0},
        {"startup_step_days": float("inf")},
        {"maximum_growth": 0.5},
        {"safety_factor": 1},
    ],
)
def test_invalid_time_policy(kw):
    with pytest.raises(ValueError):
        FacilityTimeControl(**kw)


def test_production_rate_rejection_restores_material_and_report():
    c = build(False)
    assert c.limit_next_time_step_s is None
    c.time_control = FacilityTimeControl(rate_relative_tolerance=1e-8)
    assert c.limit_initial_time_step_s(864000) == 86400
    assert c.step(3600)
    old = c.solver.fluid.pressure.clone()
    report = copy.deepcopy(c.report)
    assert not c.step(86400)
    assert "rate-change indicator" in c.last_attempt["reason"]
    assert torch.equal(c.solver.fluid.pressure, old)
    assert c.report == report
    assert [w.control_value for w in c.solver.wells.wells] == [100, 100]


def test_event_restarts_step_bound_before_any_reservoir_solve(monkeypatch):
    c = build(False)
    c.time_control = FacilityTimeControl(startup_step_days=0.5)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Reservoir must not run for an oversized startup step")

    monkeypatch.setattr(c.solver, "step", forbidden)
    assert not c.step(86400)
    assert c.last_attempt["trace"] == []
    assert c.report is None


def test_simulator_initial_hook_and_independent_rate_control_audit(tmp_path):
    from check_fim_facility_forecast import audit
    from run_fim_blackoil_surface import run_case
    from test_fim_facility_forecast import config

    c = config()
    c["simulation"]["coupled_facility"]["time_control"] = {"startup_step_days": 0.25}
    events = [{"time_days": 0.5, "controls": {"gas_valve_max_opening": 0.8}}]
    r = run_case(c, 1e6, 1, 1, facility_schedule=events)
    assert audit(c, r, expected_schedule=events, expected_days=1)["accepted"]
    assert r["accepted_substeps"][0]["dt_s"] == 0.25 * 86400
    assert sum(s["time_step_statistics"]["rejected_steps"] for s in r["interval_timings"]) == 0
    broken = copy.deepcopy(r)
    broken["accepted_substeps"][1]["facility_coupling"]["rate_time_control"]["indicator"] = 999
    with pytest.raises(ValueError):
        audit(c, broken, expected_schedule=events, expected_days=1)
