"""No repeated capacity points, solved-state commit and pressure-safe proposals."""

import copy

import pytest
import torch
from test_fim_waterflood import configuration, simulation

from surface.waterflood_search import resolve_boundary


def test_same_bisection_restored_state_matches_replay_and_well_cumulatives():
    cfg = configuration()
    spec = cfg["simulation"]["coupled_waterflood"]
    spec["dispatch_solver"] = {"method": "bisect", "commit": "replay", "reuse_initial_cell_state": False}
    old = simulation(cfg).facility_coupling
    spec["dispatch_solver"] = {"method": "bisect"}
    new = simulation(cfg).facility_coupling
    assert old.step(864, current_time=0) and new.step(864, current_time=0)
    for field in ["pressure", "s_w", "s_g"]:
        torch.testing.assert_close(getattr(old.solver.fluid, field), getattr(new.solver.fluid, field),
                                   rtol=2e-13, atol=2e-12)
    assert old.report["injection_fraction"] == new.report["injection_fraction"]
    assert new.last_attempt["reservoir_evaluations"] == old.last_attempt["reservoir_evaluations"] - 1
    for a, b in zip(old.solver.wells.wells, new.solver.wells.wells):
        for attr in ["cum_surface", "cum_prod", "cum_inj"]:
            assert getattr(a, attr) == pytest.approx(getattr(b, attr), rel=2e-13, abs=2e-12)
    report = new.report
    assert len({(r["production_fraction"], r["injection_fraction"]) for r in report["trace"]}) == len(report["trace"])
    cache = report["initial_cell_state_cache"]
    assert cache["evaluations"] == 1 and cache["hits"] == len(report["trace"]) - 1


def test_safeguarded_water_resource_uses_fewer_real_fim_solves_without_relaxed_bhp():
    cfg = configuration()
    spec = cfg["simulation"]["coupled_waterflood"]
    spec["dispatch_bhp_tolerance_pa"] = .01
    spec["dispatch_solver"] = {"method": "bisect", "commit": "replay", "reuse_initial_cell_state": False}
    old = simulation(cfg).facility_coupling
    spec["dispatch_solver"] = {}
    new = simulation(cfg).facility_coupling
    assert old.step(864, current_time=0) and new.step(864, current_time=0)
    assert new.last_attempt["reservoir_evaluations"] < old.last_attempt["reservoir_evaluations"] / 2
    assert abs(new.solver.wells.p_bh - old.solver.wells.p_bh).max() <= .02
    assert new.report["dispatch_resolution"]["injection"]["maximum_common_bhp_gap_pa"] <= .01
    assert new.report["water_rates_kg_s"]["makeup"] <= .05 + 1e-12


def test_cache_budget_fallback_and_audit_rejects_wrong_commit_evidence():
    from check_fim_waterflood import audit
    from run_fim_waterflood import run
    cfg = configuration()
    cfg["simulation"]["coupled_waterflood"]["dispatch_solver"] = {"initial_cell_state_cache_bytes": 1}
    result = run(cfg, .01, .01)
    assert audit(cfg, result)["accepted"]
    report = result["accepted_substeps"][0]["coupling"]
    stats = report["initial_cell_state_cache"]
    assert stats["bypassed"] and stats["hits"] == stats["stored_bytes"] == 0
    forged = copy.deepcopy(result)
    forged["accepted_substeps"][0]["coupling"]["dispatch_commit"]["solved_trace_index"] = -1
    with pytest.raises(ValueError, match="state reference"):
        audit(cfg, forged)


@pytest.mark.parametrize("opaque", [False, True])
def test_boundary_stops_on_actual_bhp_span_and_handles_zero_anchor(opaque):
    calls = []
    def data(x):
        return {"well_bhp_pa": {"I": 3e6 + x * 1e7},
                "constraint_margins": {} if opaque else {"water": x * x - .01}}
    def evaluate(x):
        assert x not in calls
        calls.append(x)
        return x <= .1, data(x)
    low, high, a, b = resolve_boundary(evaluate, 0, 1, data(0), data(1), 1e-4, .01)
    assert low <= .1 < high
    assert high - low <= 1e-4
    assert abs(a["well_bhp_pa"]["I"] - b["well_bhp_pa"]["I"]) <= .01
    assert len(calls) < 90


@pytest.mark.parametrize("options", [{"method": "invalid"}, {"commit": "scale"},
                                     {"reuse_initial_cell_state": 1},
                                     {"initial_cell_state_cache_bytes": True}])
def test_invalid_dispatch_policy_rejected(options):
    cfg = configuration()
    cfg["simulation"]["coupled_waterflood"]["dispatch_solver"] = options
    with pytest.raises(ValueError):
        simulation(cfg)
