"""Hierarchy reuse must preserve equations and retry from unmodified assembly."""

import copy
import pickle
from dataclasses import replace

import pytest
import torch
from test_fim_facility_coupling import build, masses

from simulator.capacity_amg_reuse import CapacityAMGReuse


def close_state(a, b):
    torch.testing.assert_close(a.solver.fluid.pressure, b.solver.fluid.pressure, rtol=0, atol=1e-4)
    torch.testing.assert_close(a.solver.fluid.s_w, b.solver.fluid.s_w, rtol=0, atol=1e-10)
    torch.testing.assert_close(a.solver.fluid.s_g, b.solver.fluid.s_g, rtol=0, atol=1e-10)
    assert a.report["fraction"] == b.report["fraction"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_events_restart_and_search_scope(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    a, b = build(device=device), build(device=device)
    b.control = replace(b.control, reuse_capacity_amg=True)
    for time, controls in ((0.0, {}), (60.0, {"gas_valve_max_opening": 0.8}), (120.0, {})):
        for c in (a, b):
            c.controls = copy.deepcopy(controls)
            assert c.step(60.0, current_time=time)
            assert not hasattr(c.solver, CapacityAMGReuse.binding_name)
        close_state(a, b)
        work = b.report["capacity_amg_work"]
        assert work["fresh_requests"] >= 1
        assert work["builds"] >= 1
        assert work["linear_attempts"] >= work["reuse_requests"] + work["fresh_requests"]
        a, b = [pickle.loads(pickle.dumps(c)) for c in (a, b)]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_failed_reused_linear_solve_reassembles_before_fresh_retry(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    a, b = build(device=device), build(device=device)
    b.control = replace(b.control, reuse_capacity_amg=True)
    original = b.solver.solve_linear
    injected = []

    def sometimes_fail(J, R, wsys, **kwargs):
        answer = original(J, R, wsys, **kwargs)
        scope = getattr(b.solver, CapacityAMGReuse.binding_name, None)
        if (
            scope is not None
            and scope.reuse_requests
            and not injected
            and not kwargs["rebuild_amg"]
        ):
            # The real call above has already Schur-modified and scaled J.
            injected.append(True)
            return None
        return answer

    monkeypatch.setattr(b.solver, "solve_linear", sometimes_fail)
    assert a.step(60.0) and b.step(60.0)
    assert injected and b.report["capacity_amg_work"]["retries"] == 1
    close_state(a, b)


def test_failure_releases_scope_and_restores_state(monkeypatch):
    c = build()
    c.control = replace(c.control, reuse_capacity_amg=True)
    before = masses(c.solver).clone()

    def fail(*a):
        raise ValueError("injected commit failure")

    monkeypatch.setattr(c, "_material_account", fail)
    with pytest.raises(ValueError):
        c.step(60.0)
    assert not hasattr(c.solver, CapacityAMGReuse.binding_name)
    torch.testing.assert_close(masses(c.solver), before, rtol=0, atol=0)


def test_explicit_iteration_rebuild_and_invalid_flag():
    c = build()
    c.control = replace(c.control, reuse_capacity_amg=True)
    c.solver.amg_rebuild = "iteration"
    assert c.step(60.0)
    assert c.report["capacity_amg_work"]["reuse_requests"] == 0
    with pytest.raises(ValueError):
        replace(c.control, reuse_capacity_amg=1)


def test_previous_time_hierarchy_is_not_admitted_without_a_fresh_trial():
    class Solver:
        _amg = object()
        amg_rebuild = "step"

    s = Solver()
    scope = CapacityAMGReuse(s, 60.0, 0.0)
    assert not scope.reuse(s, 60.0, 0.0, 0)
    scope.ready = True
    assert scope.reuse(s, 60.0, 0.0, 0)
    assert not CapacityAMGReuse(s, 61.0, 0.0).reuse(s, 61.0, 0.0, 0)
    for owner, dt, time in ((Solver(), 60.0, 0.0), (s, 61.0, 0.0), (s, 60.0, 1.0)):
        with pytest.raises(RuntimeError):
            scope.reuse(owner, dt, time, 0)
    with scope.installed(), pytest.raises(RuntimeError), scope.installed():
        pass
    assert not hasattr(s, scope.binding_name)


@pytest.mark.parametrize("key", ["reuse_requests", "builds", "linear_attempts"])
def test_offline_audit_rejects_impossible_hierarchy_work(key):
    from simulator.capacity_amg_reuse import audit_capacity_amg

    c = {
        "evaluations": 5,
        "capacity_amg_work": {
            "fresh_requests": 1,
            "reuse_requests": 4,
            "retries": 0,
            "builds": 1,
            "updates": 9,
            "linear_attempts": 10,
        },
    }
    policy = {"reuse_capacity_amg": True}
    assert audit_capacity_amg(c, policy)["builds"] == 1
    c["capacity_amg_work"][key] = 99 if key != "linear_attempts" else 0
    with pytest.raises(ValueError):
        audit_capacity_amg(c, policy)


def test_temporal_forecast_checks_both_accepted_and_discarded_hierarchy_work():
    import json
    from pathlib import Path

    from check_fim_facility_forecast import audit
    from run_fim_blackoil_surface import run_case

    root = Path(__file__).resolve().parents[1]
    c = json.loads((root / "configs/fim_blackoil_facility_depletion_fast.json").read_text())
    c["fluid"]["pvt_path"] = str(root / c["fluid"]["pvt_path"])
    c["simulation"]["coupled_facility"]["control"]["reuse_capacity_amg"] = True
    result = run_case(c, 1e6, 0.00005, 0.00005, "cpu")
    assert audit(c, result, expected_days=0.00005, expected_schedule=[])["accepted"]
    work = result["temporal_work"]["capacity_amg_work"]
    assert work["builds"] == 3 and work["reuse_requests"] > 0
    corrupt = copy.deepcopy(result)
    corrupt["temporal_work"]["capacity_amg_work"]["linear_attempts"] = 0
    with pytest.raises(ValueError):
        audit(c, corrupt, expected_days=0.00005, expected_schedule=[])
    corrupt = copy.deepcopy(result)
    corrupt["accepted_substeps"][1]["facility_coupling"]["temporal_control"][
        "total_capacity_amg_work"
    ]["builds"] = 0
    with pytest.raises(ValueError):
        audit(c, corrupt, expected_days=0.00005, expected_schedule=[])


@pytest.mark.parametrize("mutation", ["physics", "tolerance", "search", "flag"])
def test_pair_checker_cannot_hide_other_policy_changes(mutation):
    from check_facility_amg_reuse import check_policy_pair

    a = {
        "simulation": {"newton_tol_mb": 1e-12, "coupled_facility": {"control": {"scan_points": 3}}},
        "reservoir": {"permeability": 100},
    }
    b = copy.deepcopy(a)
    b["simulation"]["coupled_facility"]["control"]["reuse_capacity_amg"] = True
    check_policy_pair(a, b)
    if mutation == "physics":
        b["reservoir"]["permeability"] = 101
    elif mutation == "tolerance":
        b["simulation"]["newton_tol_mb"] = 1e-7
    elif mutation == "search":
        b["simulation"]["coupled_facility"]["control"]["scan_points"] = 4
    else:
        b["simulation"]["coupled_facility"]["control"]["reuse_capacity_amg"] = 1
    with pytest.raises(ValueError, match="Only"):
        check_policy_pair(a, b)
