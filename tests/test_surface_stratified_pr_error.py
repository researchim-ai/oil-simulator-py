"""Conservative BE embedded probes, event bootstrap and checkpoint safety."""

import copy
import json
from dataclasses import replace

import pytest
from test_surface_stratified_pr_network import setup

from surface.stratified_pr_flow import StratifiedPRFlow
from surface.stratified_pr_network import StratifiedNetworkState, StratifiedPRNetwork
from surface.vessel_wall import VesselWall


def embedded():
    original, state = setup()
    config = original.to_dict()
    config["temporal_error_control"] = "embedded_be"
    model = StratifiedPRNetwork.from_dict(config)
    return model, model.initialize(dict(state.vessels))


def test_embedded_bootstraps_and_keeps_conservative_steps():
    from check_stratified_pr_network import audit

    model, state = embedded()
    report = model.advance(state, 0.06)
    assert report["embedded_steps"] > 0
    assert report["step_doubling_steps"] >= 1
    assert report["steps"][0]["temporal_error_method"] == "step_doubling"
    assert all(0 <= s["temporal_error_ratio"] <= 1 for s in report["steps"])
    config = {
        "vessels": {name: {"model": v.to_dict()} for name, v in model.vessels.items()},
        "connections": [p.to_dict() for p in model.connections],
        "temporal_error_control": "embedded_be",
        "schedule": [{"duration_s": 0.06}],
    }
    payload = json.loads(
        json.dumps({"config": config, "initial": state.to_dict(), "intervals": [report]})
    )
    assert audit(config, payload)["accepted"]
    broken = copy.deepcopy(payload)
    broken["intervals"][0]["steps"][-1]["temporal_error_ratio"] = 1.01
    with pytest.raises(ValueError, match="temporal error"):
        audit(config, broken)


def test_control_event_and_cold_restart_invalidate_endpoint_history():
    model, state = embedded()
    first = model.advance(state, 0.04)
    current = StratifiedNetworkState.from_dict(first["state"])
    stopped = model.advance(current, 0.02, openings={"link": 0.0})
    assert stopped["step_doubling_steps"] >= 1
    restart = StratifiedPRNetwork.from_dict(model.to_dict())
    restored = StratifiedNetworkState.from_dict(stopped["state"])
    after = restart.advance(restored, 0.02, openings={"link": 0.0})
    assert after["step_doubling_steps"] >= 1
    assert after["state"]["time_s"] == pytest.approx(0.08)


def test_embedded_converges_under_time_refinement():
    model, state = embedded()
    ordinary, old = setup()
    reference = ordinary.advance(old, 0.06)
    result = model.advance(state, 0.06)
    error = model._error(
        dict(StratifiedNetworkState.from_dict(result["state"]).vessels),
        dict(StratifiedNetworkState.from_dict(reference["state"]).vessels),
    )
    assert error <= 1.0
    fine = model.to_dict()
    fine["maximum_step_s"] = 0.005
    fine["initial_step_s"] = 0.005
    refined = StratifiedPRNetwork.from_dict(fine)
    refined_state = refined.initialize(dict(state.vessels))
    refined_report = refined.advance(refined_state, 0.06)
    difference = model._error(
        dict(StratifiedNetworkState.from_dict(result["state"]).vessels),
        dict(StratifiedNetworkState.from_dict(refined_report["state"]).vessels),
    )
    assert difference <= 1.0


def test_newton_reuses_linearization_without_relaxing_residual():
    model, state = embedded()
    _final, report = model._step(state, 0.02, model._commands(None))
    assert any(a["jacobian_reuses"] > 0 for a in report["solver_attempts"])
    assert all(a["max_scaled_residual"] < 1e-9 for a in report["solver_attempts"])


def test_tangent_history_reuses_only_compatible_controls_and_is_bounded():
    model, state = embedded()
    first, _one = model._step(state, 0.005, model._commands(None))
    second, two = model._step(first, 0.005, model._commands(None))
    assert any(a["cross_solve_reuses"] > 0 for a in two["solver_attempts"])
    assert len(model._tangent_history) <= 2
    _stopped, closed = model._step(second, 0.005, model._commands({"link": 0.0}))
    assert all(a["cross_solve_reuses"] == 0 for a in closed["solver_attempts"])


def test_unknown_temporal_scheme_is_rejected():
    model, _state = setup()
    bad = model.to_dict()
    bad["temporal_error_control"] = "unchecked"
    with pytest.raises(ValueError, match="temporal error control"):
        StratifiedPRNetwork.from_dict(bad)


def test_thermal_wall_companion_matches_refinement_and_energy_ledger():
    original, state = setup()
    column = original.vessels["a"].column
    vessel = StratifiedPRFlow(
        column, VesselWall(1000.0, 100.0, ambient_ua_w_k=20.0, ambient_temperature_k=340.0)
    )

    def simulate(step):
        model = StratifiedPRNetwork(
            {"a": vessel},
            initial_step_s=step,
            maximum_step_s=step,
            temporal_error_control="embedded_be",
        )
        start = model.initialize({"a": vessel.initialize(dict(state.vessels)["a"].column, 330.0)})
        report = model.advance(start, 0.02)
        assert abs(report["energy_residual_j"]) < 1e-6
        assert report["embedded_steps"] > 0
        return model, StratifiedNetworkState.from_dict(report["state"])

    model, coarse = simulate(0.005)
    _fine_model, fine = simulate(0.001)
    assert model._error(dict(coarse.vessels), dict(fine.vessels)) <= 1.0


def test_failed_probe_reuses_primary_solve_for_step_doubling(monkeypatch):
    import surface.stratified_pr_error as error_module

    model, state = embedded()
    first = model.advance(state, 0.04)
    current = StratifiedNetworkState.from_dict(first["state"])
    monkeypatch.setattr(error_module, "temporal_probe", lambda *args, **kwargs: None)
    actual_step = model._step
    calls = []

    def step(*args, **kwargs):
        calls.append(1)
        return actual_step(*args, **kwargs)

    monkeypatch.setattr(model, "_step", step)
    report = model.advance(current, 0.001)
    assert report["embedded_probe_failures"] == 1
    assert len(calls) == 3
    assert report["embedded_steps"] == 0


def test_phase_disappearance_in_first_half_requires_macro_bootstrap():
    from surface.stratified_pr_error import temporal_probe

    original, state = setup(two=True)
    vessels = {
        name: StratifiedPRFlow(v.column, VesselWall(1e6, 1000.0, max_temperature_change_k=100.0))
        for name, v in original.vessels.items()
    }
    model = StratifiedPRNetwork(vessels, [replace(original.connections[0], opening=0.0)])
    before = model.initialize(
        {name: vessels[name].initialize(child.column, 320.0) for name, child in state.vessels}
    )
    midpoint, _report = model._step(before, 100.0, model._commands(None))
    assert all(len(child.column.phase_roots) == 1 for _, child in midpoint.vessels)
    assert any(len(child.column.phase_roots) == 2 for _, child in before.vessels)
    # Second half may stay single-phase: checking only its two endpoints
    # misses the phase event that happened in the FIRST half of the macrostep.
    assert (
        temporal_probe(
            model,
            midpoint,
            midpoint,
            1.0,
            model._commands(None),
            None,
            model._last_step_probe,
            macro_start=before,
        )
        is None
    )
