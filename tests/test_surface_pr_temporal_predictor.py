"""Fine states are guesses for discarded probes, never new conservative data."""

from dataclasses import replace

import pytest
from test_surface_pr_subcycled_probe import fixture

from surface.liquid import HydraulicConvergenceError
from surface.pr_gravity_transport import PRGravityProbeGuess
from surface.pr_surface_control import PRControlledSurfaceState


def test_only_coarse_probe_uses_predictor_and_fine_trajectory_is_unchanged():
    model, state = fixture()
    old = state.to_dict()
    model.slow.plant._reuse_spatial_predictor = True
    model._reuse_temporal_predictor = False
    reference = model.advance(state, 1e-8)
    model._reuse_temporal_predictor = True
    actual = model.advance(state, 1e-8)
    assert model.slow.plant.temporal_predictor_attempts == 1
    assert model.slow.plant.temporal_predictor_uses == 1
    assert model.slow.plant.temporal_vector_uses == 1
    assert model.slow.plant.spatial_vector_uses > 0
    a = PRControlledSurfaceState.from_dict(reference["state"])
    b = PRControlledSurfaceState.from_dict(actual["state"])
    assert a == b
    assert state.to_dict() == old
    for step in actual["steps"]:
        slow = step["local_subcycling"]["stages"][1]["report"]["plant"]["steps"]
        assert all(not s.get("temporal_predictor_used", False) for s in slow)
        assert all("nonlinear_predictor" not in s["transport"] for s in slow)
    for report in (reference, actual):
        assert max(map(abs, report["conservation_residuals"])) < 1e-5


@pytest.mark.parametrize("change", ["model", "clock", "phase", "phase_count", "support", "order"])
def test_changed_unknown_meanings_disable_predictor(change):
    model, state = fixture()
    plant, old = model.plant, state.plant
    dt = 1e-4
    order = old.columns[0][1].quadrature_order
    guess = replace(old, transport=replace(old.transport, time_s=old.time_s + dt))
    assert plant._compatible_temporal_predictor(old, guess, dt, order)
    if change == "model":
        guess = replace(guess, model_id="other-model")
    elif change == "clock":
        guess = replace(guess, transport=replace(guess.transport, time_s=old.time_s + 2 * dt))
    elif change == "order":
        order *= 2
    else:
        name, column = guess.columns[0]
        if change == "phase":
            root = "vapor" if column.phase_roots[0] == "liquid" else "liquid"
            column = replace(column, phase_roots=(root, *column.phase_roots[1:]))
            assert column.phase_roots != old.columns[0][1].phase_roots
        elif change == "phase_count":
            column = replace(
                column,
                phase_roots=column.phase_roots[:-1],
                phase_seeds=column.phase_seeds[:-1],
                interface_heights_m=column.interface_heights_m[:-1],
            )
        else:
            column = replace(column, component_moles=(0.0, *column.component_moles[1:]))
            assert old.columns[0][1].component_moles[0] > 0
        guess = replace(guess, columns=((name, column), *guess.columns[1:]))
    assert not plant._compatible_temporal_predictor(old, guess, dt, order)


def test_failed_warm_probe_retries_original_path_without_changing_fine_states(monkeypatch):
    model, state = fixture()
    model._reuse_temporal_predictor = True
    original = model._step
    calls = []

    def step(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("_probe_predictor") is not None:
            raise HydraulicConvergenceError("deliberately rejected probe guess")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_step", step)
    coarse, fine, one, two = model._temporal_trial(state, 1e-8, 5e-9, 5e-9, {}, {}, {})
    assert len(calls) == 4
    assert calls[0] == calls[1] == calls[3] == {}
    assert calls[2]["_probe_predictor"] is not None
    assert coarse.time_s == fine.time_s == 1e-8
    assert one["state"]["plant"]["transport"]["time_s"] == 5e-9
    assert two["state"] == fine.to_dict()


def test_uncontrolled_hint_cannot_bypass_temporal_qualification():
    model, state = fixture()
    with pytest.raises(ValueError, match="controlled discarded probe"):
        model.plant.advance(state.plant, 1e-8, _probe_predictor=state.plant)


@pytest.mark.parametrize("corruption", ["layout", "nonfinite"])
def test_unusable_spatial_vector_retains_original_conservative_solve(monkeypatch, corruption):
    model, state = fixture()
    plant = model.slow.plant
    plant._reuse_spatial_predictor = False
    reference = model.advance(state, 1e-8)
    plant._reuse_spatial_predictor = True
    original = PRGravityProbeGuess.from_endpoint

    def damaged(cls, endpoint, record):
        hint = original(endpoint, record)
        if not isinstance(hint, PRGravityProbeGuess):
            return hint
        if corruption == "layout":
            return replace(hint, layout="other", base_layout="other")
        return replace(hint, variables=(float("nan"),) * len(hint.variables))

    monkeypatch.setattr(PRGravityProbeGuess, "from_endpoint", classmethod(damaged))
    actual = model.advance(state, 1e-8)
    assert plant.spatial_vector_uses == 0
    assert actual["state"] == reference["state"]
    assert max(map(abs, actual["conservation_residuals"])) < 1e-5
