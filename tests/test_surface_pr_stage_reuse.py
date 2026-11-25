"""Local SDIRK predictors/tangents retain residual and branch safeguards."""

import pytest
import torch
from test_surface_pr_surface_control import controlled

from surface.liquid import PropertyModelError
from surface.pr_collocation import PRCollocation, _sdirk_guess
from surface.pr_surface_collocation import PRSurfaceCollocation


@pytest.mark.parametrize("bad", [False, True])
def test_local_tangent_reuse_refreshes_bad_predictor_and_does_not_cross_steps(monkeypatch, bad):
    model, old = controlled(feedback=False)
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    solver.plant.engine._predict_sdirk_stage = False
    solver.plant.engine._reuse_sdirk_tangent = False
    expected, _ = solver._step(old, 0.001, {"v": 0.6}, {}, {"openings": {"port": False}})
    solver.plant.engine._reuse_sdirk_tangent = True
    engine = solver.plant.engine
    original = PRCollocation._solve
    histories = []

    def track(self, fn, *args, **kwargs):
        history = getattr(fn, "linearization_history", None)
        if history is not None:
            histories.append(history)
            if bad and history.entry is not None:
                history.entry["factors"] = history.entry["factors"] * 1000
        return original(self, fn, *args, **kwargs)

    monkeypatch.setattr(PRCollocation, "_solve", track)
    before = dict(engine._linear_work)
    result, _ = solver._step(old, 0.001, {"v": 0.6}, {}, {"openings": {"port": False}})
    assert model._error(expected, result) < 1e-5
    assert histories[0] is histories[1]
    if bad:
        assert engine._linear_work["cross_solve_refreshes"] > before["cross_solve_refreshes"]
    else:
        assert engine._linear_work["cross_solve_reuses"] > before["cross_solve_reuses"]
        assert (
            engine._linear_work["jacobian_builds"] - before["jacobian_builds"]
            < before["jacobian_builds"]
        )
    first = list(histories)
    histories.clear()
    solver._step(result, 0.001, {"v": 0.6}, {}, {"openings": {"port": False}})
    assert not any(a is b for a in first for b in histories)


@pytest.mark.parametrize("kind", ["linear", "nonlinear", "nonfinite", "invalid", "bound"])
def test_stage_predictor_is_only_a_residual_checked_bounded_guess(kind):
    initial = torch.tensor([0.0], dtype=torch.float64)
    first = initial + 0.2
    low, high = initial, initial + 1
    target = initial + 2 / 3
    function = lambda x: x - target
    if kind == "nonlinear":
        function = lambda x: (100 * x).exp() - torch.exp(initial + 19)
    baseline = function(first)
    if kind == "nonfinite":
        function = lambda x: x * torch.nan
    if kind == "invalid":

        def function(x):
            raise PropertyModelError("Predictor outside property domain")

    if kind == "bound":
        high = initial + 0.3
        target = high
        function = lambda x: x - target
        baseline = function(first)
    guess, status = _sdirk_guess(
        function, initial, first, 0.3, baseline, low, high, initial + torch.inf
    )
    assert bool(((guess >= low) & (guess <= high)).all())
    if kind in ("linear", "bound"):
        assert status == "accepted"
        assert function(guess).square().sum() <= 0.5 * baseline.square().sum()
    else:
        assert status == ("rejected" if kind == "nonlinear" else "inadmissible")
        torch.testing.assert_close(guess, first, rtol=0, atol=0)


def test_stage_reuse_and_predictor_keep_isolation_and_restart_budgets():
    model, old = controlled(feedback=True)
    end = []
    for enabled in (False, True):
        solver = PRSurfaceCollocation(model)
        engine = solver.plant.engine
        engine._reuse_sdirk_tangent = engine._predict_sdirk_stage = enabled
        state = old
        for opening in (True, False, True):
            result = solver.advance(state, 1e-6, openings={"port": opening})
            assert max(map(abs, result["conservation_residuals"])) < 1e-6
            state = type(old).from_dict(result["state"])
        end.append(state)
    assert model._error(*end) < 1e-4
