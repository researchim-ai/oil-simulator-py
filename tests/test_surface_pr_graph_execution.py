"""Dense AD selection and within-step reuse preserve coupled PR dynamics."""

import pytest
import torch
from test_surface_pr_surface_control import controlled

from surface.pr_surface_collocation import PRSurfaceCollocation, _ControlStages


@pytest.mark.parametrize("feedback", [False, True])
def test_dense_reverse_jacobian_matches_forward_and_finite_difference(feedback):
    model, state = controlled(feedback=feedback)
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    targets = {} if feedback else {"v": 0.6}
    controls = _ControlStages(model, state, targets, {}, {})
    prepared, _ = solver.plant._problem(
        state.plant, 2, {"port": True}, {}, {}, {}, _controls=controls
    )
    y, _, _, evaluate, rates, *_ = prepared
    y = y + torch.linspace(-1e-6, 1e-6, len(y), dtype=y.dtype)

    def residual(v):
        row, point = evaluate(v)
        return row - 1e-7 * rates(point)

    reverse = torch.func.jacrev(residual)(y)
    forward = torch.func.jacfwd(residual)(y)
    torch.testing.assert_close(reverse, forward, rtol=2e-9, atol=2e-9)
    direction = torch.linspace(-1e-3, 1e-3, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (residual(y + eps * direction) - residual(y - eps * direction)) / (2 * eps)
    torch.testing.assert_close(reverse @ direction, fd, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("feedback", [False, True])
def test_within_stage_reuse_preserves_full_step_report(feedback):
    model, old = controlled(feedback=feedback)
    solver = PRSurfaceCollocation(model, phase_fallback=False)
    engine = solver.plant.engine
    targets = {} if feedback else {"v": 0.6}
    engine._reuse_sdirk_points = False
    expected, old_report = solver._step(old, 1e-7, targets, {}, {})
    engine._reuse_sdirk_points = True
    actual, report = solver._step(old, 1e-7, targets, {}, {})
    assert actual.to_dict() == expected.to_dict()
    assert report == old_report


def test_dense_mode_and_endpoint_reuse_preserve_isolation_and_restart():
    model, old = controlled(feedback=True)
    states = []
    for mode, reuse in (("forward", False), ("reverse", True)):
        solver = PRSurfaceCollocation(model)
        solver.plant.engine._dense_cpu_jacobian_mode = mode
        solver.plant.engine._reuse_sdirk_points = reuse
        state = old
        for openings in ({"port": True}, {"port": False}, {"port": True}):
            result = solver.advance(state, 1e-7, openings=openings)
            assert max(map(abs, result["conservation_residuals"])) < 1e-6
            state = type(state).from_dict(result["state"])
        states.append(state)
    assert model._error(*states) < 1e-5
