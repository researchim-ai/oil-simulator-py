"""Two independent PR stages share execution, never a thermodynamic state."""

import pytest
import torch
from check_pr_collocation import thermal_fixture
from test_surface_pr_open_transport import build

from surface import PRCollocation
from surface.pr_open_transport import _BoundaryProperties, _OpenCoupling


@pytest.mark.parametrize("scheme", ["gauss4", "radau3"])
@pytest.mark.parametrize("opened", [False, True])
def test_stage_residual_and_ad_agree_without_scalar_fallback(scheme, opened, monkeypatch):
    if opened:
        owner, state = build(reverse=True, temperature=196.2)
        model = owner.transport
        coupling = _OpenCoupling(
            owner,
            _BoundaryProperties(owner, {"boundary": 1.0}),
            state,
            1e-5,
            model.default_openings,
        )
    else:
        model, state = thermal_fixture()
        coupling = None
    solver = PRCollocation(model, scheme=scheme)
    heat = [0.0] * len(model.nodes)
    y, lo, hi, evaluate, rates, *_ = solver._problem(
        state, model.default_openings, heat, {}, coupling
    )
    size = len(y)
    # Deliberately different stage P/T/compositions, inside the branch bounds.
    direction = torch.linspace(-0.002, 0.003, size, dtype=y.dtype)
    z = torch.cat(
        (
            torch.minimum(hi, torch.maximum(lo, y + direction)),
            torch.minimum(hi, torch.maximum(lo, y - direction)),
        )
    )
    fn = lambda v: solver._stage_rows(v, size, 1e-5, evaluate, rates)
    solver._batch_stage_residual = False
    expected = fn(z)
    jac = torch.func.jacfwd(fn)(z)
    original = solver._stage_rows
    solver._batch_stage_residual = True

    # Scalar execution would receive ordinary arguments, whereas vmap passes
    # a BatchedTensor, also inside forward-mode Jacobian transforms.
    def only_batched(v):
        assert torch._C._functorch.is_batchedtensor(v)
        return evaluate(v)

    actual = original(z, size, 1e-5, only_batched, rates)
    actual_jac = torch.func.jacfwd(lambda v: original(v, size, 1e-5, only_batched, rates))(z)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-13)
    torch.testing.assert_close(actual_jac, jac, atol=2e-12, rtol=2e-12)
    eps = 1e-5
    perturb = torch.linspace(-0.001, 0.002, len(z), dtype=z.dtype)
    fd = (fn(z + eps * perturb) - fn(z - eps * perturb)) / (2 * eps)
    torch.testing.assert_close(actual_jac @ perturb, fd, atol=2e-7, rtol=2e-5)


def test_unsupported_batch_uses_original_operator_and_original_error(monkeypatch):
    model, _ = thermal_fixture()
    solver = PRCollocation(model)
    calls = []

    def evaluate(v):
        calls.append(1)
        return v.square(), v

    def unsupported(*args, **kwargs):
        raise RuntimeError("unsupported transform")

    monkeypatch.setattr(torch, "vmap", unsupported)
    z = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    actual = solver._stage_rows(z, 2, 0.1, evaluate, lambda p: p)
    solver._batch_stage_residual = False
    torch.testing.assert_close(
        actual, solver._stage_rows(z, 2, 0.1, evaluate, lambda p: p), atol=0, rtol=0
    )
    assert len(calls) == 4
    solver._batch_stage_residual = True

    def broken(v):
        raise RuntimeError("original physical failure")

    with pytest.raises(RuntimeError, match="original physical failure"):
        solver._stage_rows(z, 2, 0.1, broken, lambda p: p)


@pytest.mark.parametrize("scheme", ["gauss4", "radau3"])
def test_subminimum_report_remainder_does_not_destroy_next_proposal(scheme, monkeypatch):
    model, state = thermal_fixture()
    solver = PRCollocation(model, scheme=scheme, phase_fallback=False)
    # Nonzero roundoff/error estimate: the old power-law extrapolation from
    # a tiny report remainder shrank next_step to minimum_step_s.
    monkeypatch.setattr(model, "_error", lambda coarse, fine: 0.01)
    duration = min(1e-12, model.minimum_step_s / 10)
    snapshot = state.to_dict()
    report = solver.advance(state, duration)
    assert report["state"]["time_s"] == duration
    assert sum(s["dt_s"] for s in report["steps"]) == duration
    assert len(report["steps"]) == 2
    assert report["state"]["next_step_s"] >= state.next_step_s
    assert state.to_dict() == snapshot


@pytest.mark.parametrize("scheme", ["gauss4", "radau3"])
def test_subminimum_remainder_never_bypasses_error_rejection(scheme, monkeypatch):
    from surface.liquid import HydraulicConvergenceError

    model, state = thermal_fixture()
    solver = PRCollocation(model, scheme=scheme, phase_fallback=False)
    monkeypatch.setattr(model, "_error", lambda coarse, fine: 2.0)
    snapshot = state.to_dict()
    with pytest.raises(HydraulicConvergenceError, match="minimum step"):
        solver.advance(state, min(1e-12, model.minimum_step_s / 10))
    assert state.to_dict() == snapshot
