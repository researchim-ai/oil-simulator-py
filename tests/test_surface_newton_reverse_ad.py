"""Reverse AD accelerates bounded dense blocks without changing sparse assembly."""

import pytest
import torch
from test_surface_pr_check_valve_active_set import captured_branch

from surface.torch_nonlinear import TorchNewton


@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_bounded_dense_newton_same_root(mode):
    solver = TorchNewton("cpu", maximum_iterations=30)
    solver._dense_jacobian_mode = mode
    start = torch.tensor([1.4, 1.7], dtype=torch.float64)
    x, r, ok, _ = solver.solve(
        lambda z: z.square() - z.new_tensor([2.0, 3.0]),
        start,
        torch.zeros_like(start),
        torch.full_like(start, 4.0),
        tolerance=3e-12,
    )
    assert bool(ok)
    assert float(r.abs().max()) < 3e-12
    torch.testing.assert_close(x, start.new_tensor([2.0, 3.0]).sqrt(), rtol=0, atol=2e-12)


def test_actual_coupled_pr_restart_forward_reverse_tangents():
    _, _, _, y, evaluate = captured_branch()
    function = lambda z: evaluate(z)[0]
    with torch.no_grad():
        forward = torch.func.jacfwd(function)(y)
        reverse = torch.func.jacrev(function)(y)
    torch.testing.assert_close(reverse, forward, rtol=2e-10, atol=3e-11)
    # A directional finite difference independently checks the AD mode agreement.
    direction = torch.linspace(-0.001, 0.001, len(y), dtype=y.dtype)
    eps = 1e-4
    finite = (function(y + eps * direction) - function(y - eps * direction)) / (2 * eps)
    torch.testing.assert_close(reverse @ direction, finite, rtol=2e-5, atol=2e-6)


def test_reverse_option_keeps_sparse_coloring(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Sparse graph cannot allocate a dense reverse Jacobian")

    monkeypatch.setattr(torch.func, "jacrev", forbidden)
    solver = TorchNewton("cpu", dense_limit=0)
    solver._dense_jacobian_mode = "reverse"
    start = torch.ones(3, dtype=torch.float64)
    _, residual, ok, _ = solver.solve(
        lambda z: z.square() - 2,
        start,
        torch.zeros_like(start),
        torch.full_like(start, 4),
        ([0, 1, 2], [0, 1, 2]),
        blocks=[[0], [1], [2]],
        tolerance=3e-12,
    )
    assert bool(ok) and float(residual.abs().max()) < 3e-12
    assert solver.coloring is not None
