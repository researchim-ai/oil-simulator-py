"""Carry only the accepted scalar residual and its own arithmetic error budget."""

import pytest
import torch

from surface.torch_nonlinear import TorchNewton

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sparse", [False, True])
def test_same_root_and_fewer_actual_evaluations(device, sparse):
    results = []
    for enabled in (False, True):
        x = torch.ones(2, dtype=torch.float64, device=device)
        solver = TorchNewton(device, dense_limit=0 if sparse else 512)
        solver.host_convergence = True
        solver._reuse_accepted_residual = enabled
        with torch.no_grad():
            root, residual, ok, evaluations = solver.solve(
                lambda z: z.square() - 2,
                x,
                x * 0,
                x * 4,
                ([0, 1], [0, 1]) if sparse else None,
                tolerance=3e-12,
            )
        assert bool(ok) and residual.abs().max() < 3e-12
        results.append((root, evaluations, solver.accepted_residual_reuses))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert results[1][1] < results[0][1]
    assert results[1][2] == results[0][1] - results[1][1]


def test_backtracking_restores_budget_of_selected_point_not_last_trial():
    roots = []
    for enabled in (False, True):
        x = torch.tensor([0.5], dtype=torch.float64)
        buffer = torch.zeros_like(x)

        def values(z):
            # The last rejected scalar trial can have a huge arithmetic budget;
            # the accepted trial does not. Mutable storage exposes alias bugs.
            return z**4 - 2, torch.where((z > 0.6) & (z < 0.7), 100.0, 0.0)

        def accept(bounds, buffer=buffer):
            buffer.copy_(bounds)

        def function(z):
            row, budget = values(z)
            if not torch._C._functorch.is_functorch_wrapped_tensor(z):
                accept(budget)
            return row

        function.runtime_values = values
        function.accept_runtime_bounds = accept
        solver = TorchNewton("cpu", maximum_iterations=30)
        solver._reuse_accepted_residual = enabled
        with torch.no_grad():
            root, residual, ok, _ = solver.solve(
                function,
                x,
                x * 0,
                x * 10,
                tolerance=3e-12,
                stagnation_tolerance=1e-11,
                roundoff_tolerance=lambda buffer=buffer: buffer,
            )
        assert bool(ok) and residual.abs().max() < 3e-12
        roots.append(root)
    torch.testing.assert_close(roots[0], roots[1], rtol=0, atol=0)


def test_unknown_roundoff_callback_keeps_fresh_scalar_evaluation():
    bounds = torch.zeros(1, dtype=torch.float64)
    solver = TorchNewton("cpu")
    x = torch.ones_like(bounds)
    with torch.no_grad():
        _, residual, ok, _ = solver.solve(
            lambda z: z.square() - 2,
            x,
            x * 0,
            x * 4,
            tolerance=3e-12,
            stagnation_tolerance=1e-11,
            roundoff_tolerance=lambda: bounds,
        )
    assert bool(ok) and residual.abs().max() < 1e-11
    assert solver.accepted_residual_reuses == 0


def test_parameter_sensitivities_keep_original_differentiable_path():
    p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([0.4], dtype=p.dtype)
    solver = TorchNewton("cpu")
    root, _, ok, _ = solver.solve(lambda z: p * z - 1, x, x * 0, x * 5)
    assert bool(ok) and solver.accepted_residual_reuses == 0
    torch.testing.assert_close(torch.autograd.grad(root.sum(), p)[0], p.new_tensor(-0.25))
