"""Early full-step acceptance must retain convergence and guarded backtracking."""

import pytest
import torch

from surface.torch_nonlinear import TorchNewton

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_exact_linear_newton_step_avoids_redundant_residuals(device):
    start = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    matrix = start.new_tensor([[2.0, -0.1], [0.3, 3.0]])
    target = start.new_tensor([1.0, -2.0])
    results = []
    for enabled in (False, True):
        solver = TorchNewton(device)
        solver.host_convergence = True
        solver._early_full_step = enabled
        result = solver.solve(
            lambda x: matrix @ (x - target),
            start,
            start.new_full((2,), -10.0),
            start.new_full((2,), 10.0),
            tolerance=3e-12,
        )
        assert bool(result[2])
        torch.testing.assert_close(result[0], target, rtol=0, atol=3e-12)
        assert solver.stop_reason == "target_tolerance"
        results.append(result)
    assert results[1][3] == results[0][3] - 5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["overshoot", "invalid_full_step", "bounds"])
def test_rejected_full_step_retains_backtracking_and_physical_solution(device, case):
    start = torch.tensor([0.1], dtype=torch.float64, device=device)
    if case == "overshoot":
        residual = lambda x: x.square() - 2.0
        lower, upper, target = 0.0, 20.0, 2.0**0.5
    elif case == "invalid_full_step":
        start.fill_(0.01)
        # A full Newton update crosses the domain x < 1; half-steps recover.
        residual = lambda x: torch.log1p(-x) + 2.0
        lower, upper, target = -1.0, 2.0, 0.8646647167633873
    else:
        start.fill_(9.0)
        residual = lambda x: torch.exp(x) - 2.0
        lower, upper, target = 0.0, 10.0, 0.6931471805599453
    for enabled in (False, True):
        solver = TorchNewton(device, maximum_iterations=30)
        solver.host_convergence = True
        solver._early_full_step = enabled
        x, r, ok, _ = solver.solve(
            residual,
            start,
            start.new_full((1,), lower),
            start.new_full((1,), upper),
            tolerance=3e-12,
        )
        assert bool(ok) and float(r.abs().max()) <= 3e-12
        torch.testing.assert_close(x, start.new_tensor([target]), rtol=0, atol=3e-12)
