"""Only caller-certified arithmetic rows may stop above the target tolerance."""

import pytest
import torch

from surface.torch_nonlinear import TorchNewton

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_certified_floor_saves_work_without_changing_the_solution(device):
    start = torch.tensor([0.0, 1e-4], dtype=torch.float64, device=device)
    results = []
    for enabled in (False, True):
        solver = TorchNewton(device, maximum_iterations=40)
        solver.host_convergence = True
        solver._early_certified_roundoff = enabled
        result = solver.solve(
            lambda x: torch.stack(((x[0] + 1e10) - 1e10 + 1.2e-11, x[1])),
            start,
            start.new_full((2,), -1.0),
            start.new_ones(2),
            tolerance=3e-12,
            stagnation_tolerance=1e-11,
            roundoff_tolerance=lambda: start.new_tensor([2e-11, 0.0]),
        )
        assert bool(result[2])
        results.append((solver, result))
    assert results[1][0].stop_reason == "certified_roundoff"
    assert results[1][1][3] < results[0][1][3] / 3
    torch.testing.assert_close(results[1][1][0], results[0][1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[1][1][1], results[0][1][1], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("certificate", [[0.0, 2e-11], [float("nan"), 0.0], [-1.0, 0.0]])
def test_wrong_row_or_invalid_certificate_does_not_authorize_stop(device, certificate):
    start = torch.zeros(2, dtype=torch.float64, device=device)
    solver = TorchNewton(device, maximum_iterations=4)
    solver.host_convergence = True
    _, _, ok, _ = solver.solve(
        lambda x: torch.stack(((x[0] + 1e10) - 1e10 + 1.2e-11, x[1])),
        start,
        start - 1,
        start + 1,
        tolerance=3e-12,
        stagnation_tolerance=1e-11,
        roundoff_tolerance=lambda: start.new_tensor(certificate),
    )
    assert not bool(ok)
    assert solver.stop_reason != "certified_roundoff"


@pytest.mark.parametrize("device", DEVICES)
def test_uncertified_equations_still_reach_strict_target(device):
    start = torch.tensor([0.0, 7e-12], dtype=torch.float64, device=device)
    solver = TorchNewton(device, maximum_iterations=20)
    solver.host_convergence = True
    _, r, ok, _ = solver.solve(
        lambda x: torch.stack(((x[0] + 1e10) - 1e10 + 1.2e-11, x[1])),
        start,
        start - 1,
        start + 1,
        tolerance=3e-12,
        stagnation_tolerance=1e-11,
        roundoff_tolerance=lambda: start.new_tensor([2e-11, 0.0]),
    )
    assert bool(ok) and abs(float(r[1])) <= 3e-12
