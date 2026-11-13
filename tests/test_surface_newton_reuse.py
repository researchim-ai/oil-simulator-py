"""Lagged tangents must reduce assembly while retaining fresh-Newton recovery."""

import pytest
import torch

from surface.torch_nonlinear import TorchNewton

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def run(device, function, start, *, enabled=True, sparse=False, solver=None):
    solver = solver or TorchNewton(device, maximum_iterations=30, dense_limit=0 if sparse else 128)
    solver.host_convergence = True
    solver._reuse_jacobian = enabled
    size = start.numel()
    pattern = ([i for i in range(size)], [i for i in range(size)]) if sparse else None
    result = solver.solve(
        function,
        start,
        start.new_full((size,), -10.0),
        start.new_full((size,), 10.0),
        pattern,
        tolerance=3e-12,
        blocks=[[i] for i in range(size)] if sparse else None,
    )
    return solver, result


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sparse", [False, True])
def test_smooth_nonlinear_solution_requires_fewer_jacobian_builds(device, sparse):
    start = torch.tensor([1.45, 1.75], device=device, dtype=torch.float64)
    target = start.new_tensor([2.0, 3.0])
    function = lambda x: x.square() - target
    fresh, reference = run(device, function, start, enabled=False, sparse=sparse)
    lagged, actual = run(device, function, start, sparse=sparse)
    for result in (reference, actual):
        assert bool(result[2]) and float(result[1].abs().max()) <= 3e-12
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=3e-12)
    assert lagged.jacobian_reuses > 0
    assert lagged.jacobian_builds < fresh.jacobian_builds


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sparse", [False, True])
def test_obsolete_direction_retries_same_iterate_with_fresh_tangent(device, sparse):
    start = torch.zeros(1, device=device, dtype=torch.float64)
    # At x=0 J=-1, so the full step reaches x=1 with a 10x smaller
    # residual. There J=+2.2: the old direction is now uphill.
    function = lambda x: 1 - x - 2.9 * x.square() + 3 * x**3
    _fresh, reference = run(device, function, start, enabled=False, sparse=sparse)
    lagged, actual = run(device, function, start, sparse=sparse)
    assert bool(reference[2]) and bool(actual[2])
    assert float(actual[1].abs().max()) <= 3e-12
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=3e-12)
    assert lagged.jacobian_retry_refreshes >= 1
    assert lagged.iterations <= lagged.maximum_iterations


@pytest.mark.parametrize("device", DEVICES)
def test_singular_dense_tangent_retains_regularized_device_fallback(device):
    start = torch.zeros(2, device=device, dtype=torch.float64)
    function = lambda x: torch.stack((x.sum() - 1, 2 * (x.sum() - 1)))
    solver, (x, r, ok, _) = run(device, function, start)
    assert bool(ok) and float(r.abs().max()) <= 3e-12
    assert float(x.sum()) == pytest.approx(1.0, abs=3e-12)
    assert solver.jacobian_builds > 0


@pytest.mark.parametrize("device", DEVICES)
def test_new_solve_never_reuses_previous_numeric_factors(device):
    start = torch.ones(1, device=device, dtype=torch.float64)
    solver, first = run(device, lambda x: x - 2, start)
    assert bool(first[2])
    solver, second = run(device, lambda x: 7 * (x - 3), start, solver=solver)
    assert bool(second[2]) and float(second[1].abs().max()) <= 3e-12
    assert float(second[0][0]) == pytest.approx(3.0, abs=3e-12)
    assert solver.jacobian_builds == 1 and solver.jacobian_reuses == 0


@pytest.mark.parametrize("device", DEVICES)
def test_new_sparse_topology_and_block_partition_are_compiled_again(device):
    start = torch.zeros(2, device=device, dtype=torch.float64)
    solver, first = run(device, lambda x: x - 1, start, sparse=True)
    assert bool(first[2])
    matrix = start.new_tensor([[2.0, 0.5], [-0.3, 3.0]])
    target = start.new_tensor([3.0, 4.0])
    x, r, ok, _ = solver.solve(
        lambda x: matrix @ (x - target),
        start,
        start.new_full((2,), -10.0),
        start.new_full((2,), 10.0),
        ([0, 0, 1, 1], [0, 1, 0, 1]),
        blocks=[[0, 1]],
        tolerance=3e-12,
    )
    assert bool(ok) and float(r.abs().max()) <= 3e-12
    torch.testing.assert_close(x, target, rtol=0, atol=3e-12)
    assert solver.matrix.values.numel() == 4
    assert solver.block_preconditioner.groups[0][0] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fixed_budget_cuda_retains_no_host_status_reads():
    start = torch.ones(2, device="cuda", dtype=torch.float64)
    solver = TorchNewton("cuda", maximum_iterations=4)
    _x, r, ok, _ = solver.solve(
        lambda x: x - 2,
        start,
        torch.zeros_like(start),
        start.new_full((2,), 10),
        tolerance=3e-12,
    )
    assert bool(ok) and float(r.abs().max()) <= 3e-12
    assert solver.status_reads == 0 and solver.jacobian_reuses == 0
