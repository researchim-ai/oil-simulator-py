"""Physical blocks retain pressure/composition couplings in device Krylov work."""

import pytest
import torch

from surface.torch_linear import BlockJacobi, TorchCOO, gmres_device


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_discontinuous_physical_blocks_solve_constrained_sparse_system(device):
    count = 40
    blocks = [[i, i + count] for i in range(count)]
    rows, cols, values = [], [], []
    for i in range(count):
        for r, c, value in [
            (i, i, 0.0),
            (i, i + count, 10.0 ** (i / 10 - 2)),
            (i + count, i, 10.0 ** (2 - i / 10)),
            (i + count, i + count, 0.1),
        ]:
            rows.append(r)
            cols.append(c)
            values.append(value)
    values = torch.tensor(values, dtype=torch.float64, device=device)
    matrix = TorchCOO(rows, cols, values, 2 * count, device)
    preconditioner = BlockJacobi(rows, cols, blocks, 2 * count, device)
    preconditioner.update(values)
    exact = torch.linspace(-1, 1, 2 * count, dtype=values.dtype, device=device)
    rhs = matrix.mv(exact)
    actual = gmres_device(matrix.mv, rhs, matrix.diagonal(), preconditioner=preconditioner)
    torch.testing.assert_close(actual, exact, rtol=1e-10, atol=1e-10)
    assert float((matrix.mv(actual) - rhs).abs().max()) < 1e-10
    # Reuse topology with new coefficients: factors cannot be stale.
    changed = values * 1.3
    preconditioner.update(changed)
    torch.testing.assert_close(preconditioner(rhs * 1.3), exact, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("blocks", [[[0], [0, 1]], [[0]], [[], [0, 1]], [[0, 2]]])
def test_incomplete_or_overlapping_block_partition_is_rejected(blocks):
    with pytest.raises(ValueError):
        BlockJacobi([0, 1], [0, 1], blocks, 2, "cpu")


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_log_inventory_updates_do_not_freeze_other_physical_unknowns(device):
    from surface.torch_linear import TorchNewton

    dtype = torch.float64
    target = torch.tensor(1e-8, dtype=dtype, device=device)
    initial = torch.tensor([1e-18, 1.0], dtype=dtype, device=device)
    initial[0] = initial[0].log()
    solver = TorchNewton(device, maximum_iterations=24)
    solver.host_convergence = True

    def residual(y):
        return torch.stack(((y[0].exp() - target) / target, (y[1] - 1000) / 1000))

    result, residual_value, accepted, _ = solver.solve(
        residual,
        initial,
        initial.new_tensor([-700.0, -1e6]),
        initial.new_tensor([100.0, 1e6]),
        maximum_update=initial.new_tensor([4.0, float("inf")]),
    )
    assert bool(accepted)
    assert float(result[0].exp()) == pytest.approx(1e-8, rel=1e-11)
    assert float(result[1]) == pytest.approx(1000.0, abs=1e-10)
    assert float(residual_value.abs().max()) < 2e-11


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_stiff_positive_root_uses_backtracking_beyond_six_trials(device):
    from surface.torch_linear import TorchNewton

    y = torch.tensor([-12.0], dtype=torch.float64, device=device)
    solver = TorchNewton(device, maximum_iterations=20)
    solver.host_convergence = True
    result, residual, accepted, _ = solver.solve(
        lambda z: z.exp() - 1,
        y,
        y.new_tensor([-100.0]),
        y.new_tensor([100.0]),
    )
    assert bool(accepted)
    assert float(result[0]) == pytest.approx(0.0, abs=2e-11)
    assert float(residual.abs().max()) < 2e-11


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_irrecoverable_stagnation_returns_failure_without_repeating_same_iterate(device):
    from surface.torch_linear import TorchNewton

    y = torch.zeros(1, dtype=torch.float64, device=device)
    solver = TorchNewton(device, maximum_iterations=40)
    solver.host_convergence = True
    _, residual, accepted, evaluations = solver.solve(
        lambda z: 1 + 0 * z,
        y,
        y - 1,
        y + 1,
    )
    assert not bool(accepted)
    assert float(residual[0]) == 1.0
    assert evaluations < 40
