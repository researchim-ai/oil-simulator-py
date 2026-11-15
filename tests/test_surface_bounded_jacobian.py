"""Sparse extraction bounds temporary directions even on a high-degree hub."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from surface.torch_linear import ColoredJacobian


class LargestTensor(TorchDispatchMode):
    def __init__(self):
        self.largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))

        def visit(value):
            if isinstance(value, torch.Tensor):
                self.largest = max(self.largest, value.numel())
            elif isinstance(value, (tuple, list)):
                for v in value:
                    visit(v)

        visit(result)
        return result


def hub(size):
    # One global balance row, all other rows local: O(N) nonzero entries,
    # but every column conflicts with every other through the first row.
    rows = [0] * size + list(range(1, size))
    cols = list(range(size)) + list(range(1, size))

    def function(x):
        return torch.cat(((x.sin().sum()).reshape(1), x[1:].square() + 2 * x[1:]))

    return rows, cols, function


@pytest.mark.parametrize("chunk_size", [1, 7, 16, 128])
def test_chunked_values_match_dense_jacobian_and_original_coordinate_order(chunk_size):
    n = 37
    rows, cols, fn = hub(n)
    # Non-row-sorted request, including duplicate declared coordinates.
    rows, cols = rows[::-1] + [0], cols[::-1] + [3]
    x = torch.linspace(0.1, 1.1, n, dtype=torch.float64)
    plan = ColoredJacobian(n, rows, cols, "cpu", chunk_size=chunk_size)
    expected = torch.func.jacfwd(fn)(x)[rows, cols]
    torch.testing.assert_close(plan.values(fn, x), expected, rtol=0, atol=0)
    assert plan.color_count == n
    # A second state cannot reuse derivatives from the first state.
    y = x + 0.3
    torch.testing.assert_close(
        plan.values(fn, y), torch.func.jacfwd(fn)(y)[rows, cols], rtol=0, atol=0
    )


def test_hub_never_allocates_color_by_size_or_identity_seed_tensor():
    n, chunk = 513, 8
    rows, cols, fn = hub(n)
    allocations = LargestTensor()
    with allocations:
        plan = ColoredJacobian(n, rows, cols, "cpu", chunk_size=chunk)
        x = torch.linspace(0.1, 1.1, n, dtype=torch.float64)
        result = plan.values(fn, x)
    assert plan.color_count == n
    assert result.numel() == 2 * n - 1
    assert allocations.largest <= chunk * n
    expected = torch.cat((x.cos(), 2 * x[1:] + 2))
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_local_graph_storage_and_directions_remain_linear_in_size():
    n = 20000
    rows = list(range(n)) + list(range(n - 1))
    cols = list(range(n)) + list(range(1, n))

    def fn(x):
        return x.square() + torch.cat((0.2 * x[1:], x.new_zeros(1)))

    with LargestTensor() as allocations:
        plan = ColoredJacobian(n, rows, cols, "cpu", chunk_size=4)
        x = torch.linspace(0.1, 1.1, n, dtype=torch.float64)
        values = plan.values(fn, x)
    assert plan.color_count == 2
    assert allocations.largest <= 2 * n
    torch.testing.assert_close(values[:n], 2 * x, rtol=0, atol=0)
    torch.testing.assert_close(values[n:], x.new_full((n - 1,), 0.2), rtol=0, atol=0)


def test_empty_pattern_and_invalid_budget():
    plan = ColoredJacobian(3, [], [], "cpu")
    assert plan.values(lambda x: x * 0, torch.ones(3, dtype=torch.float64)).numel() == 0
    for chunk in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="batch size"):
            ColoredJacobian(3, [0], [0], "cpu", chunk_size=chunk)
    with pytest.raises(ValueError, match="coordinates"):
        ColoredJacobian(3, [0], [], "cpu")


def test_large_sparse_newton_never_calls_global_dense_jacobian_or_lu(monkeypatch):
    from surface.torch_nonlinear import TorchNewton

    n = 2048
    rows = list(range(n)) + list(range(1, n)) + list(range(n - 1))
    cols = list(range(n)) + list(range(n - 1)) + list(range(1, n))

    def operator(x):
        left = torch.cat((x.new_zeros(1), x[:-1]))
        right = torch.cat((x[1:], x.new_zeros(1)))
        return 3 * x + 0.01 * x.square() - 0.1 * (left + right)

    target = operator(torch.ones(n, dtype=torch.float64))

    def function(x):
        return operator(x) - target

    def forbidden(*args, **kwargs):
        raise AssertionError("Global dense algebra on sparse solve")

    monkeypatch.setattr(torch.func, "jacfwd", forbidden)
    monkeypatch.setattr(torch.linalg, "lu_factor_ex", forbidden)
    solver = TorchNewton("cpu", maximum_iterations=8, dense_limit=64)
    x = torch.zeros_like(target)
    with LargestTensor() as allocations:
        root, residual, ok, _ = solver.solve(function, x, x - 10, x + 10, pattern=(rows, cols))
    assert bool(ok) and float(residual.abs().max()) < 2e-11
    torch.testing.assert_close(root, torch.ones_like(root), rtol=0, atol=1e-11)
    assert solver.matrix.values.numel() == 3 * n - 2
    assert solver.coloring.color_count <= 3
    assert allocations.largest <= 32 * n


def test_entry_budget_reduces_batch_width_without_approximating_derivatives():
    n = 33
    rows, cols, fn = hub(n)
    x = torch.linspace(0.1, 1.1, n, dtype=torch.float64)
    plan = ColoredJacobian(n, rows, cols, "cpu", chunk_size=64, maximum_seed_entries=2 * n)
    with LargestTensor() as allocations:
        result = plan.values(fn, x)
    assert plan.chunk_size == 2
    assert allocations.largest <= 2 * n
    torch.testing.assert_close(result, torch.func.jacfwd(fn)(x)[rows, cols], rtol=0, atol=0)
    one = ColoredJacobian(n, rows, cols, "cpu", maximum_seed_entries=1)
    assert one.chunk_size == 1
    torch.testing.assert_close(one.values(fn, x), result, rtol=0, atol=0)
    with pytest.raises(ValueError, match="seed entry budget"):
        ColoredJacobian(n, rows, cols, "cpu", maximum_seed_entries=0)
