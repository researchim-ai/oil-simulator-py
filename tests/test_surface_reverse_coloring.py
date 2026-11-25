"""Compressed reverse AD must recover the original network derivative."""

import pytest
import torch
from test_surface_pr_surface_control import controlled

from surface.pr_collocation import PRCollocation
from surface.pr_surface_collocation import PRSurfaceCollocation
from surface.torch_linear import ReverseColoredJacobian
from surface.torch_nonlinear import TorchNewton


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_reverse_recovery_matches_full_ad_with_duplicates_and_empty_rows(chunk):
    x = torch.linspace(0.2, 1.2, 12, dtype=torch.float64)

    def residual(z):
        return torch.cat((z[:-1].sin() + z[1:] ** 3, z.new_zeros(1)))

    rows = [i for i in range(11) for _ in range(2)] + [0]
    cols = [j for i in range(11) for j in (i, i + 1)] + [0]
    planner = ReverseColoredJacobian(12, rows, cols, "cpu", chunk_size=chunk)
    assert planner.color_count <= 2
    expected = torch.func.jacrev(residual)(x)
    actual = planner.values(residual, x)
    torch.testing.assert_close(actual, expected[rows, cols], atol=1e-14, rtol=1e-14)
    # No tape or state cache survives a new thermodynamic trial.
    torch.testing.assert_close(
        planner.values(residual, x * 2),
        torch.func.jacrev(residual)(x * 2)[rows, cols],
    )


def test_dense_newton_consumes_sparse_reverse_derivative():
    x = torch.ones(20, dtype=torch.float64)
    rows, cols = [], []
    for i in range(20):
        for j in {i, max(0, i - 1)}:
            rows.append(i)
            cols.append(j)

    def residual(z):
        return z.square() + 0.1 * torch.cat((z[:1], z[:-1])) - 2

    results = []
    for mode in ("reverse", "colored_reverse"):
        solver = TorchNewton("cpu", dense_limit=32)
        solver._dense_jacobian_mode = mode
        with torch.no_grad():
            result, row, ok, _ = solver.solve(residual, x, x * 0.1, x * 3, (rows, cols))
        assert bool(ok)
        assert float(row.abs().max()) < 2e-11
        results.append(result)
    torch.testing.assert_close(*results, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("opening", [True, False])
def test_controlled_gravity_stage_support_matches_full_derivative(monkeypatch, opening):
    model, old = controlled(feedback=True)
    integrator = PRSurfaceCollocation(model, phase_fallback=False)
    integrator.plant.engine._dense_cpu_jacobian_mode = "colored_reverse"
    original = PRCollocation._solve
    checked = set()

    def verify(self, fn, initial, lo, hi, pattern, update):
        assert pattern is not None
        signature = (len(initial), tuple(pattern[0]), tuple(pattern[1]))
        if signature not in checked:
            checked.add(signature)
            expected = torch.func.jacrev(fn)(initial)
            planner = ReverseColoredJacobian(len(initial), *pattern, "cpu")
            actual = torch.zeros_like(expected)
            actual[planner.plan.c, planner.plan.r] = planner.values(fn, initial)
            torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
        return original(self, fn, initial, lo, hi, pattern, update)

    monkeypatch.setattr(PRCollocation, "_solve", verify)
    integrator._step(old, 1e-6, {}, {}, {"openings": {"port": opening}})
    assert checked


def test_reverse_coloring_builds_one_tape_per_call():
    calls = []
    x = torch.ones(9, dtype=torch.float64)

    def function(z):
        calls.append(True)
        return z.exp()

    planner = ReverseColoredJacobian(9, range(9), range(9), "cpu", chunk_size=1)
    torch.testing.assert_close(planner.dense(function, x), torch.diag(x.exp()))
    assert len(calls) == 1
    planner.dense(function, x + 1)
    assert len(calls) == 2


def test_structural_cache_is_bounded_and_invalidated_by_support():
    model, _ = controlled(feedback=False)
    engine = PRSurfaceCollocation(model).plant.engine
    x = torch.ones(12, dtype=torch.float64)
    diagonal = (list(range(12)), list(range(12)))
    first = engine._reverse_plan(x, diagonal)
    assert engine._reverse_plan(x + 1, diagonal) is first
    for i in range(1, 10):
        other = (diagonal[0] + [0], diagonal[1] + [i])
        assert engine._reverse_plan(x, other) is not first
    assert len(engine._reverse_colorings) == 8
    assert engine._reverse_plan(x, diagonal) is not first


def test_supplied_plan_cannot_alias_another_dependency_graph():
    x = torch.ones(3, dtype=torch.float64)
    solver = TorchNewton("cpu")
    solver._dense_jacobian_mode = "colored_reverse"
    coloring = ReverseColoredJacobian(3, [0, 1, 2], [0, 1, 2], "cpu")
    with pytest.raises(ValueError, match="exact residual support"):
        solver.solve(
            lambda z: z - 2,
            x,
            x * 0,
            x * 3,
            ([0, 1, 2, 0], [0, 1, 2, 1]),
            reverse_coloring=coloring,
        )


@pytest.mark.parametrize("pattern", [None, ([0, 0, 1, 1], [0, 1, 0, 1])])
def test_absent_or_dense_support_uses_complete_reverse_ad(pattern):
    x = torch.ones(2, dtype=torch.float64)
    solver = TorchNewton("cpu")
    solver._dense_jacobian_mode = "colored_reverse"
    with torch.no_grad():
        result, _, ok, _ = solver.solve(lambda z: z - 2, x, x * 0, x * 3, pattern)
    assert bool(ok)
    torch.testing.assert_close(result, x * 2)


def test_global_balance_selects_reverse_without_constructing_forward_conflicts(monkeypatch):
    from test_surface_bounded_jacobian import LargestTensor, hub

    import surface.torch_linear as linear

    n = 20000
    rows, cols, function = hub(n)
    original = linear.ColoredJacobian

    def transpose_only(size, r, c, *args, **kwargs):
        # Forward planning would construct the complete column-conflict graph.
        assert r == cols and c == rows
        return original(size, r, c, *args, **kwargs)

    monkeypatch.setattr(linear, "ColoredJacobian", transpose_only)
    with LargestTensor() as allocations:
        plan = linear.sparse_jacobian_plan(n, rows, cols, "cpu", reverse=True, chunk_size=8)
        x = torch.linspace(0.1, 1.1, n, dtype=torch.float64)
        values = plan.values(function, x)
    assert isinstance(plan, linear.ReverseColoredJacobian)
    assert plan.color_count == 2
    assert allocations.largest <= 8 * n
    expected = torch.cat((x.cos(), 2 * x[1:] + 2))
    torch.testing.assert_close(values, expected, rtol=0, atol=0)


def test_auto_sparse_newton_keeps_sparse_algebra_for_global_balance(monkeypatch):
    from surface.torch_linear import ReverseColoredJacobian

    n = 513
    rows = [0] * n + list(range(1, n))
    cols = list(range(n)) + list(range(1, n))
    x = torch.ones(n, dtype=torch.float64) * 0.9

    def residual(z):
        return torch.cat(((z.sum() - n).reshape(1), z[1:].square() - 1))

    def forbidden(*args, **kwargs):
        raise AssertionError("Sparse solver cannot build or factor a dense global Jacobian")

    monkeypatch.setattr(torch.func, "jacrev", forbidden)
    monkeypatch.setattr(torch.func, "jacfwd", forbidden)
    monkeypatch.setattr(torch.linalg, "lu_factor_ex", forbidden)
    solver = TorchNewton("cpu", dense_limit=32)
    solver._sparse_jacobian_mode = "auto"
    with torch.no_grad():
        result, row, ok, _ = solver.solve(residual, x, x * 0, x * 3, (rows, cols))
    assert bool(ok)
    assert float(row.abs().max()) < 2e-11
    assert isinstance(solver.coloring, ReverseColoredJacobian)
    assert solver.matrix.values.numel() == 2 * n - 1
    # The first stock is inferred from a sum of n independently bounded rows.
    torch.testing.assert_close(result[1:], torch.ones_like(result[1:]), rtol=0, atol=2e-11)
    assert abs(float(result[0]) - 1) <= n * 2e-11


def test_auto_preserves_forward_on_local_graph():
    from surface.torch_linear import ColoredJacobian, sparse_jacobian_plan

    n = 25
    plan = sparse_jacobian_plan(n, range(n), range(n), "cpu", reverse=True)
    assert isinstance(plan, ColoredJacobian)
    assert plan.color_count == 1


def test_global_sensor_column_does_not_build_reverse_conflicts(monkeypatch):
    import surface.torch_linear as linear

    n = 20000
    rows = list(range(n)) + list(range(1, n))
    cols = [0] * n + list(range(1, n))

    def forbidden(*args, **kwargs):
        raise AssertionError("Forward coloring is already below the reverse conflict lower bound")

    monkeypatch.setattr(linear, "ReverseColoredJacobian", forbidden)
    plan = linear.sparse_jacobian_plan(n, rows, cols, "cpu", reverse=True)
    assert isinstance(plan, linear.ColoredJacobian)
    assert plan.color_count == 2
