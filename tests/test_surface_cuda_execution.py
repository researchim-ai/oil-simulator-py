"""Bounded execution must preserve Newton acceptance, AD and live state."""

import pytest
import torch

from surface.cuda_execution import CUDAExecution, cuda_graph_execution, current_execution
from surface.torch_nonlinear import TorchNewton


def test_scoped_cache_lifetime_and_nested_reuse():
    assert current_execution() is None
    with cuda_graph_execution(maximum_graphs=2) as first:
        with cuda_graph_execution() as nested:
            assert nested is first
            assert current_execution() is first
        assert not first.disabled
    assert first.disabled and not first.entries and not first.compiled
    assert current_execution() is None


@pytest.mark.parametrize(
    "budgets", [{"maximum_graphs": 0}, {"maximum_bytes": True}, {"minimum_size": 1.5}]
)
def test_invalid_execution_budgets(budgets):
    with pytest.raises(ValueError, match="Positive integer"):
        CUDAExecution(**budgets)


def test_cpu_and_parameter_derivatives_do_not_compile():
    runtime = CUDAExecution(minimum_size=1)
    x = torch.tensor([2.0, 3.0], dtype=torch.float64)
    p = torch.tensor(1.5, dtype=x.dtype, requires_grad=True)
    jac = runtime.jacobian(lambda z: p * z.square(), x)
    torch.testing.assert_close(jac, torch.diag(2 * p * x))
    assert torch.autograd.grad(jac.sum(), p)[0] == 10
    assert not runtime.compiled and runtime.stats["captures"] == 0


def test_newton_runtime_bounds_are_updated_outside_pure_kernel(monkeypatch):
    runtime = CUDAExecution(minimum_size=1)
    calls = {"residual": 0, "jacobian": 0, "bounds": 0, "colored": 0}
    bounds = [None]

    def pure(z):
        calls["residual"] += 1
        return z.square() - 2, torch.zeros_like(z)

    def function(z):
        return z.square() - 2

    def accept(value):
        calls["bounds"] += 1
        bounds[0] = value

    def jacobian(fn, x):
        calls["jacobian"] += 1
        return torch.func.jacrev(fn)(x)

    def colored(coloring, fn, x):
        calls["colored"] += 1
        return coloring.values(fn, x)

    function.runtime_values = pure
    function.accept_runtime_bounds = accept
    monkeypatch.setattr(runtime, "colored_values", colored)
    monkeypatch.setattr(runtime, "eligible", lambda x: True)
    monkeypatch.setattr(runtime, "jacobian", jacobian)
    monkeypatch.setattr(runtime, "residual_values", lambda fn, x: fn(x))
    import surface.torch_nonlinear as nonlinear

    monkeypatch.setattr(nonlinear, "current_execution", lambda: runtime)
    solver = TorchNewton("cpu")
    x = torch.ones(2, dtype=torch.float64)
    root, residual, ok, _ = solver.solve(
        function,
        x,
        x * 0,
        x * 4,
        tolerance=3e-12,
        stagnation_tolerance=1e-11,
        roundoff_tolerance=lambda: bounds[0],
    )
    assert bool(ok) and residual.abs().max() < 3e-12
    torch.testing.assert_close(root, x * 2**0.5)
    assert calls["jacobian"] > 0
    assert calls["bounds"] - calls["residual"] == solver.accepted_residual_reuses
    calls.update(residual=0, jacobian=0, bounds=0, colored=0)
    solver.dense_limit = 0
    solver.solve(function, x, x * 0, x * 4, ([0, 1], [0, 1]))
    assert calls["jacobian"] == 0 and calls["colored"] > 0
    assert calls["residual"] == calls["bounds"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires accessible CUDA")
def test_gpu_newton_replay_live_coefficients_and_owned_outputs():
    x = torch.ones(2, dtype=torch.float64, device="cuda")
    coefficients = x.new_full((2,), 2.0)
    bounds = [None]

    def pure(z):
        return z.square() - coefficients, torch.zeros_like(z)

    def accept(value):
        bounds[0] = value

    def function(z):
        row, error = pure(z)
        if not torch._C._functorch.is_functorch_wrapped_tensor(z):
            accept(error)
        return row

    function.runtime_values = pure
    function.accept_runtime_bounds = accept
    with cuda_graph_execution(minimum_size=1) as execution, torch.no_grad():
        solver = TorchNewton("cuda", maximum_iterations=20)
        solver.host_convergence = True
        solver._dense_jacobian_mode = "reverse"
        root, residual, ok, _ = solver.solve(function, x, x * 0, x * 4)
        assert bool(ok) and residual.abs().max() < 2e-11
        assert bounds[0] is not None
        torch.testing.assert_close(root, coefficients.sqrt())
        first = execution.jacobian(function, root)
        saved = first.clone()
        coefficients.add_(1)
        changed, residual, ok, _ = solver.solve(function, x * 1.5, x * 0, x * 4)
        assert bool(ok) and residual.abs().max() < 2e-11
        torch.testing.assert_close(changed, coefficients.sqrt())
        second = execution.jacobian(function, changed)
        torch.testing.assert_close(second, torch.diag(2 * changed))
        torch.testing.assert_close(first, saved, rtol=0, atol=0)
        assert execution.stats["captures"] == 2  # One primal and one derivative graph.
        assert execution.stats["replays"] > 2
        assert execution.stats["verified_calls"] >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires accessible CUDA")
def test_gpu_graph_admission_budget_retains_original_result():
    x = torch.ones(2, dtype=torch.float64, device="cuda")
    coefficients = x.new_full((2,), 2)
    with cuda_graph_execution(maximum_graphs=1, minimum_size=1) as execution, torch.no_grad():
        value = execution.residual_values(lambda z: z.square() - coefficients, x)
        jac = execution.jacobian(lambda z: z.square() - coefficients, x)
        torch.testing.assert_close(value, x.square() - coefficients)
        torch.testing.assert_close(jac, torch.diag(2 * x))
        assert execution.stats["captures"] == 1
        assert execution.stats["fallback_reasons"]["capture_admission"] >= 1
        assert len(execution.entries) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires accessible CUDA")
def test_gpu_replay_across_streams_cannot_overwrite_previous_output():
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cuda")
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    outputs = []
    function = lambda z: z**3
    with cuda_graph_execution(minimum_size=1) as execution, torch.no_grad():
        for i in range(6):
            with torch.cuda.stream(streams[i % 2]):
                outputs.append(execution.jacobian(function, x + i))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for i, result in enumerate(outputs):
            torch.testing.assert_close(result, torch.diag(3 * (x + i).square()))
        assert execution.stats["captures"] == 1


def test_wrong_kernel_is_rejected_before_returning_to_newton(monkeypatch):
    runtime = CUDAExecution(minimum_size=1)
    monkeypatch.setattr(runtime, "eligible", lambda x: True)
    runtime.compiled["jacobian"] = lambda fn, x: torch.zeros((len(x), len(x)))
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="verification failed"):
        actual = runtime.jacobian(lambda z: z.square(), x)
    torch.testing.assert_close(actual, torch.diag(2 * x))
    assert runtime.disabled
    assert runtime.stats["fallback_reasons"] == {"numerical_verification": 1}


def test_unsupported_compilation_retains_original_ad(monkeypatch):
    runtime = CUDAExecution(minimum_size=1)
    monkeypatch.setattr(runtime, "eligible", lambda x: True)

    def unsupported(fn, x):
        raise torch._dynamo.exc.Unsupported("deliberately unsupported kernel")

    runtime.compiled["jacobian"] = unsupported
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="original AD retained"):
        actual = runtime.jacobian(lambda z: z.square(), x)
    torch.testing.assert_close(actual, torch.diag(2 * x))
    assert runtime.disabled and runtime.stats["fallback_calls"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires accessible CUDA")
def test_gpu_live_parameter_and_structural_branch_guards():
    x = torch.tensor([0.7, 1.3], dtype=torch.float64, device="cuda")
    coefficient = x.new_tensor(1.5)
    mode = {"cubic": False}

    def function(z):
        return coefficient * (z**3 if mode["cubic"] else z.square())

    with cuda_graph_execution(minimum_size=1) as execution, torch.no_grad():
        first = execution.jacobian(function, x)
        torch.testing.assert_close(first, torch.diag(coefficient * 2 * x))
        coefficient.add_(0.25)
        second = execution.jacobian(function, x)
        torch.testing.assert_close(second, torch.diag(coefficient * 2 * x))
        assert execution.stats["captures"] == 1
        mode["cubic"] = True
        third = execution.jacobian(function, x)
        torch.testing.assert_close(third, torch.diag(coefficient * 3 * x.square()))
        assert execution.stats["captures"] == 2
        torch.testing.assert_close(first, torch.diag(1.5 * 2 * x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires accessible CUDA")
def test_gpu_colored_sparse_values_never_form_dense_jacobian(monkeypatch):
    from surface.torch_linear import ColoredJacobian

    def forbidden(*args, **kwargs):
        raise AssertionError("Colored kernels must not call dense jacrev")

    monkeypatch.setattr(torch.func, "jacrev", forbidden)
    size = 2048
    x = torch.linspace(0.7, 1.3, size, device="cuda", dtype=torch.float64)
    coloring = ColoredJacobian(size, range(size), range(size), "cuda")
    with cuda_graph_execution() as execution, torch.no_grad():
        first = execution.colored_values(coloring, lambda z: z.square(), x)
        second = execution.colored_values(coloring, lambda z: z.square(), x + 0.1)
        assert first.shape == second.shape == (size,)
        torch.testing.assert_close(first, 2 * x)
        torch.testing.assert_close(second, 2 * (x + 0.1))
        assert execution.stats["captures"] >= 1
        assert execution.stats["replays"] >= 2
        assert not execution.disabled


def test_exact_graph_optimization_preserves_live_mutations_and_derivatives():
    from torch.fx.experimental.proxy_tensor import make_fx

    from surface.cuda_execution import optimize_aten

    x = torch.tensor([0.7, 1.3], dtype=torch.float64)

    def pure(z):
        unused = z.sin()  # Deliberate dead work, independent of the result.
        return z.square() + z.square() + unused * 0

    plain = make_fx(pure)(x)
    expected = plain(x)
    tangent = torch.func.jacrev(plain)(x)
    optimized = optimize_aten(plain)
    torch.testing.assert_close(optimized(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.func.jacrev(optimized)(x), tangent, rtol=0, atol=0)

    def mutations(z):
        local = z.clone()
        before = local.square()
        local.add_(1)
        after = local.square()
        z.log()  # A genuinely dead expression.
        return before + after

    graph = make_fx(mutations)(x)
    expected = graph(x)
    count = len(list(graph.graph.nodes))
    graph = optimize_aten(graph)
    assert len(list(graph.graph.nodes)) < count
    torch.testing.assert_close(graph(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(x, torch.tensor([0.7, 1.3], dtype=x.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires accessible CUDA")
def test_gpu_lifted_literal_buffers_remain_alive_during_replay():
    x = torch.tensor([0.7, 1.3], dtype=torch.float64, device="cuda")

    def function(z):
        return z * z.new_tensor([0.2, 0.8])

    with cuda_graph_execution(minimum_size=1) as execution, torch.no_grad():
        first = execution.residual_values(function, x)
        assert any(v.is_cuda for v in execution.entries[0]["module"].buffers())
        # Allocate same-size tensors to expose freed literal storage if the
        # generated ATen module is not pinned for the graph's whole lifetime.
        garbage = [torch.full_like(x, -i) for i in range(20)]
        second = execution.residual_values(function, x + 1)
        torch.testing.assert_close(first, x * x.new_tensor([0.2, 0.8]))
        torch.testing.assert_close(second, (x + 1) * x.new_tensor([0.2, 0.8]))
        assert len(garbage) == 20 and execution.stats["captures"] == 1
