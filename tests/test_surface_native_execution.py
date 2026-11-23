"""Native kernels preserve live coefficients, phase Jacobians and eager acceptance."""

import pytest
import torch

from surface.cuda_execution import cuda_graph_execution, current_execution
from surface.native_execution import NativeExecution, native_kernel_execution
from surface.torch_nonlinear import TorchNewton


def test_native_live_inputs_and_scope():
    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    coefficient = x.new_tensor([2.0, 3.0])

    def function(y):
        return coefficient * y.square() - 1

    function.runtime_residual = function
    with torch.no_grad(), native_kernel_execution() as engine:
        first = engine.jacobian(function, x)
        torch.testing.assert_close(first, torch.diag(2 * coefficient * x))
        kept = first.clone()
        coefficient.mul_(2)
        actual = engine.jacobian(function, x + 0.1)
        torch.testing.assert_close(actual, torch.diag(2 * coefficient * (x + 0.1)))
        torch.testing.assert_close(first, kept)
        with native_kernel_execution() as nested:
            assert nested is engine and current_execution() is engine
        assert engine.stats["compiled_calls"] >= 2
        assert not engine.disabled
    assert engine.disabled and not engine.entries and current_execution() is None


def test_parameter_sensitivities_keep_eager_ad():
    engine = NativeExecution()
    x = torch.ones(2, dtype=torch.float64)
    coefficient = torch.tensor(2.0, dtype=x.dtype, requires_grad=True)
    result = engine.jacobian(lambda z: coefficient * z.square(), x)
    torch.testing.assert_close(torch.autograd.grad(result.sum(), coefficient)[0], x.new_tensor(4.0))
    assert not engine.compiled


def test_fresh_step_closures_reuse_code_with_live_tensor_coefficients():
    x = torch.ones(3, dtype=torch.float64)

    def make(value):
        def residual(z):
            return value * z.square() - 1

        residual.runtime_residual = residual
        return residual

    with native_kernel_execution() as engine, torch.no_grad():
        for i in range(12):
            coefficient = x.new_full(x.shape, i + 1)
            actual = engine.jacobian(make(coefficient), x)
            torch.testing.assert_close(actual, torch.diag(2 * coefficient * x))
        assert not engine.disabled and engine.stats["fallback_calls"] == 0
        assert engine.stats["compilations"] == 1


def test_mixed_nested_engines_are_explicitly_rejected():
    with native_kernel_execution() as first:
        with pytest.raises(ValueError, match="Different Newton"), cuda_graph_execution():
            pass
        assert current_execution() is first and not first.disabled


@pytest.mark.parametrize("dense_limit", [0, 128])
def test_newton_final_acceptance_is_original_function(dense_limit):
    x = torch.ones(2, dtype=torch.float64)
    evaluations = []

    def pure(z):
        return z.square() - 2

    def function(z):
        if not torch._C._functorch.is_functorch_wrapped_tensor(z):
            evaluations.append(z.clone())
        return pure(z)

    function.runtime_residual = pure
    solver = TorchNewton("cpu", dense_limit=dense_limit)
    with torch.no_grad(), native_kernel_execution() as engine:
        root, row, ok, _ = solver.solve(
            function, x, x * 0, x * 4, pattern=([0, 1], [0, 1]), tolerance=3e-12
        )
        assert bool(ok) and not engine.disabled
        torch.testing.assert_close(row, pure(root), atol=0, rtol=0)
        torch.testing.assert_close(evaluations[-1], root, atol=0, rtol=0)
        assert engine.stats["compilations"] > 0


def test_two_phase_compiled_forward_matches_both_eager_ad_modes():
    from test_surface_pr_equilibrium_transport import pipe

    model, state = pipe()
    seeds = [model._seed(i, v) for i, (_, v) in enumerate(state.nodes)]
    y, _, _, evaluate, *_ = model._problem(state, seeds, 0.001, {"internal": True}, [0.0, 0.0])

    def row(z):
        return evaluate(z)[0]

    row.runtime_residual = row
    with torch.no_grad(), native_kernel_execution() as engine:
        actual = engine.jacobian(row, y)
        torch.testing.assert_close(actual, torch.func.jacfwd(row)(y), atol=3e-10, rtol=2e-10)
        torch.testing.assert_close(actual, torch.func.jacrev(row)(y), atol=3e-10, rtol=2e-10)
        assert not engine.disabled and engine.stats["compilations"] == 1


def test_multiphase_characteristic_tangents_remain_constrained():
    from test_surface_pr_equilibrium_transport import pipe

    model, state = pipe()
    entries = []
    for i, (_, value) in enumerate(state.nodes):
        data = model._state_data(i, value)
        active = torch.nonzero(model.fluid.tensor(value.component_moles) > 0).flatten()
        entries.append(model._characteristic_problem(i, data, active))
    y = torch.stack([e[0] for e in entries])
    function = entries[0][1]
    with torch.no_grad(), native_kernel_execution() as engine:
        actual = engine.characteristic_matrices(function, y)
        expected = torch.vmap(torch.func.jacfwd(function))(y)
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=3e-10)
        changed = y * 1.0000001
        torch.testing.assert_close(
            engine.characteristic_matrices(function, changed),
            torch.vmap(torch.func.jacfwd(function))(changed),
            rtol=2e-10,
            atol=3e-10,
        )
        assert not engine.disabled and engine.stats["compiled_calls"] > 0


def test_bad_compiler_result_disables_kernel_before_use(monkeypatch):
    engine = NativeExecution()
    x = torch.ones(2, dtype=torch.float64)
    engine.compiled["residual"] = lambda function, z: function(z) + 1
    with torch.no_grad(), pytest.warns(RuntimeWarning, match="verification"):
        actual = engine.residual_values(lambda z: z.square(), x)
    assert engine.disabled
    torch.testing.assert_close(actual, x.square())
