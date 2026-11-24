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
        assert engine.stats["resident_input_bytes"] > 0
        assert not engine.disabled
    assert engine.disabled and not engine.entries and current_execution() is None
    assert engine.stats["resident_input_bytes"] == 0


def test_private_replay_buffers_preserve_live_views_and_output_ownership():
    storage = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    coefficient = storage[:, ::2]
    x = torch.ones_like(coefficient)

    def function(z):
        return coefficient * z

    with native_kernel_execution() as engine, torch.no_grad():
        old = engine.residual_values(function, x)
        allocations = engine.stats["input_buffer_allocations"]
        retained = old.clone()
        storage.add_(2)
        updated = engine.residual_values(function, x * 3)
        torch.testing.assert_close(updated, coefficient * 3, atol=0, rtol=0)
        torch.testing.assert_close(old, retained, atol=0, rtol=0)
        torch.testing.assert_close(x, torch.ones_like(x), atol=0, rtol=0)
        assert engine.stats["compilations"] == engine.stats["verified_calls"] == 1
        assert engine.stats["fallback_calls"] == 0
        assert engine.stats["input_buffer_allocations"] == allocations
        assert engine.stats["input_buffer_updates"] > allocations
        assert engine.stats["input_clone_allocations"] == 0
        engine._reuse_input_buffers = False
        storage.add_(2)
        torch.testing.assert_close(
            engine.residual_values(function, x * 4), coefficient * 4, atol=0, rtol=0
        )
        assert engine.stats["input_clone_allocations"] > 0
        assert engine.stats["compilations"] == engine.stats["verified_calls"] == 1


def test_replay_storage_shares_the_explicit_resident_memory_budget():
    x = torch.ones(8, dtype=torch.float64)
    with native_kernel_execution(maximum_constant_bytes=1) as engine, torch.no_grad():
        actual = engine.residual_values(torch.sin, x)
        torch.testing.assert_close(actual, x.sin())
        assert engine.stats["fallback_reasons"]["input_budget"] == 1
        assert engine.stats["resident_input_bytes"] == 0
        assert not engine.entries and not engine.disabled


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


def test_verification_belongs_to_kernel_not_unrelated_compilation():
    x = torch.ones(3, dtype=torch.float64)

    def make(power):
        def residual(z):
            return z.pow(power)

        return residual

    square, cube = make(2), make(3)
    with native_kernel_execution() as engine, torch.no_grad():
        torch.testing.assert_close(engine.jacobian(square, x), torch.diag(2 * x))
        assert engine.stats["verified_calls"] == 1
        torch.testing.assert_close(engine.residual_values(torch.sin, x), x.sin())
        assert engine.stats["verified_calls"] == 2
        torch.testing.assert_close(engine.jacobian(square, x * 2), torch.diag(4 * x))
        assert engine.stats["verified_calls"] == 2
        torch.testing.assert_close(engine.jacobian(cube, x * 2), torch.diag(12 * x))
        assert engine.stats["verified_calls"] == 3
        torch.testing.assert_close(engine.jacobian(square, x * 3), torch.diag(6 * x))
        assert engine.stats["verified_calls"] == 3
        assert engine.stats["compilations"] == 3 and not engine.disabled
        assert engine.stats["compile_wall_s"] == pytest.approx(
            engine.stats["lowering_wall_s"] + engine.stats["codegen_wall_s"]
        )
        assert engine.stats["trace_dispatch_wall_s"] >= 0
        assert engine.stats["verification_wall_s"] > 0


def test_new_specialization_is_verified_after_an_existing_valid_kernel(monkeypatch):
    import importlib

    compiler = importlib.import_module("torch._inductor.compile_fx")
    original = compiler.compile_fx
    x = torch.ones(2, dtype=torch.float64)

    def make(power):
        def residual(z):
            return z.pow(power)

        return residual

    def wrong_compile(*args, **kwargs):
        compiled = original(*args, **kwargs)

        def corrupt(*values):
            return torch.utils._pytree.tree_map(lambda v: v + 1, compiled(*values))

        return corrupt

    with native_kernel_execution() as engine, torch.no_grad():
        engine.jacobian(make(2), x)
        monkeypatch.setattr(compiler, "compile_fx", wrong_compile)
        with pytest.warns(RuntimeWarning, match="verification failed"):
            actual = engine.jacobian(make(3), x)
        torch.testing.assert_close(actual, torch.diag(3 * x.square()))
        assert engine.disabled and engine.stats["compilations"] == 2
        assert engine.stats["fallback_reasons"]["numerical_verification"] == 1


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
    function = lambda z: z.square()
    engine.compiled[engine._operator_key("residual", function, x)] = lambda fn, z: fn(z) + 1
    with torch.no_grad(), pytest.warns(RuntimeWarning, match="verification"):
        actual = engine.residual_values(function, x)
    assert engine.disabled
    torch.testing.assert_close(actual, x.square())


def test_unsupported_operator_does_not_disable_valid_equipment_kernels():
    x = torch.ones(2, dtype=torch.float64)

    @torch.compiler.disable
    def opaque(z):
        return z.sin()

    def unavailable(z):
        return opaque(z)

    def supported(z):
        return z.square()

    with torch.no_grad(), native_kernel_execution() as engine:
        torch.testing.assert_close(engine.jacobian(supported, x), torch.diag(2 * x))
        with pytest.warns(RuntimeWarning, match="operator unavailable"):
            actual = engine.jacobian(unavailable, x)
        torch.testing.assert_close(actual, torch.diag(x.cos()))
        torch.testing.assert_close(engine.jacobian(unavailable, x * 2), torch.diag((x * 2).cos()))
        before = engine.stats["compiled_calls"]
        torch.testing.assert_close(engine.jacobian(supported, x * 3), torch.diag(6 * x))
        assert engine.stats["compiled_calls"] > before
        assert not engine.disabled and len(engine.blocked) == 1
        assert engine.stats["fallback_reasons"]["blocked_operator"] == 1


def test_operator_and_kernel_admission_stays_bounded_and_keeps_existing_kernel():
    x = torch.ones(2, dtype=torch.float64)
    supported = lambda z: z.square()
    with torch.no_grad(), native_kernel_execution(maximum_kernels=1) as engine:
        engine.jacobian(supported, x)
        for size in range(1, 21):
            values = x.new_ones(size)
            torch.testing.assert_close(engine.residual_values(torch.sin, values), values.sin())
        assert len(engine.compiled) == len(engine.entries) == 1
        before = engine.stats["compiled_calls"]
        torch.testing.assert_close(engine.jacobian(supported, x * 2), torch.diag(4 * x))
        assert not engine.disabled and engine.stats["compiled_calls"] > before
        assert engine.stats["fallback_reasons"]["operator_admission"] == 20


def test_scaled_linearization_roundoff_does_not_admit_real_tangent_errors():
    matrix = torch.tensor([[1e6, 0.0], [0.0, 1.0]], dtype=torch.float64)
    rounded = matrix.clone()
    rounded[0, 1] = 1e-9
    assert NativeExecution._linearization_roundoff(rounded, matrix)
    rounded[0, 1] = 1e-4
    assert not NativeExecution._linearization_roundoff(rounded, matrix)
    rounded = matrix.clone()
    rounded[1, 0] = 1e-9
    assert not NativeExecution._linearization_roundoff(rounded, matrix)
    rounded[0, 0] = float("nan")
    assert not NativeExecution._linearization_roundoff(rounded, matrix)


def test_compiler_wrapped_deadline_propagates_without_fallback():
    x = torch.ones(2, dtype=torch.float64)
    function = lambda z: z.square()

    def interrupted(*args):
        raise torch._dynamo.exc.BackendCompilerFailed("probe", TimeoutError("deadline"), None)

    with native_kernel_execution() as engine, torch.no_grad():
        engine.compiled[engine._operator_key("jacobian", function, x)] = interrupted
        with pytest.raises(TimeoutError, match="deadline"):
            engine.jacobian(function, x)
        assert not engine.disabled and not engine.blocked and engine.stats["fallback_calls"] == 0


def test_column_samples_reuse_kernel_across_heights_and_thermodynamic_states():
    from test_surface_stratified_pr import binary

    from surface.separator_geometry import SeparatorGeometry
    from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState

    column = StratifiedPRColumn(binary("cpu"), SeparatorGeometry(0.1, 2.0, "vertical"))
    states = [
        StratifiedPRState.from_dict(column.initialize(1e5, t, [0.5, 0.5])["state"])
        for t in (320, 321, 322)
    ]
    points = [(s, h) for s in states for h in (0.05, 0.35, 0.65, 1.4)]
    expected = [column.sample(s, h) for s, h in points]
    column._samples.clear()
    with native_kernel_execution() as engine:
        actual = [column.sample(s, h) for s, h in points]
        assert not engine.disabled and engine.stats["fallback_calls"] == 0
        assert engine.stats["compilations"] == 1
    for a, b in zip(actual, expected):
        assert a["pressure_pa"] == pytest.approx(b["pressure_pa"], abs=1e-7)
        assert a["mole_fractions"] == pytest.approx(b["mole_fractions"], abs=1e-12)
