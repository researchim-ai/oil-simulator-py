"""Scoped native CPU kernels for pure Newton residuals and forward derivatives.

No approximation, lookup table or host/device offload. Dynamo guards structure;
changing tensors remain inputs. Real ATen lowering precedes Inductor. Reverse AD
is deliberately not compiled: the installed compiler miscompiles its scatter
pullback on a two-phase PR fixture. Forward AD and colored JVPs are checked
against eager evaluations before use. Final Newton acceptance is eager.
"""

import threading
import time
import types
import warnings
from contextlib import contextmanager

import torch

from .cuda_execution import (
    _values,
    kernel_execution,
    native_literal_factories,
    optimize_aten,
)


def _forward(function, x):
    return torch.func.jacfwd(function)(x)


def _forward_aux(function, x):
    return torch.func.jacfwd(function, has_aux=True)(x)[0]


def _colored(coloring, function, x):
    return coloring.values(function, x)


def _colored_aux(coloring, function, x):
    return coloring.values(lambda z: function(z)[0], x)


def _characteristic_matrices(function, x):
    # Avoid the installed Dynamo nested-vmap forward-AD tracing bug. The
    # independent cells are unrolled only in the compiled graph, then fused.
    parts = [torch.func.jacfwd(function)(row) for row in x.unbind()]
    return tuple(torch.stack([part[k] for part in parts]) for k in range(2))


def _propagate_cancellation(error):
    """Compiler wrappers must not consume the caller's deadline or interrupt."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (TimeoutError, KeyboardInterrupt, SystemExit)):
            raise current
        for attr in ("inner_exception", "__cause__", "__context__"):
            nested = getattr(current, attr, None)
            if isinstance(nested, BaseException):
                pending.append(nested)


class NativeExecution:
    eager_final = True
    _reuse_input_buffers = True

    def __init__(
        self,
        *,
        maximum_kernels=8,
        maximum_nodes=50000,
        maximum_constant_bytes=256 * 2**20,
        minimum_size=1,
        maximum_size=4096,
        progress=None,
    ):
        for v in (
            maximum_kernels,
            maximum_nodes,
            maximum_constant_bytes,
            minimum_size,
            maximum_size,
        ):
            if type(v) is not int or v < 1:
                raise ValueError("Positive integer native execution budgets required")
        if minimum_size > maximum_size or (progress is not None and not callable(progress)):
            raise ValueError("Invalid native execution range/progress")
        self.maximum_kernels, self.maximum_nodes = maximum_kernels, maximum_nodes
        self.maximum_constant_bytes = maximum_constant_bytes
        self.minimum_size, self.maximum_size = minimum_size, maximum_size
        self.progress = progress
        self.lock = threading.RLock()
        self.entries, self.compiled = [], {}
        self._input_buffers = []
        self.blocked = {}
        self._active_operator = None
        self._last_kernel = None
        self.disabled = False
        self._verified = {}
        self.verification_failure = None
        self.stats = {
            "compilations": 0,
            "compiled_calls": 0,
            "verified_calls": 0,
            "verified_roundoff_calls": 0,
            "fallback_calls": 0,
            "trace_nodes": 0,
            "resident_constant_bytes": 0,
            "resident_input_bytes": 0,
            "input_buffer_allocations": 0,
            "input_buffer_updates": 0,
            "input_clone_allocations": 0,
            "compile_wall_s": 0.0,
            "lowering_wall_s": 0.0,
            "codegen_wall_s": 0.0,
            "trace_dispatch_wall_s": 0.0,
            "verification_wall_s": 0.0,
            "fallback_reasons": {},
        }

    def statistics(self):
        return {
            **self.stats,
            "fallback_reasons": dict(self.stats["fallback_reasons"]),
            "maximum_kernels": self.maximum_kernels,
            "maximum_nodes": self.maximum_nodes,
            "maximum_constant_bytes": self.maximum_constant_bytes,
            "disabled": self.disabled,
            "linearization_ad": "forward_dense_or_colored_jvp",
            "eager_final": True,
            "reuse_input_buffers": self._reuse_input_buffers,
            "operator_contexts": len(self.compiled),
            "maximum_operator_contexts": 4 * self.maximum_kernels,
            "verification_failure": self.verification_failure,
            "blocked_operators": [
                {
                    "kind": key[0],
                    "function": key[1].co_qualname,
                    "shape": list(key[2]),
                    "reason": reason,
                }
                for key, reason in self.blocked.items()
            ],
        }

    @staticmethod
    def _operator_key(kind, function, x):
        code = getattr(function, "__code__", None)
        if code is None:
            code = getattr(type(function).__call__, "__code__", _values.__code__)
        return kind, code, tuple(x.shape)

    def _block_operator(self, reason):
        if self._active_operator is not None:
            self.blocked[self._active_operator] = reason
        self._fallback(reason)

    @staticmethod
    def _linearization_roundoff(actual, expected):
        """Retain entrywise tolerance plus a small row-scale FP64 allowance.

        A small derivative can result from cancellation of much larger row
        terms. This extra budget is machine-epsilon based, not a relaxation
        of Newton, state-error or physical acceptance tolerances.
        """
        if (
            not isinstance(actual, torch.Tensor)
            or not isinstance(expected, torch.Tensor)
            or actual.shape != expected.shape
            or expected.ndim != 2
            or actual.dtype != expected.dtype
            or expected.dtype != torch.float64
        ):
            return False
        rounding = 64 * torch.finfo(expected.dtype).eps * expected.abs().amax(-1, keepdim=True)
        budget = 3e-10 + 2e-10 * expected.abs() + rounding
        return bool(
            (
                torch.isfinite(actual).all()
                & torch.isfinite(expected).all()
                & ((actual - expected).abs() <= budget).all()
            ).cpu()
        )

    def eligible(self, x):
        return (
            not self.disabled
            and x.device.type == "cpu"
            and x.dtype == torch.float64
            and self.minimum_size <= x.numel() <= self.maximum_size
            and not torch.is_grad_enabled()
            and not x.requires_grad
            and not torch._C._functorch.is_functorch_wrapped_tensor(x)
        )

    def _fallback(self, reason):
        self.stats["fallback_calls"] += 1
        reasons = self.stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1

    def _backend(self, module, inputs):
        if len(self.entries) >= self.maximum_kernels:
            self._block_operator("kernel_admission")
            return module.forward
        if not all(
            isinstance(v, torch.Tensor) and v.device.type == "cpu" and not v.requires_grad
            for v in inputs
        ):
            self._block_operator("non_numeric_inputs")
            return module.forward
        from torch._inductor.compile_fx import compile_fx
        from torch.fx.experimental.proxy_tensor import make_fx

        resident = self.stats["resident_constant_bytes"] + self.stats["resident_input_bytes"]
        input_size = sum(v.numel() * v.element_size() for v in inputs)
        if resident + input_size > self.maximum_constant_bytes:
            self._block_operator("input_budget")
            return module.forward
        start = time.perf_counter()
        # Compile the layout we actually own and replay. A clone can compact
        # a sliced input, so compiling against the original view's strides
        # would create a kernel that rejects its private working copy.
        buffers = [v.detach().clone() for v in inputs]
        # Retain Dynamo's original example identities whenever layout agrees:
        # replacing all examples triggers a functorch/FakeTensor metadata bug
        # in this installed compiler. Compact examples only for compact copies.
        examples = [
            value if value.stride() == buffer.stride() else buffer
            for value, buffer in zip(inputs, buffers, strict=True)
        ]
        native_literal_factories(module)
        depth = torch._C._functorch.get_dynamic_layer_stack_depth()
        try:
            aten = make_fx(module, tracing_mode="real", _error_on_data_dependent_ops=True)(
                # Keep AD tracing TensorImpls separate from the compiler's
                # examples: this PyTorch version can retain forward-view
                # metadata on tensors used by real-mode lowering.
                *[v.detach().clone() for v in buffers]
            )
        except (RuntimeError, NotImplementedError) as exc:
            if torch._C._functorch.get_dynamic_layer_stack_depth() > depth:
                torch._C._functorch.pop_dynamic_layer_stack_and_undo_to_depth(depth)
            raise torch._dynamo.exc.Unsupported(
                "Native ATen lowering: " + str(exc).splitlines()[0]
            ) from exc
        aten = optimize_aten(aten)
        nodes = len(list(aten.graph.nodes))
        self.stats["trace_nodes"] += nodes
        if nodes > self.maximum_nodes:
            self._block_operator("graph_size")
            return module.forward
        constants = list(aten.buffers()) + list(aten.parameters())
        size = sum(v.numel() * v.element_size() for v in constants)
        if resident + size > self.maximum_constant_bytes:
            self._block_operator("constant_budget")
            return module.forward
        if resident + size + input_size > self.maximum_constant_bytes:
            self._block_operator("input_budget")
            return module.forward
        if self.progress is not None:
            key = self._active_operator
            self.progress(
                {
                    "event": "native_kernel_compile",
                    "nodes": nodes,
                    "kind": key[0],
                    "function": key[1].co_qualname,
                    "shape": list(key[2]),
                }
            )
        lowered = time.perf_counter()
        self.stats["lowering_wall_s"] += lowered - start
        compiled = compile_fx(
            aten,
            examples,
            config_patches={
                "compile_threads": 2,
                "use_fast_math": False,
                "inplace_buffers": False,
                "allow_buffer_reuse": False,
                "cpp.enable_floating_point_contract_flag": "off",
            },
        )
        # Keep traced constants alive and protect caller-owned tensor storage.
        self.entries.append((aten, compiled))
        self._input_buffers.append(buffers)
        self.stats["compilations"] += 1
        finished = time.perf_counter()
        self.stats["codegen_wall_s"] += finished - lowered
        self.stats["compile_wall_s"] += finished - start
        self.stats["resident_constant_bytes"] += size
        self.stats["resident_input_bytes"] += input_size
        self.stats["input_buffer_allocations"] += len(buffers)
        kernel = self.stats["compilations"]
        # Guarded metadata is fixed for this kernel. Keep private storage and
        # refresh every value, including changing coefficients and aliases.
        # The enclosing execution lock serializes copies and replay; outputs
        # are cloned before these buffers can be overwritten by another call.

        def run(*values):
            self._last_kernel = kernel
            self.stats["compiled_calls"] += 1
            if self._reuse_input_buffers:
                for buffer, value in zip(buffers, values, strict=True):
                    buffer.copy_(value)
                self.stats["input_buffer_updates"] += len(buffers)
                owned = buffers
            else:
                owned = [v.detach().clone() for v in values]
                self.stats["input_clone_allocations"] += len(owned)
            answer = compiled(*owned)
            return torch.utils._pytree.tree_map(
                lambda v: v.clone() if isinstance(v, torch.Tensor) else v, answer
            )

        return run

    def _execute(self, kind, function, x, coloring=None):
        # Resolve the residual protocol OUTSIDE Dynamo. In particular a pure
        # function's runtime_residual may point to itself: tracing attribute
        # lookup there guards object identity and recompiles at every step.
        if kind in ("jacobian", "colored"):
            if hasattr(function, "runtime_residual"):
                function = function.runtime_residual
            elif hasattr(function, "runtime_values"):
                function = function.runtime_values
                kind += "_aux"
        eager = {
            "jacobian": _forward,
            "jacobian_aux": _forward_aux,
            "residual": _values,
            "colored": _colored,
            "colored_aux": _colored_aux,
            "characteristics": _characteristic_matrices,
        }[kind]
        arguments = (coloring, function, x) if kind.startswith("colored") else (function, x)
        if not self.eligible(x):
            return eager(*arguments)
        # A Radau endpoint is a view into the stage vector. Normalize storage
        # before Dynamo/forward AD: otherwise FakeTensor can give its tangent
        # a different view layout on a subsequent trace (PyTorch FW-view bug).
        x = x.detach().clone(memory_format=torch.contiguous_format)
        arguments = (coloring, function, x) if kind.startswith("colored") else (function, x)
        key = self._operator_key(kind, function, x)
        with self.lock:
            if key in self.blocked:
                self._fallback("blocked_operator")
                return eager(*arguments)
            if key not in self.compiled:
                if (
                    len(self.compiled) >= 4 * self.maximum_kernels
                    or len(self.entries) >= self.maximum_kernels
                ):
                    self._fallback("operator_admission")
                    return eager(*arguments)
                from .three_phase_properties import _DiskBandArea

                torch.compiler.allow_in_graph(_DiskBandArea)
                # A distinct code object gives this operator its OWN Dynamo
                # guard cache. Mixed equipment equations must not consume the
                # recompile budget of the pipe Jacobian's generic AD wrapper.
                frame = types.FunctionType(
                    eager.__code__.replace(),
                    eager.__globals__,
                    eager.__name__,
                    eager.__defaults__,
                    eager.__closure__,
                )
                self.compiled[key] = torch.compile(
                    frame, backend=self._backend, fullgraph=True, dynamic=False
                )
            self._active_operator = key
            self._last_kernel = None
            compilations_before = self.stats["compilations"]
            backend_before = self.stats["compile_wall_s"]
            dispatch_started = time.perf_counter()
            try:
                # The installed FakeTensor dispatch cache can reuse forward
                # view metadata from a different functorch level. This is a
                # tracing-only cache; numerical kernels/guards stay enabled.
                with torch._dynamo.config.patch(
                    fake_tensor_cache_enabled=False, recompile_limit=2 * self.maximum_kernels
                ):
                    actual = self.compiled[key](*arguments)
            except (
                torch._dynamo.exc.Unsupported,
                torch._dynamo.exc.BackendCompilerFailed,
                torch._dynamo.exc.TorchRuntimeError,
                torch._dynamo.exc.FailOnRecompileLimitHit,
            ) as exc:
                _propagate_cancellation(exc)
                self._block_operator(type(exc).__name__)
                warnings.warn(
                    "Native operator unavailable; original AD retained for "
                    + key[1].co_qualname
                    + ": "
                    + str(exc).splitlines()[0],
                    RuntimeWarning,
                    stacklevel=2,
                )
                return eager(*arguments)
            finally:
                self._active_operator = None
            if self.stats["compilations"] != compilations_before:
                self.stats["trace_dispatch_wall_s"] += max(
                    0.0,
                    time.perf_counter() - dispatch_started
                    - (self.stats["compile_wall_s"] - backend_before),
                )
            # An unrelated equipment kernel cannot invalidate this kernel's
            # numerical check. A new guarded specialization has its own token
            # and is always checked, including when a previous branch returns.
            verification = (key, self._last_kernel)
            if verification not in self._verified:
                verification_started = time.perf_counter()
                expected = eager(*arguments)
                self.stats["verification_wall_s"] += time.perf_counter() - verification_started
                try:
                    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=3e-10)
                except AssertionError:
                    if kind.startswith("jacobian") and self._linearization_roundoff(
                        actual, expected
                    ):
                        self.stats["verified_roundoff_calls"] += 1
                    else:
                        self.disabled = True
                        self._fallback("numerical_verification")
                        self.verification_failure = {
                            "kind": kind,
                            "function": key[1].co_qualname,
                            "shape": list(key[2]),
                        }
                        warnings.warn(
                            "Native kernel verification failed; original AD retained for "
                            + key[1].co_qualname,
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        return expected
                self._verified[verification] = True
                self.stats["verified_calls"] += 1
            return actual

    def jacobian(self, function, x):
        return self._execute("jacobian", function, x)

    def residual_values(self, function, x):
        return self._execute("residual", function, x)

    def colored_values(self, coloring, function, x):
        return self._execute("colored", function, x, coloring)

    def characteristic_matrices(self, function, x):
        return self._execute("characteristics", function, x)

    def close(self):
        with self.lock:
            self.entries.clear()
            for buffers in self._input_buffers:
                buffers.clear()
            self._input_buffers.clear()
            self.compiled.clear()
            self.blocked.clear()
            self.disabled = True
            self.stats["resident_constant_bytes"] = 0
            self.stats["resident_input_bytes"] = 0


@contextmanager
def native_kernel_execution(**budgets):
    with kernel_execution(NativeExecution(**budgets)) as execution:
        yield execution
