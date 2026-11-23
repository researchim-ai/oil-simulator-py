"""Scoped native CPU kernels for pure Newton residuals and forward derivatives.

No approximation, lookup table or host/device offload. Dynamo guards structure;
changing tensors remain inputs. Real ATen lowering precedes Inductor. Reverse AD
is deliberately not compiled: the installed compiler miscompiles its scatter
pullback on a two-phase PR fixture. Forward AD and colored JVPs are checked
against eager evaluations before use. Final Newton acceptance is eager.
"""

import threading
import time
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


class NativeExecution:
    eager_final = True

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
        self.disabled = False
        self._verified = {}
        self.stats = {
            "compilations": 0,
            "compiled_calls": 0,
            "verified_calls": 0,
            "fallback_calls": 0,
            "trace_nodes": 0,
            "resident_constant_bytes": 0,
            "compile_wall_s": 0.0,
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
        }

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
            self._fallback("kernel_admission")
            return module.forward
        if not all(
            isinstance(v, torch.Tensor) and v.device.type == "cpu" and not v.requires_grad
            for v in inputs
        ):
            self._fallback("non_numeric_inputs")
            return module.forward
        from torch._inductor.compile_fx import compile_fx
        from torch.fx.experimental.proxy_tensor import make_fx

        start = time.perf_counter()
        native_literal_factories(module)
        depth = torch._C._functorch.get_dynamic_layer_stack_depth()
        try:
            aten = make_fx(module, tracing_mode="real", _error_on_data_dependent_ops=True)(
                *[v.detach().clone() for v in inputs]
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
            self._fallback("graph_size")
            return module.forward
        constants = list(aten.buffers()) + list(aten.parameters())
        size = sum(v.numel() * v.element_size() for v in constants)
        if self.stats["resident_constant_bytes"] + size > self.maximum_constant_bytes:
            self._fallback("constant_budget")
            return module.forward
        if self.progress is not None:
            self.progress({"event": "native_kernel_compile", "nodes": nodes})
        compiled = compile_fx(
            aten,
            inputs,
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
        self.stats["compilations"] += 1
        self.stats["compile_wall_s"] += time.perf_counter() - start
        self.stats["resident_constant_bytes"] += size

        def run(*values):
            self.stats["compiled_calls"] += 1
            answer = compiled(*[v.detach().clone() for v in values])
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
        with self.lock:
            if kind not in self.compiled:
                from .three_phase_properties import _DiskBandArea

                torch.compiler.allow_in_graph(_DiskBandArea)
                self.compiled[kind] = torch.compile(
                    eager, backend=self._backend, fullgraph=True, dynamic=False
                )
            try:
                # The installed FakeTensor dispatch cache can reuse forward
                # view metadata from a different functorch level. This is a
                # tracing-only cache; numerical kernels/guards stay enabled.
                with torch._dynamo.config.patch(
                    fake_tensor_cache_enabled=False, recompile_limit=2 * self.maximum_kernels
                ):
                    actual = self.compiled[kind](*arguments)
            except (
                torch._dynamo.exc.Unsupported,
                torch._dynamo.exc.BackendCompilerFailed,
                torch._dynamo.exc.TorchRuntimeError,
                torch._dynamo.exc.FailOnRecompileLimitHit,
            ) as exc:
                self.disabled = True
                self._fallback(type(exc).__name__)
                warnings.warn(
                    "Native kernels unavailable; original AD retained: " + str(exc).splitlines()[0],
                    RuntimeWarning,
                    stacklevel=2,
                )
                return eager(*arguments)
            generation = self.stats["compilations"]
            if self._verified.get(kind) != generation:
                expected = eager(*arguments)
                try:
                    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=3e-10)
                except AssertionError:
                    self.disabled = True
                    self._fallback("numerical_verification")
                    warnings.warn(
                        "Native kernel verification failed; original AD retained",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return expected
                self._verified[kind] = generation
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
            self.compiled.clear()
            self.disabled = True
            self.stats["resident_constant_bytes"] = 0


@contextmanager
def native_kernel_execution(**budgets):
    with kernel_execution(NativeExecution(**budgets)) as execution:
        yield execution
