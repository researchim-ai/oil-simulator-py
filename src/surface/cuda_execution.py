"""Bounded, scoped CUDA execution for pure Newton kernels.

Dynamo guards model/phase/control structure; current states and coefficients
remain graph inputs. ATen lowering removes AD interpreter state before capture.
Sparse differentiation retains colored JVPs; sensitivities keep their eager path.
"""

import threading
import warnings
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_ACTIVE = ContextVar("surface_cuda_execution", default=None)


def native_literal_factories(module):
    """Replace only uniform literal uploads by identical device allocations."""
    changed = 0
    for node in module.graph.nodes:
        if node.op != "call_method" or node.target != "new_tensor" or len(node.args) != 2:
            continue
        literal = node.args[1]
        if type(literal) in (float, int, bool):
            shape, value = (), literal
        elif (
            isinstance(literal, (tuple, list))
            and literal
            and all(type(v) in (float, int, bool) for v in literal)
            and all(type(v) is type(literal[0]) and v == literal[0] for v in literal)
            and all(type(v) is not float or v.hex() == literal[0].hex() for v in literal)
        ):
            shape, value = (len(literal),), literal[0]
        else:
            continue
        node.target, node.args = "new_full", (node.args[0], shape, value)
        changed += 1
    module.graph.lint()
    module.recompile()
    return changed


def _linearize(function, x):
    if hasattr(function, "runtime_residual"):
        return torch.func.jacrev(function.runtime_residual)(x)
    if hasattr(function, "runtime_values"):
        # The public residual also updates Newton's arithmetic error budget.
        # Differentiate its pure tensor kernel, returning that budget as aux;
        # tracing the stateful wrapper would freeze or repeat its side effects.
        return torch.func.jacrev(function.runtime_values, has_aux=True)(x)[0]
    return torch.func.jacrev(function)(x)


def optimize_aten(module):
    """Prune dead diagnostics and exact CSE, respecting mutation regions."""
    from torch._functorch.compile_utils import fx_graph_cse

    module.graph.eliminate_dead_code()
    module.graph = fx_graph_cse(module.graph)
    module.graph.eliminate_dead_code()
    module.graph.lint()
    module.recompile()
    return module


def _values(function, x):
    return function(x)


def _colored_values(coloring, function, x):
    if hasattr(function, "runtime_values"):
        return coloring.values(lambda z: function.runtime_values(z)[0], x)
    return coloring.values(function, x)


def current_execution():
    return _ACTIVE.get()


class CUDAExecution:
    """One caller-owned cache; no model state or tensor storage survives close.

    Graph admission is capped rather than repeatedly evicting and recapturing
    rare phase patterns. First use is checked against independent eager AD.
    All replay inputs are copied, including tensors changed through aliases.
    """

    def __init__(
        self,
        *,
        maximum_graphs=4,
        maximum_bytes=256 * 2**20,
        minimum_size=64,
        maximum_size=4096,
        progress=None,
    ):
        for value in (maximum_graphs, maximum_bytes, minimum_size, maximum_size):
            if type(value) is not int or value < 1:
                raise ValueError("Positive integer CUDA execution budgets required")
        if minimum_size > maximum_size:
            raise ValueError("Minimum CUDA kernel size exceeds maximum")
        if progress is not None and not callable(progress):
            raise ValueError("CUDA execution progress must be callable")
        self.progress = progress
        self.maximum_graphs, self.maximum_bytes = maximum_graphs, maximum_bytes
        self.minimum_size = minimum_size
        self.maximum_size = maximum_size
        self.lock = threading.RLock()
        self.entries = []
        self.disabled = False
        self.compiled = {}
        self.stats = {
            "captures": 0,
            "replays": 0,
            "graphs_traced": 0,
            "verified_calls": 0,
            "fallback_calls": 0,
            "resident_bytes": 0,
            "native_literal_factories": 0,
            "fallback_reasons": {},
        }
        self._verified_generation = {}

    def statistics(self):
        return {
            **self.stats,
            "fallback_reasons": dict(self.stats["fallback_reasons"]),
            "maximum_graphs": self.maximum_graphs,
            "maximum_bytes": self.maximum_bytes,
            "minimum_size": self.minimum_size,
            "maximum_size": self.maximum_size,
            "linearization_ad": "reverse_dense_or_colored_jvp",
            "disabled": self.disabled,
        }

    def _fallback(self, reason):
        self.stats["fallback_calls"] += 1
        reasons = self.stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1

    def eligible(self, x):
        return (
            not self.disabled
            and x.is_cuda
            and x.numel() >= self.minimum_size
            and x.numel() <= self.maximum_size
            and not torch.is_grad_enabled()
            and not x.requires_grad
            and not torch._C._functorch.is_functorch_wrapped_tensor(x)
        )

    def _backend(self, module, inputs):
        self.stats["graphs_traced"] += 1
        self.stats["native_literal_factories"] += native_literal_factories(module)
        if self.progress is not None:
            self.progress({"event": "cuda_graph_trace", "nodes": len(list(module.graph.nodes))})
        return self._compile_aten(module, inputs)

    def _compile_aten(self, module, inputs):
        holder = {}

        def run(*values):
            if "skip" in holder:
                self._fallback(holder["skip"])
                return module(*values)
            if "graph" not in holder:
                if (
                    len(self.entries) >= self.maximum_graphs
                    or not values
                    or not all(
                        isinstance(v, torch.Tensor) and v.is_cuda and not v.requires_grad
                        for v in values
                    )
                    or len({v.device for v in values}) != 1
                ):
                    holder["skip"] = "capture_admission"
                    self._fallback("capture_admission")
                    return module(*values)
                device = values[0].device
                with torch.cuda.device(device):
                    static = [v.detach().clone() for v in values]
                    # Dynamo already guards Python structure and lifts live
                    # tensors. Lower that graph without training alias analysis.
                    from torch.fx.experimental.proxy_tensor import make_fx

                    depth = torch._C._functorch.get_dynamic_layer_stack_depth()
                    try:
                        aten = make_fx(
                            module, tracing_mode="real", _error_on_data_dependent_ops=True
                        )(*static)
                    except (RuntimeError, NotImplementedError) as exc:
                        if torch._C._functorch.get_dynamic_layer_stack_depth() > depth:
                            torch._C._functorch.pop_dynamic_layer_stack_and_undo_to_depth(depth)
                        raise torch._dynamo.exc.Unsupported(
                            "CUDA kernel lowering failed: " + str(exc).splitlines()[0]
                        ) from exc
                    # Pure ATen graph: remove unused diagnostic calculations
                    # and duplicate expressions without reordering arithmetic
                    # or reusing buffers with different numerical values.
                    aten = optimize_aten(aten)
                    if self.progress is not None:
                        self.progress(
                            {"event": "cuda_graph_lowered", "nodes": len(list(aten.graph.nodes))}
                        )
                    stream = torch.cuda.Stream(device=device)
                    current = torch.cuda.current_stream(device)
                    stream.wait_stream(current)
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            aten(*static)
                    stream.synchronize()
                    before = torch.cuda.memory_allocated(device)
                    before_reserved = torch.cuda.memory_reserved(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = aten(*static)
                    current.wait_stream(stream)
                    used = max(0, torch.cuda.memory_allocated(device) - before)
                    used = max(used, torch.cuda.memory_reserved(device) - before_reserved)
                    used += sum(v.numel() * v.element_size() for v in static)
                    used += sum(v.numel() * v.element_size() for v in aten.buffers())
                    if self.stats["resident_bytes"] + used > self.maximum_bytes:
                        holder["skip"] = "capture_memory_budget"
                        self._fallback("capture_memory_budget")
                        return module(*values)
                    holder.update(
                        graph=graph,
                        module=aten,
                        inputs=static,
                        outputs=outputs,
                        event=torch.cuda.Event(),
                        used=used,
                        replayed=False,
                    )
                    self.entries.append(holder)
                    self.stats["captures"] += 1
                    self.stats["resident_bytes"] += used
                    if self.progress is not None:
                        self.progress(
                            {
                                "event": "cuda_graph_captured",
                                "captures": self.stats["captures"],
                                "bytes": used,
                            }
                        )
            # Lock serializes host access; event orders accesses across streams.
            current = torch.cuda.current_stream(values[0].device)
            if holder["replayed"]:
                current.wait_event(holder["event"])
            for destination, source in zip(holder["inputs"], values):
                destination.copy_(source)
            holder["graph"].replay()
            outputs = tuple(v.clone() for v in holder["outputs"])
            holder["event"].record(current)
            holder["replayed"] = True
            self.stats["replays"] += 1
            return outputs

        return run

    def _execute(self, kind, function, x, coloring=None):
        eager = {"jacobian": _linearize, "residual": _values, "colored": _colored_values}[kind]
        arguments = (coloring, function, x) if kind == "colored" else (function, x)
        if not self.eligible(x):
            return eager(*arguments)
        with self.lock:
            if kind not in self.compiled:
                # Only the pure aperture primitive bypasses Dynamo; all tensors
                # are explicit inputs. No model or residual is made opaque.
                from .three_phase_properties import _DiskBandArea

                torch.compiler.allow_in_graph(_DiskBandArea)
                self.compiled[kind] = torch.compile(
                    eager, backend=self._backend, fullgraph=True, dynamic=False
                )
            try:
                actual = self.compiled[kind](*arguments)
            except (torch._dynamo.exc.Unsupported, torch._dynamo.exc.BackendCompilerFailed) as exc:
                self.disabled = True
                self._fallback(type(exc).__name__)
                warnings.warn(
                    "CUDA execution unavailable; original AD retained: " + str(exc).splitlines()[0],
                    RuntimeWarning,
                    stacklevel=2,
                )
                return eager(*arguments)
            generation = self.stats["captures"]
            if generation != self._verified_generation.get(kind):
                expected = eager(*arguments)
                try:
                    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=3e-10)
                except AssertionError:
                    self.disabled = True
                    self._fallback("numerical_verification")
                    warnings.warn(
                        "CUDA kernel verification failed; original AD retained",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return expected
                self._verified_generation[kind] = generation
                self.stats["verified_calls"] += 1
            return actual

    def jacobian(self, function, x):
        return self._execute("jacobian", function, x)

    def residual_values(self, function, x):
        return self._execute("residual", function, x)

    def colored_values(self, coloring, function, x):
        return self._execute("colored", function, x, coloring=coloring)

    def close(self):
        with self.lock:
            for entry in self.entries:
                if entry.get("replayed"):
                    entry["event"].synchronize()
                entry.clear()
            self.entries.clear()
            self.compiled.clear()
            self.disabled = True
            self.stats["resident_bytes"] = 0


@contextmanager
def cuda_graph_execution(**budgets):
    """Opt in for a complete simulation; nested calls share the bounded cache."""
    previous = current_execution()
    if previous is not None:
        yield previous
        return
    execution = CUDAExecution(**budgets)
    token = _ACTIVE.set(execution)
    try:
        yield execution
    finally:
        _ACTIVE.reset(token)
        execution.close()
