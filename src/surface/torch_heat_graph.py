"""Bounded reusable CUDA graphs of the unchanged caloric linear step.

Operator coefficients, masses, controls and optional LU factors are live
inputs, not captured constants. Outputs are packed/cloned before another
replay so accepted histories and rejected trials never alias graph buffers.
"""

from collections import OrderedDict

import torch


class TorchHeatLinearGraphs:
    def __init__(self, equations, capacity=4):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Positive integer thermal graph cache capacity required")
        self.e = equations
        self.capacity = capacity
        self.entries = OrderedDict()
        self.captures = self.replays = 0
        self.input_bytes = self.input_packets = 0

    def linear(self, t0, m0, m1, dt, forcing, diagonal, dense, boundary, ambient, factors):
        e = self.e
        if e.device.type != "cuda":
            raise ValueError("Thermal graph execution requires CUDA")
        arrays = (
            t0,
            m0,
            m1,
            dt,
            forcing,
            diagonal,
            boundary,
            ambient,
            e.operator.values,
            e.incoming.values,
            e.outgoing.values,
        )
        if dense is not None:
            arrays += (dense,)
        if factors is not None:
            arrays += tuple(factors)
        energy_offset = (e.size + 31) // 32 * 32
        key = (
            dense is not None,
            factors is not None,
            e.iterations,
            id(getattr(e.linear_step, "__func__", e.linear_step)),
            tuple((tuple(t.shape), t.dtype, t.device) for t in arrays),
        )
        if key not in self.entries:
            # Model-owned cache: no process-global graph or mutable model lookup.
            groups = {}
            for index, value in enumerate(arrays):
                groups.setdefault(value.dtype, []).append(index)
            blocks = {
                dtype: torch.cat([arrays[i].detach().reshape(-1) for i in indices])
                for dtype, indices in groups.items()
            }
            static = [None] * len(arrays)
            for dtype, indices in groups.items():
                offset = 0
                for i in indices:
                    static[i] = blocks[dtype][offset : offset + arrays[i].numel()].view(
                        arrays[i].shape
                    )
                    offset += arrays[i].numel()
            static = tuple(static)
            # Preserve the allocation alignment of the eager delta vector.
            # An unaligned slice changes CUDA reduction order by one ulp even
            # when every cell value agrees; accepted energy journals must agree.
            padding = torch.zeros(energy_offset - e.size, dtype=e.dtype, device=e.device)
            old_values = e.operator.values, e.incoming.values, e.outgoing.values

            def calculate():
                t, a, b, h, f, d, bc, ac, op, incoming, outgoing = static[:11]
                e.operator.values, e.incoming.values, e.outgoing.values = op, incoming, outgoing
                matrix = static[11] if dense is not None else None
                lu = static[12 if dense is not None else 11 :] if factors is not None else None
                result = e.linear_step(t, a, b, h, f, d, matrix, bc, ac, lu)
                return torch.cat(
                    (
                        result[0],
                        padding,
                        result[1],
                        torch.stack((*result[2:-1], result[-1].to(e.dtype))),
                    )
                )

            with torch.no_grad(), torch.cuda.device(e.device):
                stream = torch.cuda.Stream(device=e.device)
                stream.wait_stream(torch.cuda.current_stream(e.device))
                try:
                    with torch.cuda.stream(stream):
                        calculate()
                        calculate()
                    torch.cuda.current_stream(e.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = calculate()
                    torch.cuda.current_stream(e.device).wait_stream(stream)
                finally:
                    e.operator.values, e.incoming.values, e.outgoing.values = old_values
            self.entries[key] = (static, graph, output, e.linear_step, blocks, groups)
            self.captures += 1
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        static, graph, output, _method, blocks, groups = self.entries[key]
        self.entries.move_to_end(key)
        with torch.no_grad(), torch.cuda.device(e.device):
            # One device-side gather per dtype replaces a separate launch for
            # every pressure/mass/operator/factor tensor. Integer pivots retain
            # their native dtype and never pass through floating point.
            for dtype, indices in groups.items():
                torch.cat([arrays[i].detach().reshape(-1) for i in indices], out=blocks[dtype])
            graph.replay()
            packed = output.clone()
        self.replays += 1
        self.input_packets += len(groups)
        self.input_bytes += sum(t.numel() * t.element_size() for t in arrays)
        n = e.size
        scalars = packed[energy_offset + n :]
        return (
            packed[:n],
            packed[energy_offset : energy_offset + n],
            scalars[0],
            scalars[1],
            scalars[2],
            scalars[3],
            scalars[4].to(torch.bool),
        )
