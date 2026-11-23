"""Isolated CUDA-graph qualification of a coupled PR Jacobian.

This process-local experiment does not enable a production solver backend.
It checks a changed Newton point, physical state and dt against eager AD.
Reported kernel speed excludes capture and is not a whole-network speedup.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from check_pr_runtime_graph import ROOT, linearize, problem

from surface.cuda_execution import native_literal_factories
from surface.pr_gravity_transport import PRGravityTransportNetwork, PRGravityTransportState
from surface.three_phase_properties import _DiskBandArea


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=ROOT / "tests/data/surface/pr_restart_check_valve.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; run this check on an accessible GPU")
    torch.cuda.set_per_process_memory_fraction(0.30)
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__),
        ROOT / "scripts/check_pr_runtime_graph.py",
        args.fixture,
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    case = json.loads(args.fixture.read_text())
    model = PRGravityTransportNetwork.from_dict(case["model"], device="cuda")
    state = PRGravityTransportState.from_dict(case["state"])
    # Pure primitive with all tensor inputs explicit; local to this process.
    torch.compiler.allow_in_graph(_DiskBandArea)
    graphs, captures, factories = [], [], []

    def backend(module, inputs):
        factories.append(native_literal_factories(module))
        graphs.append(len(list(module.graph.nodes)))
        holder = {}

        def run(*values):
            if "graph" not in holder:
                if not all(isinstance(v, torch.Tensor) and v.is_cuda for v in values):
                    raise RuntimeError("Qualification requires explicit CUDA tensor inputs")
                static = [v.detach().clone() for v in values]
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        module(*static)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = module(*static)
                holder.update(graph=graph, inputs=static, outputs=outputs)
                captures.append(len(values))
            for destination, source in zip(holder["inputs"], values):
                destination.copy_(source)
            holder["graph"].replay()
            # Callers own their result; later replays cannot overwrite it.
            return tuple(v.clone() for v in holder["outputs"])

        return run

    compiled = torch.compile(linearize, backend=backend, fullgraph=True, dynamic=False)

    def timed(function, repeats=1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            result = function()
        torch.cuda.synchronize()
        return result, (time.perf_counter() - start) / repeats

    errors = []
    with torch.no_grad():
        y, function = problem(model, state, case, case["dt_s"])
        expected = linearize(function, y)
        actual, cold_s = timed(lambda: compiled(function, y))
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=3e-10)
        errors.append(float((actual - expected).abs().max().cpu()))
        # A distinct point must consume live data and preserve previous output.
        saved = actual.clone()
        shifted = y + 1e-7 * torch.linspace(-1, 1, len(y), device=y.device, dtype=y.dtype)
        shifted_actual = compiled(function, shifted)
        torch.testing.assert_close(
            shifted_actual, linearize(function, shifted), rtol=2e-10, atol=3e-10
        )
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        _, graph_s = timed(lambda: compiled(function, y), args.repeats)
        _, eager_s = timed(lambda: linearize(function, y), args.repeats)
        new, _ = model._step(
            state,
            case["dt_s"],
            case["openings"],
            case["heat_w"],
            case["utility"],
            case["losses"],
            drive_torque_nm=case["drive_torque_nm"],
        )
        y2, function2 = problem(model, new, case, case["dt_s"] * 1.3)
        actual2 = compiled(function2, y2)
        expected2 = linearize(function2, y2)
        torch.testing.assert_close(actual2, expected2, rtol=2e-10, atol=3e-10)
        errors.append(float((actual2 - expected2).abs().max().cpu()))
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in paths):
        raise RuntimeError("Sources changed during CUDA graph verification")
    result = {
        "accepted": len(graphs) == len(captures) == 1,
        "scope": "coupled dense Jacobian only; not a production backend or network benchmark",
        "cold_capture_s": cold_s,
        "warm_graph_s": graph_s,
        "eager_s": eager_s,
        "kernel_speedup": eager_s / graph_s,
        "graph_nodes": graphs,
        "capture_input_counts": captures,
        "native_literal_factories": factories,
        "max_abs_errors": errors,
        "changed_point_state_and_dt": True,
        "outputs_own_storage": True,
        "whole_network_speedup_qualified": False,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "source_sha256": hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
