"""Reproduce the bounded dense CPU pair using the saved pre-change solver."""

import gzip
import json
import statistics
import time
import types
from pathlib import Path

import torch

from surface.torch_nonlinear import TorchNewton


def main():
    legacy = types.ModuleType("surface._baseline_nonlinear")
    legacy.__package__ = "surface"
    code = gzip.decompress(Path(__file__).with_name("baseline_torch_nonlinear.py.gz").read_bytes())
    exec(compile(code, "baseline_torch_nonlinear.py", "exec"), legacy.__dict__)
    start = torch.zeros(128, dtype=torch.float64)
    matrix = torch.eye(128, dtype=torch.float64) * 2
    indices = torch.arange(127)
    matrix[indices, indices + 1] = 0.2
    matrix[indices + 1, indices] = -0.1
    target = torch.linspace(0.5, 1.5, 128, dtype=torch.float64)

    def residual(x):
        return matrix @ (x + 0.01 * x.square() - target)

    def run(cls, repeats):
        before = time.perf_counter()
        for _ in range(repeats):
            solver = cls("cpu", maximum_iterations=20, dense_limit=128)
            solver.host_convergence = True
            x, r, ok, _ = solver.solve(
                residual, start, start - 10, start + 10, tolerance=3e-12
            )
            assert bool(ok) and float(r.abs().max()) <= 3e-12
        return time.perf_counter() - before, x

    for cls in (legacy.TorchNewton, TorchNewton):
        run(cls, 2)
    samples = {"before": [], "after": []}
    solutions = {}
    order = [("before", legacy.TorchNewton), ("after", TorchNewton)]
    for pair in range(6):
        for label, cls in order if pair % 2 == 0 else reversed(order):
            elapsed, solutions[label] = run(cls, 15)
            samples[label].append(elapsed)
    assert torch.equal(solutions["before"], solutions["after"])
    print(json.dumps({
        "scope": "128-variable dense CPU kernel; not network throughput",
        "samples_s": samples,
        "speedup": statistics.median(samples["before"]) / statistics.median(samples["after"]),
        "identical_solution": True,
    }, indent=2))


if __name__ == "__main__":
    main()
