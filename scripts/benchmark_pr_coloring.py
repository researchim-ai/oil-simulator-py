"""Sparse global-balance derivative microbenchmark, not a plant speed claim."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from surface.torch_linear import ColoredJacobian, sparse_jacobian_plan


def benchmark(sizes, repeats=7):
    results = []
    for n in sizes:
        rows = [0] * n + list(range(1, n))
        cols = list(range(n)) + list(range(1, n))
        x = torch.linspace(0.1, 1.1, n, dtype=torch.float64)

        def function(z):
            return torch.cat((z.sin().sum().reshape(1), z[1:].square() + 2 * z[1:]))

        expected = torch.cat((x.cos(), 2 * x[1:] + 2))
        start = time.perf_counter()
        forward = ColoredJacobian(n, rows, cols, "cpu")
        forward_preparation = time.perf_counter() - start
        start = time.perf_counter()
        selected = sparse_jacobian_plan(n, rows, cols, "cpu", reverse=True)
        selected_preparation = time.perf_counter() - start
        plans = {"forward": forward, "selected": selected}
        samples = {name: [] for name in plans}
        errors = {}
        with torch.no_grad():
            for name, plan in plans.items():
                answer = plan.values(function, x)
                torch.testing.assert_close(answer, expected, rtol=2e-15, atol=2e-15)
                errors[name] = float((answer - expected).abs().max())
            for repeat in range(repeats):
                order = ("forward", "selected") if repeat % 2 == 0 else ("selected", "forward")
                for name in order:
                    start = time.perf_counter()
                    answer = plans[name].values(function, x)
                    samples[name].append(time.perf_counter() - start)
                    torch.testing.assert_close(answer, expected, rtol=2e-15, atol=2e-15)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        results.append(
            {
                "unknowns": n,
                "nonzeros": len(rows),
                "directions": {name: plan.color_count for name, plan in plans.items()},
                "preparation_s": {"forward": forward_preparation, "selected": selected_preparation},
                "samples_s": samples,
                "median_derivative_s": medians,
                "derivative_speedup": medians["forward"] / medians["selected"],
                "maximum_absolute_error_against_analytic": errors,
            }
        )
    return {
        "accepted": True,
        "scope": "synthetic sparse Jacobian with one global balance row, CPU float64; no full Jacobian or integration; does not measure full-plant/GPU/Norne speed",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[129, 513, 2049])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 2 for n in args.sizes):
        parser.error("Positive repeats and sizes >=2 required")
    result = benchmark(args.sizes, args.repeats)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
