"""CPU block-only PR thermal-ring scaling; global dense AD/LU forbidden.

This fixed-step algebraic stress probe does not qualify transient accuracy,
multiphase hydraulic scaling, full-facility performance or GPU execution.
"""

import argparse
import gc
import gzip
import hashlib
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from audit_pr_collocation import audit_gauss_step
from benchmark_pr_block_preconditioner import fixture, step, trace_linear_solves
from check_pr_collocation import ROOT

from surface.torch_backend import runtime_fingerprint


@contextmanager
def forbid_global_dense():
    original_jac = torch.func.jacfwd
    original_lu = torch.linalg.lu_factor_ex

    def jacobian(function, *args, **kwargs):
        derivative = original_jac(function, *args, **kwargs)

        def checked(*points, **options):
            if any(torch.is_tensor(p) and p.numel() > 256 for p in points):
                raise AssertionError("Global dense Jacobian in sparse scaling probe")
            return derivative(*points, **options)

        return checked

    def factor(matrix, *args, **kwargs):
        if matrix.shape[-1] > 256:
            raise AssertionError("Global dense LU in sparse scaling probe")
        return original_lu(matrix, *args, **kwargs)

    with (
        patch.object(torch.func, "jacfwd", jacobian),
        patch.object(torch.linalg, "lu_factor_ex", factor),
    ):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        torch.get_num_threads() != 1
        or not 1 <= args.repeats <= 5
        or not args.counts
        or any(not 4 <= n <= 512 for n in args.counts)
        or len(set(args.counts)) != len(args.counts)
    ):
        parser.error("One CPU thread, unique counts 4..512 and 1..5 repeats required")
    if (args.output / "summary.json").exists():
        parser.error("Choose a fresh output directory")
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=True)
    model, state = fixture(4)
    step(model, state, "radau3", True)
    rows = []
    for count in args.counts:
        for repeat in range(args.repeats):
            model, state = fixture(count)
            gc.collect()
            with forbid_global_dense(), trace_linear_solves() as trace:
                start = time.perf_counter()
                final, record = step(model, state, "radau3", True)
                elapsed = time.perf_counter() - start
            assert final.to_dict() == record["state"]
            assert audit_gauss_step(
                model.to_dict(),
                state.to_dict(),
                record,
                model.fluid,
                {"openings": model.default_openings},
            )["accepted"]
            assert all(
                x["success"] and x["stored_entries"] is not None and x["largest_block"] <= 256
                for x in trace
            )
            row = {
                "nodes": count,
                "repeat": repeat,
                "elapsed_s": elapsed,
                "linear_solves": trace,
                "max_scaled_residual": record["max_scaled_residual"],
                "independent_step_audit_passed": True,
                "global_dense_ad_lu_forbidden": True,
            }
            rows.append(row)
            with gzip.open(args.output / f"n{count}_repeat{repeat}.json.gz", "wt") as f:
                json.dump(
                    {
                        "model": model.to_dict(),
                        "initial": state.to_dict(),
                        "record": record,
                        "measurement": row,
                    },
                    f,
                    allow_nan=False,
                )
            print(json.dumps(row), flush=True)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    summary = {
        "scope": __doc__,
        "fixed_step_s": 0.02,
        "repeats": args.repeats,
        "rows": rows,
        "median_elapsed_s": {
            n: statistics.median(r["elapsed_s"] for r in rows if r["nodes"] == n)
            for n in args.counts
        },
        "runtime": runtime_fingerprint(),
        "cpu_threads": 1,
        "source_sha256": hashes,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["median_elapsed_s"], indent=2))


if __name__ == "__main__":
    main()
