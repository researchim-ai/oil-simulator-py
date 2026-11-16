"""Same-initial-system work diagnostic; not a completed physical time step.

Stop after the first coupled Newton solve, before transport acceptance. Compare
legacy and operand-roundoff policies on unchanged full-facility equations.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch
from benchmark_journal import BenchmarkJournal
from run_pr_controlled_surface import run

from surface.pr_pump_connection import PumpConnectionBundle
from surface.torch_backend import runtime_fingerprint
from surface.torch_nonlinear import TorchNewton

ROOT = Path(__file__).resolve().parents[1]


class ProbeFinished(BaseException):
    """Private diagnostic stop; never a production convergence exception."""


def first_solve(config, enabled):
    original = TorchNewton.solve
    captured = {}

    def observed(self, function, initial, *args, **kwargs):
        start = time.perf_counter()
        point, residual, success, evaluations = original(self, function, initial, *args, **kwargs)
        elapsed = time.perf_counter() - start
        bounds = kwargs["roundoff_tolerance"]().detach()
        budget = torch.maximum(bounds, bounds.new_full((), kwargs["stagnation_tolerance"]))
        captured.update(
            elapsed_s=elapsed,
            unknowns=initial.numel(),
            initial_sha256=hashlib.sha256(
                json.dumps(initial.detach().tolist()).encode()
            ).hexdigest(),
            success=bool(success),
            stop_reason=self.stop_reason,
            nonlinear_evaluations=evaluations,
            jacobian_builds=self.jacobian_builds,
            jacobian_reuses=self.jacobian_reuses,
            point=point.detach().tolist(),
            residual=residual.detach().tolist(),
            roundoff_bounds=bounds.tolist(),
            max_scaled_residual=float(residual.abs().max()),
            max_budget_ratio=float((residual.abs() / budget).max()),
            finite=bool(torch.isfinite(point).all() & torch.isfinite(residual).all()),
        )
        assert bool(success) == (captured["finite"] and captured["max_budget_ratio"] <= 1)
        raise ProbeFinished()

    with (
        patch.object(PumpConnectionBundle, "_enthalpy_roundoff", enabled),
        patch.object(TorchNewton, "_roundoff_line_search", enabled),
        patch.object(TorchNewton, "solve", observed),
    ):
        try:
            run(config, device="cpu")
        except ProbeFinished:
            pass
    if not captured:
        raise RuntimeError("Full facility did not reach its first Newton solve")
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    if torch.get_num_threads() != 1:
        parser.error("Use OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1")
    config_path = ROOT / "configs/surface_pr_integrated_facility.json"
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__), config_path]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol = {
        "scope": __doc__,
        "source_sha256": hashes,
        "runtime": runtime_fingerprint(),
        "cpu_threads": 1,
        "repeats": args.repeats,
        "timing": "First Newton solve only; initialization and physical acceptance excluded",
    }
    journal = BenchmarkJournal(args.output, protocol)
    config = json.loads(config_path.read_text())
    rows = []
    for pair in range(args.repeats):
        for enabled in (False, True) if pair % 2 == 0 else (True, False):
            row = {"pair": pair, "mode": "roundoff" if enabled else "legacy"}
            row.update(first_solve(config, enabled))
            rows.append(row)
            assert row["initial_sha256"] == rows[0]["initial_sha256"]
            journal.checkpoint(rows, [], {})
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("point", "residual", "roundoff_bounds")
                    }
                ),
                flush=True,
            )
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    medians = {
        mode: statistics.median(r["elapsed_s"] for r in rows if r["mode"] == mode)
        for mode in ("legacy", "roundoff")
    }
    journal.complete({**protocol, "runs": rows, "median_elapsed_s": medians})
    print(json.dumps(medians))


if __name__ == "__main__":
    main()
