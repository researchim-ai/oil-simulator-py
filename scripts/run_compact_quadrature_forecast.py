"""Same full FIM and pressure grid, dense versus compact quadrature points.

Dense retains every interval's Gauss nodes. Compact retains degenerate-span
first/second AD contributions via midpoint samples; error budgets are unchanged.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import run_structured_well_forecast as driver
from dense_quadrature_reference import DenseQuadraturePlan

from surface import blackoil_pipe_batch
from surface.blackoil_quadrature_plan import CompactQuadraturePlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrature", choices=["dense", "compact"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--days", type=float, default=1)
    parser.add_argument("--report-days", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("dense_quadrature_reference.py").resolve(),
    ]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    original = blackoil_pipe_batch.make_quadrature_plan
    base = DenseQuadraturePlan if args.quadrature == "dense" else CompactQuadraturePlan
    plans = 0
    evaluations = 0
    points = 0
    maximum_points = 0

    class Counted(base):
        def __init__(self, *a, **kw):
            nonlocal plans
            plans += 1
            super().__init__(*a, **kw)

        def evaluate(self, *a, **kw):
            nonlocal evaluations, points, maximum_points
            old = self.point_evaluations
            result = super().evaluate(*a, **kw)
            delta = self.point_evaluations - old
            evaluations += 1
            points += delta
            maximum_points = max(maximum_points, delta)
            return result

    argv = sys.argv
    blackoil_pipe_batch.make_quadrature_plan = Counted
    try:
        sys.argv = [
            str(Path(driver.__file__)),
            "--config",
            str(args.config),
            "--days",
            str(args.days),
            "--report-days",
            str(args.report_days),
            "--output",
            str(args.output),
        ]
        driver.main()
    finally:
        blackoil_pipe_batch.make_quadrature_plan = original
        sys.argv = argv
    if hashes() != before or not evaluations:
        raise RuntimeError("Quadrature policy source changed or did not execute")
    with (args.output / "quadrature_policy.json").open("x") as handle:
        json.dump(
            {
                "scope": __doc__,
                "policy": args.quadrature,
                "planning_calls": plans,
                "quadrature_calls": evaluations,
                "evaluated_property_points": points,
                "maximum_points_per_property_pass": maximum_points,
                "source_sha256": before,
                "sources_unchanged": True,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
