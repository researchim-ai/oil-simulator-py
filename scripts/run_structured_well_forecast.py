"""Bounded evidence for actual producer-only FIM + structured gathering wells."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from benchmark_journal import BenchmarkJournal
from check_fim_well_forecast import audit
from run_fim_blackoil_surface import run_case

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--days", type=float, default=1)
    parser.add_argument("--report-days", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    c["fluid"]["pvt_path"] = str(ROOT / c["fluid"]["pvt_path"])
    if (
        c["simulation"].get("coupled_facility") is not None
        or c["simulation"].get("dynamic_receiver") is not None
    ):
        parser.error("This qualification isolates physical gathering/well coupling")
    paths = [
        *sorted((ROOT / "src/simulator").glob("*.py")),
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__),
        ROOT / "scripts/check_fim_well_forecast.py",
        ROOT / "scripts/run_fim_blackoil_surface.py",
        args.config.resolve(),
        Path(c["fluid"]["pvt_path"]),
    ]

    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    before = hashes()
    journal = BenchmarkJournal(
        args.output,
        {
            "scope": __doc__,
            "config": c,
            "source_sha256": before,
            "days": args.days,
            "report_days": args.report_days,
            "device": "cpu",
        },
    )
    journal.checkpoint([], [], {})
    network = c["simulation"]["physical_surface_network"]["network"]
    receiving = [n["pressure_pa"] for n in network["nodes"] if n.get("pressure_pa") is not None]
    if len(receiving) != 1:
        raise ValueError("One receiving pressure required")
    result = run_case(c, receiving[0], args.days, args.report_days, "cpu")
    path = args.output / "trajectory.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump({"config": c, "requested_days": args.days, "result": result}, f, allow_nan=False)
    checked = audit(c, result, expected_days=args.days)
    if before != hashes():
        raise RuntimeError("Source changed during qualification")
    summary = {
        "scope": __doc__,
        "audit": checked,
        "solve_wall_s": result["solve_wall_s"],
        "reservoir_cells": math.prod(c["reservoir"]["dimensions"]),
        "source_sha256": before,
        "sources_unchanged": True,
        "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    journal.complete(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
