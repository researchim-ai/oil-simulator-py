"""Long-interval black-oil FIM + quasi-steady facility, with exact operating events.

Reuses the existing production gathering/plant feedback. No microscopic PR pipe
integration, steady-state inference for slugging, or Norne qualification implied.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

from benchmark_journal import BenchmarkJournal
from check_fim_facility_forecast import audit
from run_fim_blackoil_surface import run_case

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/fim_blackoil_facility_coupled.json"
    )
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--days", type=float, default=30)
    parser.add_argument("--report-days", type=float, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Select an explicit CUDA_VISIBLE_DEVICES; no automatic GPU selection")
    config = json.loads(args.config.read_text())
    if "coupled_facility" not in config["simulation"]:
        parser.error("Expected a quasi-steady coupled_facility configuration")
    config["fluid"]["pvt_path"] = str(ROOT / config["fluid"]["pvt_path"])
    events = [] if args.schedule is None else json.loads(args.schedule.read_text())
    paths = [
        *sorted((ROOT / "src/simulator").glob("*.py")),
        *sorted((ROOT / "src/surface").glob("*.py")),
        ROOT / "scripts/run_fim_blackoil_surface.py",
        Path(__file__),
        ROOT / "scripts/check_fim_facility_forecast.py",
        args.config.resolve(),
        Path(config["fluid"]["pvt_path"]),
    ]
    if args.schedule:
        paths.append(args.schedule.resolve())

    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    before = hashes()
    journal = BenchmarkJournal(
        args.output,
        {
            "scope": __doc__,
            "source_sha256": before,
            "days": args.days,
            "report_days": args.report_days,
            "device": args.device,
            "schedule": events,
        },
    )
    journal.checkpoint([], [], {})
    receiver = config["simulation"]["coupled_facility"]["facility"]["separator_stages"][0][
        "pressure_pa"
    ]
    result = run_case(
        config, receiver, args.days, args.report_days, args.device, facility_schedule=events
    )
    # Persist the trajectory before any audit. Failure never becomes accepted evidence.
    with gzip.open(args.output / "trajectory.json.gz", "wt") as handle:
        json.dump(
            {"config": config, "result": result, "schedule": events, "requested_days": args.days},
            handle,
            allow_nan=False,
        )
    checked = audit(config, result, expected_schedule=events, expected_days=args.days)
    if before != hashes():
        raise RuntimeError("Source changed during forecast")
    summary = {
        "scope": __doc__,
        "audit": checked,
        "solve_wall_s": result["solve_wall_s"],
        "device": args.device,
        "source_sha256": before,
        "sources_unchanged": True,
        "reservoir_cells": math.prod(config["reservoir"]["dimensions"]),
        "trajectory_sha256": hashlib.sha256(
            (args.output / "trajectory.json.gz").read_bytes()
        ).hexdigest(),
    }
    journal.complete(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
