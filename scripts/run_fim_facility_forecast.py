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
import sys
from pathlib import Path

import torch

from benchmark_journal import BenchmarkJournal, atomic_json
from check_fim_facility_forecast import audit
from run_fim_blackoil_surface import run_case
from simulator.execution_memory import process_memory, enforce_host_memory

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
    parser.add_argument("--host-memory-limit-mib", type=float,
                        help="Optional reactive RSS+swap guard, checked at startup and accepted stages")
    parser.add_argument("--cuda-memory-fraction", type=float,
                        help="Optional per-process Torch allocator limit; excludes CUDA context")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.host_memory_limit_mib is not None and (
        not math.isfinite(args.host_memory_limit_mib) or args.host_memory_limit_mib <= 0
    ):
        parser.error("--host-memory-limit-mib must be positive and finite")
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Select an explicit CUDA_VISIBLE_DEVICES; no automatic GPU selection")
    if args.cuda_memory_fraction is not None:
        if args.device != "cuda" or not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= 1:
            parser.error("--cuda-memory-fraction requires CUDA and a finite fraction in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, 0)
    config = json.loads(args.config.read_text())
    if "coupled_facility" not in config["simulation"]:
        parser.error("Expected a quasi-steady coupled_facility configuration")
    config["fluid"]["pvt_path"] = str(ROOT / config["fluid"]["pvt_path"])
    events = [] if args.schedule is None else json.loads(args.schedule.read_text())
    paths = [
        *sorted((ROOT / "src/simulator").glob("*.py")),
        *sorted((ROOT / "src/surface").glob("*.py")),
        *sorted((ROOT / "src/solver").glob("*.py")),
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
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_memory_fraction": args.cuda_memory_fraction,
            "host_memory_limit_mib": args.host_memory_limit_mib,
            "schedule": events,
        },
    )
    journal.checkpoint([], [], {})
    progress_stream = sys.stdout

    last_progress_s = [0.]

    def progress(row):
        memory = process_memory()
        enforce_host_memory(args.host_memory_limit_mib, memory)
        row.update(memory)
        last_progress_s[0] = row["solve_wall_s"]
        journal.checkpoint([row], [], {})
        print(
            f"forecast {row['time_days']:.6g}/{row['requested_days']:.6g} days; "
            f"accepted={row['accepted_substeps']}; wall={row['solve_wall_s']:.2f}s; "
            f"rss_mib={None if memory['process_rss_bytes'] is None else round(memory['process_rss_bytes']/1024**2, 1)}",
            file=progress_stream,
            flush=True,
        )

    def accepted_progress(row):
        if args.host_memory_limit_mib is not None:
            enforce_host_memory(args.host_memory_limit_mib, process_memory())
        if row["solve_wall_s"] - last_progress_s[0] >= 10.:
            progress(row)

    receiver = config["simulation"]["coupled_facility"]["facility"]["separator_stages"][0][
        "pressure_pa"
    ]
    stage = "solve"
    try:
        enforce_host_memory(args.host_memory_limit_mib, process_memory())
        result = run_case(
            config,
            receiver,
            args.days,
            args.report_days,
            args.device,
            facility_schedule=events,
            progress_callback=progress,
            accepted_progress_callback=accepted_progress,
        )
        stage = "persist_trajectory"
        # Persist the trajectory before any audit. Failure never becomes accepted evidence.
        with gzip.open(args.output / "trajectory.json.gz", "wt") as handle:
            json.dump(
                {"config": config, "result": result, "schedule": events, "requested_days": args.days},
                handle,
                allow_nan=False,
            )
        stage = "accounting_audit"
        checked = audit(config, result, expected_schedule=events, expected_days=args.days)
        stage = "source_guard"
        if before != hashes():
            raise RuntimeError("Source changed during forecast")
        summary = {
            "scope": __doc__,
            "audit": checked,
            "solve_wall_s": result["solve_wall_s"],
            "device": args.device,
            "execution": result["execution"],
            "host_memory": process_memory(),
            "host_memory_limit_mib": args.host_memory_limit_mib,
            "host_memory_scope": "Current RSS and swap; RSS peak covers process lifetime including imports and output serialization. Guard is reactive, not a hard allocation cap.",
            "cuda_memory_fraction": args.cuda_memory_fraction,
            "source_sha256": before,
            "sources_unchanged": True,
            "reservoir_cells": math.prod(config["reservoir"]["dimensions"]),
            "trajectory_sha256": hashlib.sha256(
                (args.output / "trajectory.json.gz").read_bytes()
            ).hexdigest(),
        }
        stage = "publish_summary"
        journal.complete(summary)
        atomic_json(
            args.output / "progress.json",
            {
                "status": "complete",
                "time_days": result["elapsed_days"],
                "accepted_substeps": len(result["accepted_substeps"]),
                "solve_wall_s": result["solve_wall_s"],
                "summary": "summary.json",
            },
        )
        print(json.dumps(summary, indent=2))
    except (Exception, KeyboardInterrupt) as error:
        atomic_json(
            args.output / "failure.json",
            {
                "status": "failed",
                "exception": type(error).__name__,
                "stage": stage,
                "reason": str(error),
                "requested_days": args.days,
            },
        )
        raise


if __name__ == "__main__":
    main()
