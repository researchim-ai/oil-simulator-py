"""Qualify completed CPU/CUDA facility forecasts using existing independent audits.

Checks both final reservoir fields and well trajectories at common report dates.
Elapsed times on a shared host are observations, not an isolated speedup test.
"""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_journal import atomic_json
from compare_fim_facility_forecast import compare as compare_fields
from check_long_facility_trajectory import compare as compare_trajectory


def qualify(cpu, cuda, budgets):
    cpu, cuda = Path(cpu), Path(cuda)
    summaries = [json.loads((p / "summary.json").read_text()) for p in (cpu, cuda)]
    for directory, summary in zip((cpu, cuda), summaries, strict=True):
        if (directory / "failure.json").exists():
            raise ValueError("A failed forecast cannot be accepted")
        if json.loads((directory / "progress.json").read_text()).get("status") != "complete":
            raise ValueError("Forecast completion must be published")
        actual = hashlib.sha256((directory / "trajectory.json.gz").read_bytes()).hexdigest()
        if actual != summary["trajectory_sha256"]:
            raise ValueError("Trajectory checksum mismatch")
    if any(s.get("status") != "complete" or not s.get("sources_unchanged") for s in summaries):
        raise ValueError("Both forecasts must be complete and source-guarded")
    if summaries[0]["device"] != "cpu" or summaries[1]["device"] != "cuda":
        raise ValueError("Expected a CPU run and an actual CUDA run")
    if summaries[0]["source_sha256"] != summaries[1]["source_sha256"]:
        raise ValueError("CPU/CUDA forecasts require identical sources, config and PVT")
    inputs = [p / "trajectory.json.gz" for p in (cpu, cuda)]
    fields = compare_fields(*inputs, budgets)
    trajectory = compare_trajectory(*inputs)
    return {
        "scope": __doc__,
        "accepted": fields["accepted"] and trajectory["accepted"],
        "cpu": str(cpu), "cuda": str(cuda),
        "sources_config_and_pvt_identical": True,
        "final_fields": fields,
        "report_endpoint_trajectory": trajectory,
        "devices": [s.get("execution") for s in summaries],
        "observed_solve_wall_s": [s["solve_wall_s"] for s in summaries],
        "timing_scope": "Shared host; concurrent CPU/CUDA qualification is not a clean hardware speed comparison",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("cuda", type=Path)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.cpu, args.cuda, json.loads(args.budgets.read_text()))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing qualification: {args.output}")
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
