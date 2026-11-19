"""Run the existing operational forecast and an independently audited time refinement.

No new physical solver: combines the production runner and existing engineering
accuracy gate. All profiles use quasi-steady facilities, not dynamic PR vessels.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from benchmark_journal import atomic_json
from check_facility_engineering_profile import check, validate_budgets

ROOT = Path(__file__).resolve().parents[1]


def refined_config(config):
    c = copy.deepcopy(config)
    s = c["simulation"]
    if "coupled_facility" not in s or s["coupled_facility"].get("temporal_control") is not None:
        raise ValueError("This protocol requires the coupled operational maximum-step profile")
    step = s.get("max_time_step_days")
    if type(step) not in (int, float) or not math.isfinite(step) or step <= 0:
        raise ValueError("Explicit positive maximum time step required")
    initial = s.get("initial_time_step_days", step)
    if type(initial) not in (int, float) or not math.isfinite(initial) or initial <= 0:
        raise ValueError("Positive initial time step required")
    s["max_time_step_days"] = step / 2
    s["initial_time_step_days"] = min(initial, step / 2)
    return c


def workload(directory):
    import gzip

    with gzip.open(directory / "trajectory.json.gz", "rt") as stream:
        result = json.load(stream)["result"]
    stages = result["accepted_substeps"]
    return {
        "solve_wall_s": result["solve_wall_s"],
        "accepted_substeps": len(stages),
        "reservoir_evaluations": sum(
            s["facility_coupling"]["reservoir_evaluations"] for s in stages
        ),
        "linear_attempts": sum(
            s["facility_coupling"]["capacity_amg_work"]["linear_attempts"] for s in stages
        ),
        "execution": result["execution"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/fim_blackoil_facility_depletion_operational.json",
    )
    p.add_argument(
        "--schedule", type=Path, default=ROOT / "configs/fim_facility_forecast_schedule.json"
    )
    p.add_argument(
        "--budgets",
        type=Path,
        default=ROOT / "configs/fim_facility_forecast_budgets.json",
    )
    p.add_argument("--days", type=float, default=30.0)
    p.add_argument("--report-days", type=float, default=5.0)
    p.add_argument(
        "--max-step-days",
        type=float,
        help="Explicit candidate maximum step; reference uses half; physical and nonlinear settings unchanged",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.days, args.report_days)):
        p.error("Positive finite forecast and report intervals required")
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        p.error("Select CUDA_VISIBLE_DEVICES explicitly")
    original = json.loads(args.config.read_text())
    if args.max_step_days is not None:
        if not math.isfinite(args.max_step_days) or args.max_step_days <= 0:
            p.error("--max-step-days must be positive and finite")
        s = original["simulation"]
        initial = s.get("initial_time_step_days", s.get("max_time_step_days"))
        if type(initial) not in (int, float) or not math.isfinite(initial) or initial <= 0:
            p.error("Positive initial timestep required")
        s["max_time_step_days"] = args.max_step_days
        s["initial_time_step_days"] = min(initial, args.max_step_days)
    reference = refined_config(original)
    # Preserve the runner's repository-relative PVT convention.
    for c in (original, reference):
        c["fluid"]["pvt_path"] = str((ROOT / c["fluid"]["pvt_path"]).resolve())
    budgets = json.loads(args.budgets.read_text())
    validate_budgets(budgets)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, c in [("operating", original), ("refined", reference)]:
        atomic_json(args.output / (name + ".json"), c)
    atomic_json(args.output / "budgets.json", budgets)
    schedule = args.output / "schedule.json"
    atomic_json(schedule, json.loads(args.schedule.read_text()))
    atomic_json(args.output / "status.json", {"status": "running", "scope": __doc__})
    try:
        for name in ("operating", "refined"):
            command = [
                sys.executable,
                str(ROOT / "scripts/run_fim_facility_forecast.py"),
                "--config",
                str((args.output / (name + ".json")).resolve()),
                "--schedule",
                str(schedule.resolve()),
                "--days",
                str(args.days),
                "--report-days",
                str(args.report_days),
                "--device",
                args.device,
                "--output",
                str((args.output / name).resolve()),
            ]
            # Keep full runner output; expose only progress and actionable failure.
            with (args.output / (name + ".log")).open("w") as log:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith("forecast "):
                            print(name + ": " + line.strip(), flush=True)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            if code:
                raise RuntimeError(f"{name} failed with exit {code}; see {name}.log")
        comparison = check(
            args.output / "operating", args.output / "refined", budgets, days=args.days
        )
        atomic_json(args.output / "comparison.json", comparison)
        summary = {
            "status": "complete" if comparison["accepted"] else "accuracy_failed",
            "accepted": comparison["accepted"],
            "days": args.days,
            "scope": {
                "reservoir": "blackoil FIM",
                "gathering": "configured production network",
                "facility": "quasi-steady",
                "dynamic_vessel_inventory": False,
                "qualification": "Numerical refinement of this configured model; no experimental or detailed PR qualification",
            },
            "work": {name: workload(args.output / name) for name in ("operating", "refined")},
            "comparison": "comparison.json",
        }
        atomic_json(args.output / "status.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if comparison["accepted"] else 1
    except BaseException as error:
        atomic_json(args.output / "status.json", {"status": "failed", "reason": repr(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
