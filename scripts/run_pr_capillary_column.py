"""Reproduce finite gravitational-column / capillary-pipe IMEX trajectories."""

import argparse
import json
from pathlib import Path

from check_pr_capillary_column import audit
from run_pr_active_surface import fingerprints

from surface.pr_capillary_column import PRCapillaryColumnNetwork, PRCapillaryColumnState


def run(config, *, device="cpu"):
    if set(config) - {"duration_s", "schedule"} != {"model", "initial", "normal_relaxation"} or (
        "duration_s" in config
    ) == ("schedule" in config):
        raise ValueError(
            "Explicit model, stocks, split switch and either duration or schedule required"
        )
    hashes = fingerprints()
    model = PRCapillaryColumnNetwork.from_dict(config["model"], device=device)
    old = PRCapillaryColumnState.from_dict(config["initial"])
    schedule = config.get("schedule", [{"duration_s": config.get("duration_s")}])
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("Nonempty coupled operating schedule required")
    checks, intervals = [], []
    previous = old
    for period in schedule:
        if "duration_s" not in period or set(period) - {
            "duration_s",
            "column_heat_w",
            "heat_temperature_k",
        }:
            raise ValueError("Each interval requires duration and optional physical heater command")
        command = {
            "column_heat_w": period.get("column_heat_w", 0.0),
            "heat_temperature_k": period.get("heat_temperature_k"),
        }
        final, result = model.advance(
            previous, period["duration_s"], normal_relaxation=config["normal_relaxation"], **command
        )
        for step in result["steps"]:
            current = PRCapillaryColumnState.from_dict(step["state"])
            checks.append(
                audit(
                    model.to_dict(),
                    previous,
                    current,
                    step,
                    normal_relaxation=config["normal_relaxation"],
                    **command,
                )
            )
            previous = current
        intervals.append({"command": period, "result": result})
    if previous != final or hashes != fingerprints():
        raise RuntimeError("Coupled history or physical source changed during execution")
    return {
        "format": "pr_capillary_column_run_v1",
        "request": config,
        "device": device,
        "source_sha256": hashes,
        "result": {
            "state": final.to_dict(),
            "steps": [s for i in intervals for s in i["result"]["steps"]],
            "rejected": [s for i in intervals for s in i["result"]["rejected"]],
        },
        "intervals": intervals,
        "final": final.to_dict(),
        "independent_audits": checks,
        "qualification": "Conservative first-order partitioned coupling with finite column inventories, active aperture phases and controlled midpoint profiles; positive pipe phase support; not a monolithic arbitrary-network or active-set pipe integrator",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a completed coupled run")
    result = run(json.loads(args.config.read_text()), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "accepted_substeps": len(result["result"]["steps"]),
                "rejected_substeps": len(result["result"]["rejected"]),
                "time_s": result["final"]["pipe"]["time_s"],
            }
        )
    )


if __name__ == "__main__":
    main()
