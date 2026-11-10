"""Reproduce source-bound shared-column capillary facility histories."""

import argparse
import json
import time
from pathlib import Path

import torch
from check_pr_capillary_facility import audit
from run_pr_active_surface import fingerprints

from surface.operation import _digest
from surface.pr_capillary_facility import PRCapillaryFacility, PRCapillaryFacilityState


def run(config, *, device="cpu"):
    if set(config) != {"system", "initial", "normal_relaxation", "schedule", "maximum_attempts"}:
        raise ValueError("Explicit facility, stocks, split switch and schedule required")
    if not isinstance(config["schedule"], list) or not config["schedule"]:
        raise ValueError("Nonempty facility operating schedule required")
    hashes = fingerprints()
    model = PRCapillaryFacility.from_dict(config["system"], device=device)
    initial = previous = PRCapillaryFacilityState.from_dict(config["initial"])
    model.inspect(initial)
    cuda = device == "cuda"
    intervals, checks = [], []
    for period in config["schedule"]:
        if set(period) - {"duration_s", "heaters", "phase_controls"} or not {
            "duration_s",
            "heaters",
        } <= set(period):
            raise ValueError("Each facility interval requires duration and heater commands")
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        final, result = model.advance(
            previous,
            period["duration_s"],
            maximum_attempts=config["maximum_attempts"],
            normal_relaxation=config["normal_relaxation"],
            heaters=period["heaters"],
            phase_controls=period.get("phase_controls"),
        )
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        for step in result["steps"]:
            current = PRCapillaryFacilityState.from_dict(step["state"])
            checks.append(
                audit(
                    model.to_dict(),
                    previous,
                    current,
                    step,
                    normal_relaxation=config["normal_relaxation"],
                    heaters=period["heaters"],
                    phase_controls=period.get("phase_controls"),
                )
            )
            previous = current
        if previous != final:
            raise RuntimeError("Accepted facility history differs from final checkpoint")
        intervals.append({"command": period, "solver_time_s": elapsed, "result": result})
    if hashes != fingerprints() or _digest(initial.to_dict()) != _digest(config["initial"]):
        raise RuntimeError("Facility source or initial stock changed during run")
    return {
        "format": "pr_capillary_facility_run_v1",
        "request": config,
        "device": device,
        "source_sha256": hashes,
        "intervals": intervals,
        "final": previous.to_dict(),
        "independent_audits": checks,
        "qualification": "Fixed graph of finite shared PR gravity columns and horizontal capillary lines; declared initial or converged end-profile coupling of first-order split pipe operators with declared positive/active axial support and optional global reduced Schur Newton; not an unreduced all-cell matrix or empirical validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a completed facility trajectory")
    result = run(json.loads(args.config.read_text()), device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "time_s": result["final"]["pipes"][next(iter(result["final"]["pipes"]))]["time_s"],
                "independent_audits": len(result["independent_audits"]),
            }
        )
    )


if __name__ == "__main__":
    main()
