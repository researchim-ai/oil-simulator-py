"""Run source-bound conservative active-phase surface configurations."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from surface.operation import _digest
from surface.pr_active_surface import PRActiveSurfaceNetwork, PRActiveSurfaceState

ROOT = Path(__file__).resolve().parents[1]


def fingerprints():
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(config, *, device="cpu"):
    if set(config) - {
        "system",
        "initial",
        "duration_s",
        "schedule",
        "commands",
        "maximum_attempts",
        "minimum_step_s",
    }:
        raise ValueError("Unknown active surface run fields")
    before = fingerprints()
    model = PRActiveSurfaceNetwork.from_dict(config["system"], device=device)
    initial = PRActiveSurfaceState.from_dict(config["initial"])
    model.inspect(initial)
    if ("duration_s" in config) == ("schedule" in config):
        raise ValueError("Use either one duration or a nonempty operating schedule")
    schedule = config.get("schedule", [{"duration_s": config.get("duration_s")}])
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("Nonempty operating schedule required")
    commands = {k: dict(v) for k, v in config.get("commands", {}).items()}
    state = initial
    intervals = []
    for period in schedule:
        if set(period) - {"duration_s", "commands"} or "duration_s" not in period:
            raise ValueError("Each period needs duration and optional commands")
        for name, values in period.get("commands", {}).items():
            commands[name] = {**commands.get(name, {}), **values}
        result = model.advance(
            state,
            period["duration_s"],
            maximum_attempts=config.get("maximum_attempts", 10000),
            minimum_step_s=config.get("minimum_step_s", 1e-12),
            **commands,
        )
        state = PRActiveSurfaceState.from_dict(result["state"])
        intervals.append({"commands": {k: dict(v) for k, v in commands.items()}, **result})
    steps = [r for interval in intervals for r in interval["steps"]]
    result = {
        "state": state.to_dict(),
        "steps": steps,
        "attempts": sum(r["attempts"] for r in intervals),
        "conservation_residuals": model.ledger(
            initial,
            state,
            math.fsum(r["external_energy_j"] for r in steps),
            [
                math.fsum(r["external_component_moles"][j] for r in steps)
                for j in range(len(model.fluid.names))
            ],
        ),
    }
    if before != fingerprints():
        raise RuntimeError("Physical source changed during the run")
    report = {
        "format": "pr_active_surface_run_v1",
        "request": config,
        "system": model.to_dict(),
        "initial": initial.to_dict(),
        "result": result,
        "intervals": intervals,
        "source_hashes": before,
        "device": device,
    }
    report["checksum"] = _digest(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing result")
    result = run(json.loads(args.config.read_text()), device=args.device)
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(result, handle, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "steps": len(result["result"]["steps"]),
                "conservation_residuals": result["result"]["conservation_residuals"],
            }
        )
    )


if __name__ == "__main__":
    main()
