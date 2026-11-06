"""Source-bound low-drop valves / actuators / PI and gravitational transport."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(config, *, device="cpu"):
    from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState
    from surface.stratified_pr import StratifiedPRState
    from surface.torch_backend import runtime_fingerprint

    if (
        set(config) != {"model", "initial", "schedule"}
        or not isinstance(config["schedule"], list)
        or not config["schedule"]
    ):
        raise ValueError("Expected model, initial and nonempty schedule")
    model = PRControlledSurface.from_dict(config["model"], device=device)
    initial = config["initial"]
    if set(initial) == {"state"}:
        state = PRControlledSurfaceState.from_dict(initial["state"])
        model.inspect(state)
    else:
        if set(initial) != {"plant", "positions", "integrals"}:
            raise ValueError("Expected plant, positions and integrals")
        spec = initial["plant"]
        if hasattr(model.plant, "bridges"):
            if set(spec) != {"transport", "profiles"} or set(spec["profiles"]) != set(
                model.plant.names
            ):
                raise ValueError("Expected transport and exact profile registry")
            physical = model.plant.initialize(
                model.plant.transport.initialize(spec["transport"]),
                {
                    n: b.layered.initialize_equilibrium(**spec["profiles"][n])
                    for n, b in model.plant.bridges.items()
                },
            )
        else:
            if set(spec) != {"transport", "columns", "wall_temperatures"}:
                raise ValueError("Expected transport, columns and wall_temperatures")
            ts = model.plant.transport.initialize(spec["transport"])
            columns = {
                n: StratifiedPRState.from_dict(model.plant.columns[n].initialize(**v)["state"])
                for n, v in spec["columns"].items()
            }
            physical = model.plant.initialize(ts, columns, spec["wall_temperatures"])
        state = model.initialize(physical, initial["positions"], initial["integrals"])
    first = state
    intervals = []
    for command in config["schedule"]:
        if (
            set(command)
            - {
                "duration_s",
                "openings",
                "heat_w",
                "utility_temperature_k",
                "loss_coefficients",
                "drive_torque_nm",
                "restriction_openings",
                "targets",
                "actuator_modes",
                "reset_trips",
                "profile_heat_w",
                "profile_utility_temperature_k",
                "profile_openings",
            }
            or "duration_s" not in command
        ):
            raise ValueError("Unknown/missing equilibrium transport command")
        report = model.advance(state, **command)
        state = PRControlledSurfaceState.from_dict(report["state"])
        intervals.append(report)
    heat = sum(r["total_heat_j"] for r in intervals)
    work = sum(r["total_work_j"] for r in intervals)
    return {
        "config": config,
        "model": model.to_dict(),
        "initial": first.to_dict(),
        "intervals": intervals,
        "conservation_residuals": model.plant._ledger(first.plant, state.plant, heat + work),
        "total_heat_j": heat,
        "total_work_j": work,
        "runtime": runtime_fingerprint(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((root / "src/surface").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = run(json.loads(args.config.read_text()), device=args.device)
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]
        for p in paths
    ):
        raise RuntimeError("Transport sources changed during run")
    result["source_sha256"] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as handle:
            temp = handle.name
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp, args.output)
        temp = None
    finally:
        if temp:
            os.unlink(temp)
    print(args.output)


if __name__ == "__main__":
    main()
