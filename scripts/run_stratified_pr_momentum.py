"""Source-bound horizontal momentum pipe coupled to gravitational PR profiles."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(config, *, device="cpu"):
    from surface.stratified_pr import StratifiedPRState
    from surface.stratified_pr_momentum import StratifiedMomentumState, StratifiedPRMomentumNetwork
    from surface.torch_backend import runtime_fingerprint

    if (
        set(config) != {"model", "initial", "schedule"}
        or not isinstance(config["schedule"], list)
        or not config["schedule"]
    ):
        raise ValueError("Expected model, initial and nonempty schedule")
    model = StratifiedPRMomentumNetwork.from_dict(config["model"], device=device)
    initial = config["initial"]
    if set(initial) == {"equilibrium"}:
        state = model.initialize_equilibrium(**initial["equilibrium"])
    else:
        if set(initial) != {"pipe", "columns"} or len(initial["columns"]) != 2:
            raise ValueError("Expected pipe and two initial columns")
        vessels = []
        for i, (c, values) in enumerate(zip(model.columns, initial["columns"])):
            values = dict(values)
            wall = values.pop("wall_temperature_k")
            anchor = values.pop("anchor_height_m", None)
            if anchor is None:
                column = StratifiedPRState.from_dict(c.initialize(**values)["state"])
            else:
                from surface.pr_hydrostatic_initialization import initialize_column_at_height

                column, _ = initialize_column_at_height(c, **values, height_m=anchor)
            vessels.append(model.initial_vessel_state(i, column, wall))
        values = dict(initial["pipe"])
        wall = values.pop("wall_temperature_k")
        hydrostatic = values.pop("hydrostatic", False)
        if type(hydrostatic) is not bool:
            raise ValueError("hydrostatic must be boolean")
        if hydrostatic and not hasattr(model.pipe.pipe, "initialize_equilibrium"):
            raise ValueError("Hydrostatic pipe initialization requires an inclined model")
        initializer = (
            model.pipe.pipe.initialize_equilibrium if hydrostatic else model.pipe.pipe.initialize
        )
        pipe = model.pipe.initialize(initializer(**values), wall)
        state = model.initialize(pipe, vessels)
    first = state.to_dict()
    intervals = []
    for command in config["schedule"]:
        if set(command) - {"duration_s", "openings"} or "duration_s" not in command:
            raise ValueError("Unknown/missing stratified momentum command")
        report = model.advance(state, **command)
        state = StratifiedMomentumState.from_dict(report["state"])
        intervals.append(report)
    return {
        "config": config,
        "model": model.to_dict(),
        "initial": first,
        "intervals": intervals,
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
    report = run(json.loads(args.config.read_text()), device=args.device)
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]
        for p in paths
    ):
        raise RuntimeError("Stratified momentum sources changed during run")
    report["source_sha256"] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, args.output)
        temporary = None
    finally:
        if temporary:
            os.unlink(temporary)
    print(args.output)


if __name__ == "__main__":
    main()
