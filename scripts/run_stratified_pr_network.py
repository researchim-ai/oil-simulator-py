"""Run a source-bound finite-aperture gravitational PR vessel network schedule."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *sorted((root / "src/surface").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    from surface.stratified_pr import StratifiedPRState
    from surface.stratified_pr_flow import StratifiedPRFlow
    from surface.stratified_pr_network import (
        StratifiedConnection,
        StratifiedNetworkState,
        StratifiedPRNetwork,
    )
    from surface.torch_backend import runtime_fingerprint

    config = json.loads(args.config.read_text())
    if set(config) != {"vessels", "connections", "integration", "schedule"}:
        raise ValueError("Unknown or missing network config fields")
    if not isinstance(config["schedule"], list) or not config["schedule"]:
        raise ValueError("Nonempty schedule required")
    vessels = {}
    states = {}
    profiles = {}
    for name, description in sorted(config["vessels"].items()):
        if set(description) != {"model", "initial", "wall_temperature_k"}:
            raise ValueError("Each vessel needs model, initial and wall temperature")
        v = StratifiedPRFlow.from_dict(description["model"], device=args.device)
        profiles[name] = v.column.initialize(**description["initial"])
        states[name] = v.initialize(
            StratifiedPRState.from_dict(profiles[name]["state"]), description["wall_temperature_k"]
        )
        vessels[name] = v
    model = StratifiedPRNetwork(
        vessels, [StratifiedConnection(**p) for p in config["connections"]], **config["integration"]
    )
    state = model.initialize(states)
    initial = state.to_dict()
    intervals = []
    for command in config["schedule"]:
        if (
            not isinstance(command, dict)
            or set(command) - {"duration_s", "openings"}
            or "duration_s" not in command
        ):
            raise ValueError("Schedule command requires duration_s and optional openings")
        report = model.advance(state, command["duration_s"], openings=command.get("openings"))
        state = StratifiedNetworkState.from_dict(report["state"])
        intervals.append(report)
    result = {
        "config": config,
        "model": model.to_dict(),
        "initial": initial,
        "initial_profiles": profiles,
        "intervals": intervals,
        "runtime": runtime_fingerprint(),
        "source_sha256": hashes,
    }
    for p in sources:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]:
            raise RuntimeError("Source changed during network run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, args.output)
        temporary = None
    finally:
        if temporary:
            os.unlink(temporary)
    print(args.output)


if __name__ == "__main__":
    main()
