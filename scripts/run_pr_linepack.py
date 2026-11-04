"""Run finite PR linepack with monolithic Darcy faces and terminal vessels."""

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
    from surface.liquid import _number
    from surface.pr_linepack import PRLinepackNetwork
    from surface.stratified_pr import StratifiedPRState
    from surface.stratified_pr_network import StratifiedNetworkState
    from surface.torch_backend import runtime_fingerprint

    cfg = json.loads(args.config.read_text())
    if set(cfg) != {"model", "initial", "schedule"} or set(cfg["initial"]) != {
        "terminals",
        "pipes",
    }:
        raise ValueError("Unknown/missing linepack config fields")
    model = PRLinepackNetwork.from_dict(cfg["model"], device=args.device)
    if not isinstance(cfg["schedule"], list) or not cfg["schedule"]:
        raise ValueError("Nonempty linepack schedule required")
    for command in cfg["schedule"]:
        if set(command) - {"duration_s", "openings"} or "duration_s" not in command:
            raise ValueError("Invalid linepack schedule command")
        _number(command["duration_s"], "duration", positive=True)
        model.network._commands(command.get("openings"))
    if set(cfg["initial"]["terminals"]) != set(model.terminals):
        raise ValueError("Initial terminal registry differs")
    states = {}
    profiles = {}
    for name, v in model.terminals.items():
        initial = dict(cfg["initial"]["terminals"][name])
        wall = initial.pop("wall_temperature_k", initial.get("temperature_k"))
        profiles[name] = v.column.initialize(**initial)
        states[name] = v.initialize(StratifiedPRState.from_dict(profiles[name]["state"]), wall)
    state = model.initialize(states, cfg["initial"]["pipes"])
    initial = state.to_dict()
    intervals = []
    for command in cfg["schedule"]:
        report = model.advance(state, command["duration_s"], openings=command.get("openings"))
        state = StratifiedNetworkState.from_dict(report["state"])
        intervals.append(report)
    result = {
        "config": cfg,
        "model": model.to_dict(),
        "compiled_network": model.network.to_dict(),
        "initial": initial,
        "initial_terminal_profiles": profiles,
        "intervals": intervals,
        "runtime": runtime_fingerprint(),
        "source_sha256": hashes,
    }
    for p in sources:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]:
            raise RuntimeError("Sources changed during linepack calculation")
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
