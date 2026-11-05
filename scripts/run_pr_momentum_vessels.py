"""Run a source-bound finite vessel / thermal momentum pipe schedule."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(config, *, device="cpu"):
    from surface.pr_momentum_vessels import PRMomentumVesselNetwork, PRMomentumVesselNetworkState
    from surface.torch_backend import runtime_fingerprint

    if (
        set(config) != {"model", "initial", "schedule"}
        or not isinstance(config["schedule"], list)
        or not config["schedule"]
    ):
        raise ValueError("Expected model, initial and nonempty schedule")
    model = PRMomentumVesselNetwork.from_dict(config["model"], device=device)
    initial = dict(config["initial"])
    pipe = dict(initial.pop("pipe"))
    wall = pipe.pop("wall_temperature_k")
    pipe_state = model.pipe.initialize(model.pipe.pipe.initialize(**pipe), wall)
    state = model.initialize(pipe_state, **initial)
    first = state.to_dict()
    intervals = []
    for command in config["schedule"]:
        if set(command) - {"duration_s", "openings"} or "duration_s" not in command:
            raise ValueError("Unknown/missing network schedule fields")
        report = model.advance(state, **command)
        state = PRMomentumVesselNetworkState.from_dict(report["state"])
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
    result = run(json.loads(args.config.read_text()), device=args.device)
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]
        for p in paths
    ):
        raise RuntimeError("Network sources changed during run")
    result["source_sha256"] = hashes
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
