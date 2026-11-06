"""Source-bound native inclined profile-cell Euler runner (family: layered)."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(config, *, device="cpu"):
    from surface.pr_layered_transport import LayeredTransportState, PRLayeredTransportNetwork
    from surface.stratified_pr import StratifiedPRState
    from surface.torch_backend import runtime_fingerprint

    if set(config) != {"model", "initial", "schedule"} or not config["schedule"]:
        raise ValueError("Expected model, initial and nonempty schedule")
    model = PRLayeredTransportNetwork.from_dict(config["model"], device=device)
    initial = config["initial"]
    if set(initial) == {"state"}:
        state = LayeredTransportState.from_dict(initial["state"])
        model.inspect(state)
    elif set(initial) == {"equilibrium"}:
        state = model.initialize_equilibrium(**initial["equilibrium"])
    elif set(initial) <= {"cells", "velocities_m_s"} and "cells" in initial:
        profiles = [
            StratifiedPRState.from_dict(c.initialize(**v)["state"])
            for c, v in zip(model.columns, initial["cells"], strict=True)
        ]
        state = model.initialize(profiles, initial.get("velocities_m_s"))
    else:
        raise ValueError("Expected state, equilibrium, or cells with optional velocities_m_s")
    first = state
    intervals = []
    for command in config["schedule"]:
        state, reports = model.advance(state, **command)
        steps = [half for r in reports for half in r["halves"]]
        intervals.append({"state": state.to_dict(), "steps": steps, "adaptive": reports})
    return {
        "config": config,
        "model": model.to_dict(),
        "initial": first.to_dict(),
        "state": state.to_dict(),
        "intervals": intervals,
        "runtime": runtime_fingerprint(),
        "source_frozen": False,
        "scope": "native constant-D coaxial profile chain; graph bridge requires parent acceptance",
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
        hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise RuntimeError("Sources changed during layered run")
    result["source_sha256"] = hashes
    result["sources_unchanged_during_run"] = True
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
