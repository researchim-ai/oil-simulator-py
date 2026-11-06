"""Run an actual common ordinary/profile plant Newton with source provenance."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(config, *, device="cpu"):
    from surface.pr_layered_plant import LayeredPlantState, PRLayeredPlantNetwork
    from surface.torch_backend import runtime_fingerprint

    if set(config) != {"model", "initial", "schedule"} or not config["schedule"]:
        raise ValueError("Expected model, initial and nonempty schedule")
    model = PRLayeredPlantNetwork.from_dict(config["model"], device=device)
    initial = config["initial"]
    if set(initial) == {"state"}:
        state = LayeredPlantState.from_dict(initial["state"])
        model.inspect(state)
    elif set(initial) == {"transport", "layered_equilibrium"}:
        state = model.initialize(
            model.transport.initialize(initial["transport"]),
            model.layered.initialize_equilibrium(**initial["layered_equilibrium"]),
        )
    else:
        raise ValueError("Expected state or explicit transport/layered equilibrium")
    first = state
    intervals = []
    for command in config["schedule"]:
        report = model.advance(state, **command)
        state = LayeredPlantState.from_dict(report["state"])
        intervals.append(report)
    heat = sum(r["total_heat_j"] for r in intervals)
    work = sum(r["total_work_j"] for r in intervals)
    return {
        "config": config,
        "model": model.to_dict(),
        "initial": first.to_dict(),
        "state": state.to_dict(),
        "intervals": intervals,
        "total_heat_j": heat,
        "total_work_j": work,
        "conservation_residuals": model._ledger(first, state, heat + work),
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
    if any(hashlib.sha256((root / n).read_bytes()).hexdigest() != h for n, h in hashes.items()):
        raise RuntimeError("Sources changed during joint plant calculation")
    result["source_sha256"] = hashes
    result["sources_unchanged_during_run"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as f:
        temporary = f.name
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    try:
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(args.output)


if __name__ == "__main__":
    main()
