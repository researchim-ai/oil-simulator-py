"""Source-bound finite-inertia PR interphase relaxation experiments."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(config, *, device="cpu"):
    from surface.pr_interphase_relaxation import PRInterphaseRelaxation, PRRelaxationState
    from surface.torch_backend import runtime_fingerprint

    if set(config) != {"model", "initial", "schedule"} or not config["schedule"]:
        raise ValueError("Expected relaxation model, initial and schedule")
    model = PRInterphaseRelaxation.from_dict(config["model"], device=device)
    initial = config["initial"]
    if set(initial) == {"state"}:
        state = PRRelaxationState.from_dict(initial["state"])
        model.inspect(state)
    elif set(initial) == {"phases"}:
        state = model.initialize(initial["phases"])
    else:
        raise ValueError("Expected phases or full restart checkpoint")
    first = state
    intervals = []
    for command in config["schedule"]:
        result = model.advance(state, **command)
        state = PRRelaxationState.from_dict(result["state"])
        intervals.append(result)
    impulse = sum(r["wall_impulse_kg_m_s"] for interval in intervals for r in interval["steps"])
    return {
        "config": config,
        "model": model.to_dict(),
        "initial": first.to_dict(),
        "state": state.to_dict(),
        "intervals": intervals,
        "conservation_residuals": model.ledger(first, state, impulse),
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
