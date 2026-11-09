"""Source-bound finite-rate PR thermochemical and active multifluid experiments."""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(config, *, device="cpu"):
    from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
    from surface.pr_interphase_relaxation import PRRelaxationPhase
    from surface.pr_phase_equilibration import PRPhaseEquilibriumState
    from surface.pr_thermochemical_relaxation import PRThermochemicalRelaxation
    from surface.torch_backend import runtime_fingerprint

    if (
        set(config) != {"model", "initial", "schedule"}
        or not isinstance(config["schedule"], list)
        or not config["schedule"]
    ):
        raise ValueError("Expected model, initial and nonempty source/pipe schedule")
    active = config["model"].get("model") in (
        "pr_active_multifluid_pipe_v1",
        "pr_active_multifluid_pipe_v2",
        "pr_active_multifluid_pipe_v3",
    )
    factory, state_factory = (
        (PRActiveMultifluidPipe, PRActiveMultifluidState)
        if active
        else (PRThermochemicalRelaxation, PRPhaseEquilibriumState)
    )
    model = factory.from_dict(config["model"], device=device)
    initial = config["initial"]
    if set(initial) == {"state"}:
        state = state_factory.from_dict(initial["state"])
        model.inspect(state)
    elif active and set(initial) == {"cells"}:
        state = model.initialize(initial["cells"])
    elif not active and set(initial) == {"phases", "phase_roots"}:
        state = model.initialize(
            [PRRelaxationPhase.from_dict(p) for p in initial["phases"]], initial["phase_roots"]
        )
    else:
        raise ValueError("Canonical phase stocks or full checkpoint required")
    first = state
    intervals = []
    for command in config["schedule"]:
        result = model.advance(state, **command)
        state = state_factory.from_dict(result["state"])
        intervals.append(result)
    residual = [b - a for a, b in zip(model._totals(first), model._totals(state))]
    if active:
        residual[-2] -= math.fsum(
            r.get("external_heat_j", 0.0) for interval in intervals for r in interval["steps"]
        )
        residual[-1] -= math.fsum(
            r["wall_impulse_kg_m_s"] + r["gravity_impulse_kg_m_s"]
            for interval in intervals
            for r in interval["steps"]
        )
    return {
        "config": config,
        "model": model.to_dict(),
        "initial": first.to_dict(),
        "state": state.to_dict(),
        "intervals": intervals,
        "conservation_residuals": residual,
        "runtime": runtime_fingerprint(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = list({Path(__file__).resolve(), Path(sys.argv[0]).resolve()}) + [
        *sorted((root / "src/surface").glob("*.py"))
    ]
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
