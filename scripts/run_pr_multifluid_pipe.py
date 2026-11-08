"""Source-bound spatial PR multifluid pipe with composed local exchange."""

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
    from surface.pr_multifluid_pipe import PRMultifluidPipe, PRMultifluidPipeState
    from surface.torch_backend import runtime_fingerprint

    if set(config) != {"model", "initial", "schedule"} or not config["schedule"]:
        raise ValueError("Expected relaxation model, initial and schedule")
    from surface.pr_multifluid_network import PRMultifluidNetwork

    factory = (
        PRMultifluidNetwork
        if config["model"].get("model")
        in (
            "pr_multifluid_network_v1",
            "pr_multifluid_network_v2",
            "pr_multifluid_network_v3",
            "pr_multifluid_network_v4",
        )
        else PRMultifluidPipe
    )
    state_factory = PRMultifluidPipeState
    if config["model"].get("model") == "pr_equilibrium_euler_pipe_v1":
        from surface.pr_equilibrium_euler_pipe import (
            PREquilibriumEulerPipe,
            PREquilibriumEulerPipeState,
        )

        factory = PREquilibriumEulerPipe
        state_factory = PREquilibriumEulerPipeState
    plant = config["model"].get("model") == "pr_equilibrium_euler_plant_v1"
    if plant:
        from surface.pr_equilibrium_euler_plant import (
            PREquilibriumEulerPlant,
            PREquilibriumEulerPlantState,
        )

        factory, state_factory = PREquilibriumEulerPlant, PREquilibriumEulerPlantState
    model = factory.from_dict(config["model"], device=device)
    initial = config["initial"]
    if set(initial) == {"state"}:
        state = state_factory.from_dict(initial["state"])
        model.inspect(state)
    elif set(initial) == {"cells"}:
        state = model.initialize(initial["cells"])
    else:
        raise ValueError("Expected phase cells or full pipe checkpoint")
    first = state
    intervals = []
    for command in config["schedule"]:
        result = model.advance(state, **command)
        state = state_factory.from_dict(result["state"])
        intervals.append(result)
    if plant:
        energy = math.fsum(r["external_energy_j"] for i in intervals for r in i["steps"])
        impulse = [
            math.fsum(r["external_impulse_kg_m_s"][j] for i in intervals for r in i["steps"])
            for j in range(3)
        ]
        return {
            "config": config,
            "model": model.to_dict(),
            "initial": first.to_dict(),
            "state": state.to_dict(),
            "intervals": intervals,
            "conservation_residuals": model.ledger(first, state, energy, impulse),
            "runtime": runtime_fingerprint(),
        }
    impulse = model._sum_impulses(
        r["wall_impulse_kg_m_s"] for interval in intervals for r in interval["steps"]
    )
    return {
        "config": config,
        "model": model.to_dict(),
        "initial": first.to_dict(),
        "state": state.to_dict(),
        "intervals": intervals,
        "conservation_residuals": model.ledger(
            first,
            state,
            impulse,
            math.fsum(
                r.get("external_work_j", 0.0) + r.get("external_heat_j", 0.0)
                for interval in intervals
                for r in interval["steps"]
            ),
        ),
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
