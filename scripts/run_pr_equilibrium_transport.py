"""Source-bound multi-phase PR transport graph schedule (HEM or explicit slip)."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(config, *, device="cpu"):
    from surface.pr_equilibrium_transport import (
        PREquilibriumTransportNetwork,
        PREquilibriumTransportState,
    )
    from surface.pr_transport_builder import initialize_grouped
    from surface.torch_backend import runtime_fingerprint

    if (
        set(config) != {"model", "initial", "schedule"}
        or not isinstance(config["schedule"], list)
        or not config["schedule"]
    ):
        raise ValueError("Expected model, initial and nonempty schedule")
    model = PREquilibriumTransportNetwork.from_dict(config["model"], device=device)
    if set(config["initial"]) == {"nodes"}:
        state = model.initialize(config["initial"]["nodes"])
    elif set(config["initial"]) == {"groups"}:
        state = initialize_grouped(model, config["initial"]["groups"])
    else:
        raise ValueError("Initial input requires exactly nodes or groups")
    first = state
    intervals = []
    for command in config["schedule"]:
        if (
            set(command)
            - {"duration_s", "openings", "heat_w", "utility_temperature_k", "loss_coefficients"}
            or "duration_s" not in command
        ):
            raise ValueError("Unknown/missing equilibrium transport command")
        report = model.advance(state, **command)
        state = PREquilibriumTransportState.from_dict(report["state"])
        intervals.append(report)
    heat = sum(r["total_heat_j"] for r in intervals)
    return {
        "config": config,
        "model": model.to_dict(),
        "initial": first.to_dict(),
        "intervals": intervals,
        "conservation_residuals": model._ledger(first, state, heat),
        "total_heat_j": heat,
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
