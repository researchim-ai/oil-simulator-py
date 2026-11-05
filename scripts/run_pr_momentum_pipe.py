"""Source-bound single-phase PR Euler momentum/kinetic pipe schedule."""

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
    from surface.pr_momentum_pipe import PRDirichletBoundary, PRMomentumPipe, PRMomentumState
    from surface.torch_backend import runtime_fingerprint

    config = json.loads(args.config.read_text())
    if (
        set(config) != {"model", "initial", "schedule"}
        or not isinstance(config["schedule"], list)
        or not config["schedule"]
    ):
        raise ValueError("Momentum config requires model, initial and nonempty schedule")
    model = PRMomentumPipe.from_dict(config["model"], device=args.device)
    initial_values = dict(config["initial"])
    hydrostatic = initial_values.pop("hydrostatic", False)
    if type(hydrostatic) is not bool:
        raise ValueError("hydrostatic must be boolean")
    if hydrostatic and not hasattr(model, "initialize_equilibrium"):
        raise ValueError("Hydrostatic initialization requires an inclined pipe model")
    initializer = model.initialize_equilibrium if hydrostatic else model.initialize
    state = initializer(**initial_values)
    initial = state.to_dict()
    intervals = []

    def boundary(spec):
        if spec is None or spec == "wall":
            return spec
        if not isinstance(spec, dict) or set(spec) != {
            "pressure_pa",
            "temperature_k",
            "mole_fractions",
        }:
            raise ValueError("Dirichlet boundary needs exactly PT/composition")
        return PRDirichletBoundary(**spec, fluid_id=model.fluid.model_id)

    for command in config["schedule"]:
        if set(command) - {"duration_s", "left", "right"} or "duration_s" not in command:
            raise ValueError("Unknown/missing momentum schedule fields")
        report = model.advance(
            state,
            command["duration_s"],
            left=boundary(command.get("left")),
            right=boundary(command.get("right")),
        )
        state = PRMomentumState.from_dict(report["state"])
        intervals.append(report)
    result = {
        "config": config,
        "model": model.to_dict(),
        "initial": initial,
        "intervals": intervals,
        "runtime": runtime_fingerprint(),
        "source_sha256": hashes,
    }
    for p in sources:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]:
            raise RuntimeError("Momentum sources changed during run")
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
