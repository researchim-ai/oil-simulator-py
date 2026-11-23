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
    parser.add_argument("--integrator", choices=("be", "bdf2"), default="be")
    parser.add_argument("--native-kernels", action="store_true")
    args = parser.parse_args()
    if args.native_kernels:
        if args.device != "cpu":
            parser.error("Native CPU kernels require --device cpu")
        from surface.native_execution import native_kernel_execution

        with native_kernel_execution() as engine:
            return execute(args, engine)
    return execute(args)


def execute(args, engine=None):
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *sorted((root / "src/surface").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    from surface.pr_momentum_bdf import PRMomentumBDF, PRMomentumBDFState
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
    if engine is not None:
        model.fluid._compiled_chemical_cpu = True
        model.fluid._compiled_full_eos_cpu = True
    initial_values = dict(config["initial"])
    hydrostatic = initial_values.pop("hydrostatic", False)
    if type(hydrostatic) is not bool:
        raise ValueError("hydrostatic must be boolean")
    if hydrostatic and not hasattr(model, "initialize_equilibrium"):
        raise ValueError("Hydrostatic initialization requires an inclined pipe model")
    initializer = model.initialize_equilibrium if hydrostatic else model.initialize
    transport = PRMomentumBDF(model) if args.integrator == "bdf2" else model
    if args.integrator == "bdf2":
        if hydrostatic:
            raise ValueError("BDF2 does not yet support inclined hydrostatic reconstruction")
        state = (
            PRMomentumBDFState.from_dict(initial_values["state"])
            if set(initial_values) == {"state"}
            else transport.initialize(**initial_values)
        )
    else:
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
        report = transport.advance(
            state,
            command["duration_s"],
            left=boundary(command.get("left")),
            right=boundary(command.get("right")),
        )
        state = (PRMomentumBDFState if args.integrator == "bdf2" else PRMomentumState).from_dict(
            report["state"]
        )
        intervals.append(report)
    result = {
        "config": config,
        "model": model.to_dict(),
        "initial": initial,
        "intervals": intervals,
        "runtime": runtime_fingerprint(),
        "source_sha256": hashes,
        "integrator": args.integrator,
        "native_execution": engine.statistics() if engine is not None else None,
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
