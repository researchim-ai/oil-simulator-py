"""Source-bound multi-phase PR transport graph schedule (HEM or explicit slip)."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(config, *, device="cpu", integrator="backward_euler", native_kernels=False):
    if native_kernels:
        if device != "cpu":
            raise ValueError("Native CPU kernels require device=cpu")
        from surface.native_execution import native_kernel_execution

        with native_kernel_execution() as engine:
            result = _run(config, device=device, integrator=integrator, compiled_properties=True)
            result["native_execution"] = engine.statistics()
            return result
    return _run(config, device=device, integrator=integrator)


def _run(config, *, device, integrator, compiled_properties=False):
    if integrator not in ("backward_euler", "gauss4", "radau3", "radau3-embedded"):
        raise ValueError("Unknown PR integrator")
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
    if compiled_properties:
        model.fluid._compiled_chemical_cpu = True
        model.fluid._compiled_full_eos_cpu = True
    if set(config["initial"]) == {"state"}:
        state = PREquilibriumTransportState.from_dict(config["initial"]["state"])
        model.inspect(state)
    elif set(config["initial"]) == {"nodes"}:
        state = model.initialize(config["initial"]["nodes"])
    elif set(config["initial"]) == {"groups"}:
        state = initialize_grouped(model, config["initial"]["groups"])
    else:
        raise ValueError("Initial input requires exactly nodes, groups or state")
    solver = model
    if integrator in ("gauss4", "radau3", "radau3-embedded"):
        from surface import PRCollocation

        solver = PRCollocation(
            model,
            scheme=integrator.split("-")[0],
            error_control="embedded" if integrator == "radau3-embedded" else "step_doubling",
        )
    first = state
    intervals = []
    for command in config["schedule"]:
        if (
            set(command)
            - {
                "duration_s",
                "openings",
                "heat_w",
                "utility_temperature_k",
                "loss_coefficients",
                "drive_torque_nm",
                "restriction_openings",
            }
            or "duration_s" not in command
        ):
            raise ValueError("Unknown/missing equilibrium transport command")
        if integrator in ("gauss4", "radau3", "radau3-embedded") and set(command) & {
            "drive_torque_nm",
            "restriction_openings",
        }:
            raise ValueError("Machine commands require the original PR integrator")
        report = solver.advance(state, **command)
        state = PREquilibriumTransportState.from_dict(report["state"])
        intervals.append(report)
    heat = sum(r["total_heat_j"] for r in intervals)
    work = sum(r["total_work_j"] for r in intervals)
    return {
        "integrator": integrator,
        "config": config,
        "model": model.to_dict(),
        "initial": first.to_dict(),
        "intervals": intervals,
        "conservation_residuals": model._ledger(first, state, heat + work),
        "total_heat_j": heat,
        "total_work_j": work,
        "runtime": runtime_fingerprint(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-kernels", action="store_true")
    parser.add_argument(
        "--integrator",
        choices=["backward_euler", "gauss4", "radau3", "radau3-embedded"],
        default="backward_euler",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((root / "src/surface").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = run(
        json.loads(args.config.read_text()),
        device=args.device,
        integrator=args.integrator,
        native_kernels=args.native_kernels,
    )
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
