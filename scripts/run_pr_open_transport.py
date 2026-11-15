"""Run staged open PR transport with explicit method and atomic publication."""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surface.pr_equilibrium_transport import PREquilibriumTransportState
from surface.pr_open_transport import PROpenTransportNetwork
from surface.pr_transport_builder import initialize_grouped
from surface.torch_backend import runtime_fingerprint


def run(config, *, device="cpu", integrator="backward_euler"):
    if (
        set(config) != {"model", "initial", "schedule"}
        or not isinstance(config["schedule"], list)
        or not config["schedule"]
    ):
        raise ValueError("Expected model, initial and nonempty schedule")
    model = PROpenTransportNetwork.from_dict(config["model"], device=device)
    physical = model.transport
    spec = config["initial"]
    if set(spec) == {"nodes"}:
        state = physical.initialize(spec["nodes"])
    elif set(spec) == {"groups"}:
        state = initialize_grouped(physical, spec["groups"])
    elif set(spec) == {"state"}:
        state = PREquilibriumTransportState.from_dict(spec["state"])
        model.inspect(state)
    else:
        raise ValueError("Expected exactly nodes, groups or state initialization")
    first = state
    reports = []
    for command in config["schedule"]:
        if "duration_s" not in command or set(command) - {
            "duration_s",
            "port_openings",
            "openings",
            "heat_w",
            "utility_temperature_k",
            "loss_coefficients",
            "drive_torque_nm",
            "restriction_openings",
        }:
            raise ValueError("Unknown/missing open transport command")
        report = model.advance(state, **command, integrator=integrator)
        reports.append(report)
        state = PREquilibriumTransportState.from_dict(report["state"])
    heat = math.fsum(r["total_heat_j"] for r in reports)
    work = math.fsum(r["total_work_j"] for r in reports)
    incoming = [
        math.fsum(r["boundary_component_moles"][j] for r in reports)
        for j in range(len(model.fluid.names))
    ]
    energy = math.fsum(r["boundary_energy_j"] for r in reports)
    return {
        "integrator": integrator,
        "config": config,
        "model": model.to_dict(),
        "initial": first.to_dict(),
        "intervals": reports,
        "total_heat_j": heat,
        "total_work_j": work,
        "boundary_component_moles": incoming,
        "boundary_energy_j": energy,
        "conservation_residuals": physical._ledger(
            first, state, heat + work + energy, incoming_n=incoming
        ),
        "runtime": runtime_fingerprint(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--integrator", choices=["backward_euler", "gauss4", "radau3"], default="backward_euler"
    )
    args = parser.parse_args()
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = run(
        json.loads(args.config.read_text()), device=args.device, integrator=args.integrator
    )
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]
        for p in paths
    ):
        raise RuntimeError("Sources changed during open transport run")
    result["source_sha256"] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(args.output)


if __name__ == "__main__":
    main()
