"""Run a monolithic dynamic PR vessel network with checked JSON restart."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    ConnectionActuator,
    DynamicVessel,
    DynamicVesselNetwork,
    HydraulicConvergenceError,
    NetworkTrip,
    PengRobinsonFluid,
    PengRobinsonMixture,
    SeparatorGeometry,
    VesselWall,
    run_network_schedule,
    vessel_connection_from_dict,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    temporary = None
    try:
        d = json.loads(args.config.read_text())
        if not {"fluid", "vessels", "connections", "initial", "periods"} <= set(d) or set(d) - {
            "fluid",
            "vessels",
            "connections",
            "initial",
            "periods",
            "stepper",
            "protections",
            "event_time_tolerance_s",
            "walls",
            "actuators",
            "actuator_positions",
            "geometries",
        }:
            raise ValueError("Invalid dynamic network configuration fields")
        if set(d["fluid"]) != {"components", "binary_interactions"}:
            raise ValueError("Network requires explicit EOS components and kij")
        fluid = PengRobinsonMixture(
            tuple(PengRobinsonFluid(**c) for c in d["fluid"]["components"]),
            d["fluid"]["binary_interactions"],
        )
        vessels = {name: DynamicVessel(name=name, **spec) for name, spec in d["vessels"].items()}
        network = DynamicVesselNetwork(
            vessels,
            tuple(vessel_connection_from_dict(c) for c in d["connections"]),
            walls={name: VesselWall(**spec) for name, spec in d.get("walls", {}).items()},
            actuators={
                name: ConnectionActuator(**spec) for name, spec in d.get("actuators", {}).items()
            },
            geometries={
                name: SeparatorGeometry(**spec) for name, spec in d.get("geometries", {}).items()
            },
        )
        if set(d["initial"]) != set(vessels):
            raise ValueError("Initial states must match vessel names")
        states = network.initialize(fluid, d["initial"])
        checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
        result = run_network_schedule(
            fluid,
            network,
            states,
            d["periods"],
            stepper=d.get("stepper"),
            checkpoint=checkpoint,
            stop_after=args.stop_after,
            protections=tuple(NetworkTrip(**r) for r in d.get("protections", [])),
            event_time_tolerance_s=d.get("event_time_tolerance_s", 1e-7),
            initial_actuator_positions=d.get("actuator_positions"),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=args.output.parent,
            prefix="." + args.output.name,
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.output)
        temporary = None
    except (OSError, ValueError, KeyError, TypeError, HydraulicConvergenceError) as error:
        parser.exit(2, f"{error}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
