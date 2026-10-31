"""Run the steady PVT gathering tree, separators, water network, and gas export."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_surface_facility import load_facility

from surface.blackoil_network import BlackOilProductionNetwork
from surface.dispatch import FacilityCapacityControl
from surface.facility import FacilityLimit
from surface.gathering import GatheringFacility
from surface.liquid import HydraulicConvergenceError


def load_gathering(data):
    if set(data) - {"facility", "gathering", "feed_nodes", "limits", "pressure_control"}:
        raise ValueError("Unknown gathering facility fields")
    facility = load_facility(data["facility"])
    network = BlackOilProductionNetwork.from_dict(data["gathering"])
    return GatheringFacility(
        network,
        facility,
        data["feed_nodes"],
        [FacilityLimit(**s) for s in data.get("limits", [])],
        pressure_control=data.get("pressure_control"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capacity-control", type=Path)
    parser.add_argument(
        "--max-pressure",
        nargs=2,
        action="append",
        default=[],
        metavar=("NODE", "PA"),
        help="Additional absolute gathering-node pressure limit; repeat for several nodes",
    )
    args = parser.parse_args()
    try:
        plant = load_gathering(json.loads(args.config.read_text()))
        for node, raw in args.max_pressure:
            if node not in plant.gathering_network.by_name:
                raise ValueError(f"Unknown gathering pressure-limit node: {node}")
            value = float(raw)
            if not value > 0:
                raise ValueError("Gathering pressure limit must be positive Pa")
            plant.limits = (
                *plant.limits,
                FacilityLimit(
                    f"cli_max_pressure:{node}", f"gathering_pressure:{node}", "Pa", maximum=value
                ),
            )
        if args.capacity_control:
            result = FacilityCapacityControl(**json.loads(args.capacity_control.read_text())).solve(
                plant
            )
            feasible = result["facility"] is not None and result["facility"]["feasible"]
        else:
            result = plant.run()
            feasible = result["feasible"]
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end="")
    except (ValueError, TypeError, KeyError, OSError, HydraulicConvergenceError) as error:
        parser.exit(2, f"{error}\n")
    if not feasible:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
