"""Run ideal black-oil separation and a conservative separated-water network."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    FacilityCapacityControl,
    HydraulicConvergenceError,
    SurfaceFacility,
)


def load_facility(data):
    return SurfaceFacility.from_dict(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument(
        "--capacity-control", type=Path, help="JSON uniform feed-curtailment specification"
    )
    args = parser.parse_args()
    try:
        facility = load_facility(json.loads(args.config.read_text()))
        initial = (
            facility.network_initial(json.loads(args.initial_state.read_text()))
            if args.initial_state
            else None
        )
        if args.capacity_control:
            controller = FacilityCapacityControl(**json.loads(args.capacity_control.read_text()))
            result = controller.solve(facility, initial=initial)
            feasible = result["facility"] is not None and result["facility"]["feasible"]
        else:
            result = facility.run(initial=initial)
            feasible = result["feasible"]
        output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output)
        else:
            print(output, end="")
    except (OSError, KeyError, TypeError, ValueError, HydraulicConvergenceError) as error:
        parser.error(str(error))
    if not feasible:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
