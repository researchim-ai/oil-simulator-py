"""Run capacity-limited steady facility periods with checked continuation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_surface_facility import load_facility

from surface import (
    FacilityCapacityControl,
    FacilityPeriod,
    HydraulicConvergenceError,
    run_facility_schedule,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("facility", type=Path)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--gathering", action="store_true", help="Use a gathering-facility configuration"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    try:
        if args.gathering:
            from run_surface_gathering import load_gathering

            facility = load_gathering(json.loads(args.facility.read_text()))
        else:
            facility = load_facility(json.loads(args.facility.read_text()))
        data = json.loads(args.schedule.read_text())
        unknown = set(data) - {"periods", "capacity_control"}
        if unknown:
            raise ValueError(f"Unknown schedule fields: {sorted(unknown)}")
        periods = [FacilityPeriod(**period) for period in data["periods"]]
        control = (
            FacilityCapacityControl(**data["capacity_control"])
            if data.get("capacity_control") is not None
            else None
        )
        saved = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
        checkpoint = saved.get("checkpoint", saved) if saved is not None else None
        result = run_facility_schedule(facility, periods, control, checkpoint, args.stop_after)
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end="")
    except (OSError, KeyError, TypeError, ValueError, HydraulicConvergenceError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
