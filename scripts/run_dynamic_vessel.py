"""Run a rigid PR equilibrium vessel with conservative dynamic JSON restart."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    DynamicVessel,
    HydraulicConvergenceError,
    PengRobinsonFluid,
    PengRobinsonMixture,
    run_vessel_schedule,
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
        data = json.loads(args.config.read_text())
        if not {"fluid", "vessel", "initial", "periods"} <= set(data) or set(data) - {
            "fluid",
            "vessel",
            "initial",
            "periods",
            "stepper",
        }:
            raise ValueError("Invalid dynamic vessel configuration fields")
        if set(data["fluid"]) != {"components", "binary_interactions"}:
            raise ValueError("Vessel requires explicit components and kij")
        fluid = PengRobinsonMixture(
            tuple(PengRobinsonFluid(**x) for x in data["fluid"]["components"]),
            data["fluid"]["binary_interactions"],
        )
        vessel = DynamicVessel(**data["vessel"])
        initial = vessel.initialize(fluid, **data["initial"])
        checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
        result = run_vessel_schedule(
            fluid,
            vessel,
            initial,
            data["periods"],
            stepper=data.get("stepper"),
            checkpoint=checkpoint,
            stop_after=args.stop_after,
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
    except (OSError, ValueError, TypeError, KeyError, HydraulicConvergenceError) as error:
        parser.exit(2, f"{error}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
