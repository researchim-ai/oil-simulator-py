"""Run an explicit-composition steady thermal process/schedule with checked restart."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    HydraulicConvergenceError,
    PengRobinsonFluid,
    PengRobinsonMixture,
    ThermalResourceLimits,
)
from surface.thermal_flowsheet import run_thermal_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    temp = None
    try:
        data = json.loads(args.config.read_text())
        if (
            not {"fluid", "units", "periods"} <= set(data)
            or set(data) - {"fluid", "units", "periods", "resource_limits"}
            or set(data["fluid"])
            != {
                "components",
                "binary_interactions",
            }
        ):
            raise ValueError("Expected fluid, units, periods with explicit component EOS and kij")
        fluid = PengRobinsonMixture(
            tuple(PengRobinsonFluid(**x) for x in data["fluid"]["components"]),
            data["fluid"]["binary_interactions"],
        )
        checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
        result = run_thermal_schedule(
            fluid,
            data["units"],
            data["periods"],
            checkpoint,
            args.stop_after,
            resource_limits=ThermalResourceLimits(**data["resource_limits"])
            if "resource_limits" in data
            else None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=args.output.parent,
            prefix="." + args.output.name,
            suffix=".tmp",
            delete=False,
        ) as out:
            temp = Path(out.name)
            json.dump(result, out, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, args.output)
        temp = None
    except (OSError, KeyError, ValueError, TypeError, HydraulicConvergenceError) as error:
        parser.exit(2, f"{error}\n")
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
