"""Standalone frozen-composition homogeneous pipe calculation, without FIM binding."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.gas import IdealGas
from surface.liquid import HydraulicConvergenceError
from surface.multiphase import HomogeneousFluid, HomogeneousPipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        cfg = json.loads(args.config.read_text())
        if set(cfg) != {"fluid", "pipe", "inlet"}:
            raise ValueError("Expected only fluid, pipe, inlet")
        spec = dict(cfg["fluid"])
        spec["gas"] = IdealGas(
            **spec.get("gas", {"specific_gas_constant_j_kg_k": 500, "heat_capacity_ratio": 1.3})
        )
        fluid = HomogeneousFluid(**spec)
        result = HomogeneousPipe(**cfg["pipe"]).run(**cfg["inlet"], fluid=fluid)
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(text)
        else:
            print(text, end="")
    except (ValueError, TypeError, KeyError, OSError, HydraulicConvergenceError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
