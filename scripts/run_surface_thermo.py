"""Explicit pure-fluid/mixture PR operations; never infer composition from black-oil PVT."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    EquilibriumNozzle,
    HydraulicConvergenceError,
    MixtureLetdown,
    PengRobinsonFluid,
    PengRobinsonMixture,
    ThermalLetdown,
)


def run_case(data):
    if set(data) - {"fluid", "operations"}:
        raise ValueError("Unknown thermal configuration fields")
    spec = data["fluid"]
    is_mixture = "components" in spec
    if is_mixture:
        if set(spec) != {"components", "binary_interactions"}:
            raise ValueError("Mixture needs explicit components and binary_interactions")
        fluid = PengRobinsonMixture(
            tuple(PengRobinsonFluid(**c) for c in spec["components"]), spec["binary_interactions"]
        )
    else:
        fluid = PengRobinsonFluid(**spec)
    results = []
    for spec in data["operations"]:
        operation = dict(spec)
        kind = operation.pop("type")
        if is_mixture and kind in ("letdown", "pt", "ph", "ps", "state", "stability"):
            if kind == "letdown":
                result = MixtureLetdown(**operation.pop("equipment")).run(fluid=fluid, **operation)
            else:
                method = {
                    "pt": "flash_pt",
                    "ph": "flash_ph",
                    "ps": "flash_ps",
                    "state": "state",
                    "stability": "stability",
                }[kind]
                result = getattr(fluid, method)(**operation)
        elif is_mixture:
            raise ValueError(f"Unsupported mixture operation: {kind}")
        elif kind == "state":
            result = fluid.state(**operation)
        elif kind == "saturation_temperature":
            result = fluid.saturation_at_temperature(**operation)
        elif kind == "saturation_pressure":
            result = fluid.saturation_at_pressure(**operation)
        elif kind == "ph":
            result = fluid.flash_ph(**operation)
        elif kind == "ps":
            result = fluid.flash_ps(**operation)
        elif kind in ("letdown", "nozzle"):
            equipment = operation.pop("equipment")
            model = (
                ThermalLetdown(**equipment) if kind == "letdown" else EquilibriumNozzle(**equipment)
            )
            result = model.run(fluid=fluid, **operation)
        else:
            raise ValueError(f"Unknown thermal operation: {kind}")
        results.append({"type": kind, "result": result})
    return {
        "model": (
            "mixture_peng_robinson_explicit_composition"
            if is_mixture
            else "pure_fluid_peng_robinson_explicit_caloric_data"
        ),
        "fluid": data["fluid"],
        "operations": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_case(json.loads(args.config.read_text()))
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    except (OSError, KeyError, ValueError, TypeError, HydraulicConvergenceError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
