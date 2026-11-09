"""Unfitted PR saturation predictions against external NIST Antoine correlations.

This observational report does not redefine coefficients or claim validation of
pipe hydrodynamics, heat capacities, mixed hydrocarbons or real equipment.
"""

import argparse
import hashlib
import json
from pathlib import Path

from surface.peng_robinson import PengRobinsonFluid

ROOT = Path(__file__).resolve().parents[1]


def run(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if data["model"] != "external_antoine_reference_v1":
        raise ValueError("Unknown external reference")
    rows = []
    for case in data["cases"]:
        fluid = PengRobinsonFluid(**case["component"])
        lo, hi = case["valid_temperature_range_k"]
        for temperature in case["temperatures_k"]:
            if (
                not lo <= temperature <= hi
                or not fluid.min_temperature_k <= temperature < fluid.critical_temperature_k
            ):
                raise ValueError("Reference temperature outside published or PR validity envelope")
            reference = 1e5 * 10 ** (case["A"] - case["B"] / (temperature + case["C"]))
            actual = fluid.saturation_at_temperature(temperature)["pressure_pa"]
            rows.append(
                {
                    "component": fluid.name,
                    "temperature_k": temperature,
                    "reference_pressure_pa": reference,
                    "pr_pressure_pa": actual,
                    "relative_error": actual / reference - 1.0,
                    "source_url": case["source_url"],
                }
            )
    return {
        "scope": data["scope"],
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fit_performed": False,
        "maximum_absolute_relative_error": max(abs(r["relative_error"]) for r in rows),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "tests/data/surface/nist_antoine_reference.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.reference)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
