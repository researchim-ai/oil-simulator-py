"""Audit the zero-film friction offset without changing experimental scoring.

Wallis' turbulent Fanning baseline .005 implies Darcy .020 as the film
vanishes; the dry branch uses a Reynolds/roughness-dependent Darcy factor.
This diagnostic quantifies that difference, not a replacement correlation.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from validate_annular_film_experiment import validate
from validate_multiphase_experiments import ROOT, run

from surface.pr_active_pipe_friction import darcy_factor


def dry_offset(jg, rho, roughness, diameter=0.085, length=3.0):
    re = rho * jg * diameter / 1.81e-5
    if re < 4000:
        raise ValueError("This audit is restricted to the fully turbulent gas branch")
    fd = float(darcy_factor(torch.tensor(re, dtype=torch.float64), roughness / diameter))
    return length * (0.020 - fd) * rho * jg**2 / (2 * diameter)


def audit(data):
    measured = run(data)
    model = validate(data, shear_velocity="interface")
    key = lambda r: (r["sheet"], r["first_row"], r["last_row"])
    inputs = {key(r): r for r in measured["roughness_sensitivity"]["0.0"]["cases"]}
    cases = []
    for roughness, group in model["roughness_sensitivity"].items():
        for r in group["cases"]:
            raw = inputs[key(r)]
            jg = raw["gas_volume_flow_m3_h"] / 3600 / (math.pi * 0.085**2 / 4)
            rho = raw["absolute_pressure_pa"] / (287.05 * raw["temperature_k"])
            offset = dry_offset(jg, rho, float(roughness))
            cases.append(
                {
                    "window": key(r),
                    "roughness_m": float(roughness),
                    "zero_film_offset_pa": offset,
                    "experimental_error_pa": r["error_pa"],
                    "error_minus_zero_film_offset_pa": r["error_pa"] - offset,
                }
            )
    summary = {}
    for roughness in model["roughness_sensitivity"]:
        rows = [r for r in cases if r["roughness_m"] == float(roughness)]
        summary[roughness] = {
            "windows": len(rows),
            "offset_range_pa": [
                min(r["zero_film_offset_pa"] for r in rows),
                max(r["zero_film_offset_pa"] for r in rows),
            ],
            "original_mae_pa": sum(abs(r["experimental_error_pa"]) for r in rows) / len(rows),
            "offset_removed_diagnostic_mae_pa": sum(
                abs(r["error_minus_zero_film_offset_pa"]) for r in rows
            )
            / len(rows),
        }
    return {
        "summary": summary,
        "cases": cases,
        "source_sha256": model["source_sha256"],
        "experimental_qualification": model["accepted_as_validated_for_annular_countercurrent"],
        "production_changed": False,
        "fit_performed": False,
        "interpretation": "Offset removal is an algebraic diagnostic only, not a solved alternative film closure or a new validation score.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(ROOT / "tests/data/surface/hannover_countercurrent")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
