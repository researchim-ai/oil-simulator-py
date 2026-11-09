"""Challenge a separately declared annular film closure on unchanged raw data.

The original failed volume-wetted result is retained as an explicit baseline.
No coefficients, sample windows, roughness or uncertainty bounds are fitted.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from validate_multiphase_experiments import ROOT, run

from surface.annular_film import countercurrent_film
from surface.pr_active_pipe_friction import darcy_factor


def validate(data, *, shear_velocity="mean"):
    baseline = run(data)
    all_rows = baseline["roughness_sensitivity"]["0.0"]["cases"]
    rows = [r for r in all_rows if r["liquid_mass_flow_kg_s"] / (math.pi * 0.085 * 0.001002) <= 400]
    excluded = [r for r in all_rows if r not in rows]
    jg = torch.tensor(
        [r["gas_volume_flow_m3_h"] / 3600 / (math.pi * 0.085**2 / 4) for r in rows],
        dtype=torch.float64,
    )
    rho = torch.tensor(
        [r["absolute_pressure_pa"] / (287.05 * r["temperature_k"]) for r in rows],
        dtype=torch.float64,
    )
    film = countercurrent_film(
        jg, [r["liquid_mass_flow_kg_s"] for r in rows], rho, shear_velocity=shear_velocity
    )
    cases = {}
    observed = np.array([r["measured_wet_increment_pa"] for r in rows])
    for roughness in (0.0, 0.0002):
        dry_friction = (
            darcy_factor(rho * jg * 0.085 / 1.81e-5, roughness / 0.085) * rho * jg**2 / (2 * 0.085)
        )
        prediction = (3 * (-film["pressure_gradient_pa_m"] - rho * 9.80665 - dry_friction)).numpy()
        error = prediction - observed
        cases[str(roughness)] = {
            "roughness_m": roughness,
            "fit_performed": False,
            "mean_absolute_error_pa": float(np.abs(error).mean()),
            "bias_pa": float(error.mean()),
            "fraction_within_instrument_only_4pa_bound": float((np.abs(error) <= 4).mean()),
            "cases": [
                {
                    "sheet": r["sheet"],
                    "first_row": r["first_row"],
                    "last_row": r["last_row"],
                    "measured_wet_increment_pa": r["measured_wet_increment_pa"],
                    "predicted_wet_increment_pa": float(p),
                    "error_pa": float(e),
                    "liquid_holdup": float(a),
                    "film_thickness_m": float(d),
                }
                for r, p, e, a, d in zip(
                    rows, prediction, error, film["liquid_holdup"], film["film_thickness_m"]
                )
            ],
        }
    return {
        "format": "annular_film_experimental_challenge_v1",
        "closure": "laminar_annular_wallis_interface_v1"
        if shear_velocity == "interface"
        else "laminar_annulus_wallis_relative_v1",
        "source": baseline["source"],
        "source_sha256": baseline["source_sha256"],
        "runtime_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["src/surface/annular_film.py", "src/surface/pr_active_pipe_friction.py"]
            + (
                ["src/surface/pr_active_annular_friction.py"]
                if shear_velocity == "interface"
                else []
            )
        },
        "validator_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "scripts/validate_annular_film_experiment.py",
                "scripts/validate_multiphase_experiments.py",
            ]
        },
        "correlation_reference": "https://doi.org/10.1016/j.applthermaleng.2026.130975",
        "correlation_variant": f"Wallis Fanning factor, relative core/{shear_velocity}-film speed, exact laminar annulus; not the superficial-gas-speed variant",
        "qualification_scope": baseline["qualification_scope"],
        "baseline": {
            key: value
            for key, value in baseline.items()
            if key not in ("roughness_sensitivity", "dry_baseline")
        },
        "baseline_mae_pa": {
            k: v["mean_absolute_error_pa"] for k, v in baseline["roughness_sensitivity"].items()
        },
        "baseline_mae_on_same_16_windows_pa": {
            k: float(
                np.mean(
                    [
                        abs(r["error_pa"])
                        for r in v["cases"]
                        if r["liquid_mass_flow_kg_s"] / (math.pi * 0.085 * 0.001002) <= 400
                    ]
                )
            )
            for k, v in baseline["roughness_sensitivity"].items()
        },
        "matched_wet_windows": len(rows),
        "held_out": True,
        "fit_performed": False,
        "excluded_by_laminar_domain": [
            {"sheet": r["sheet"], "first_row": r["first_row"], "last_row": r["last_row"]}
            for r in excluded
        ],
        "roughness_sensitivity": cases,
        "instrument_bound_note": baseline["instrument_bound_note"],
        "water_properties_assumption": baseline["water_properties_assumption"],
        "accepted_as_validated_for_annular_countercurrent": all(
            v["fraction_within_instrument_only_4pa_bound"] == 1.0 for v in cases.values()
        ),
        "integration_status": "Opt-in dynamic source, with conservative interface work; unqualified experimentally."
        if shear_velocity == "interface"
        else "Steady constitutive solver only; not enabled in transient active PR pipe.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "tests/data/surface/hannover_countercurrent")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shear-velocity", choices=["mean", "interface"], default="mean")
    args = p.parse_args()
    result = validate(args.data, shear_velocity=args.shear_velocity)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("baseline", "roughness_sensitivity")}
        )
    )
    for k, v in result["roughness_sensitivity"].items():
        print(k, {key: value for key, value in v.items() if key != "cases"})


if __name__ == "__main__":
    main()
