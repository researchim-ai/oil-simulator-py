"""Separate input, reference, velocity-definition and wetting sensitivities.

No fitted parameters or altered experimental gates. Contact-angle thresholds
are conditional predictions from the existing criterion, not material data.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from diagnose_annular_experiment import independent_film
from validate_annular_film_experiment import validate
from validate_multiphase_experiments import ROOT, run

from surface.annular_film import countercurrent_film
from surface.film_wetting import FilmWettingCriterion
from surface.pr_active_pipe_friction import darcy_factor


def critical_angle(force, surface_tension):
    if (
        not math.isfinite(force)
        or force < 0
        or not math.isfinite(surface_tension)
        or surface_tension <= 0
    ):
        raise ValueError("Finite nonnegative force and positive surface tension required")
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - force / surface_tension))))


def audit(data):
    baseline = run(data)
    challenge = validate(data, shear_velocity="interface")
    identity = lambda r: (r["sheet"], r["first_row"], r["last_row"])
    inputs = {identity(r): r for r in baseline["roughness_sensitivity"]["0.0"]["cases"]}
    rows = []
    # 20 C water reference, explicitly not a measured surface tension.
    sigma = 0.0728
    criterion = FilmWettingCriterion(sigma, 0.0, 0.0)
    for row in challenge["roughness_sensitivity"]["0.0"]["cases"]:
        raw = inputs[identity(row)]
        jg = raw["gas_volume_flow_m3_h"] / 3600 / (math.pi * 0.085**2 / 4)
        rho = raw["absolute_pressure_pa"] / (287.05 * raw["temperature_k"])
        film = countercurrent_film(
            jg, raw["liquid_mass_flow_kg_s"], rho, shear_velocity="interface"
        )
        alpha = film["liquid_holdup"]
        wetting = criterion.diagnose(
            alpha,
            0.085,
            0.001002,
            alpha * 0 + 998.2,
            film["liquid_velocity_m_s"],
            film["interfacial_shear_pa"],
        )
        valid = bool(wetting["profile_valid"])
        _, superficial_gradient = independent_film(
            jg, raw["liquid_mass_flow_kg_s"], rho, shear_velocity="superficial"
        )
        factor = float(
            darcy_factor(torch.tensor(rho * jg * 0.085 / 1.81e-5, dtype=torch.float64), 0.0)
        )
        dry = 3 * factor * rho * jg**2 / (2 * 0.085)
        superficial_increment = 3 * (-superficial_gradient - rho * 9.80665) - dry
        rows.append(
            {
                "window": identity(row),
                "gas_reynolds": rho * jg * 0.085 / 1.81e-5,
                "liquid_film_reynolds_gamma_mu": raw["liquid_mass_flow_kg_s"]
                / (math.pi * 0.085 * 0.001002),
                "dry_measured_pa": raw["measured_dry_baseline_pa"],
                "dry_friction_prediction_pa": dry,
                "wet_increment_measured_pa": raw["measured_wet_increment_pa"],
                "interface_prediction_pa": row["predicted_wet_increment_pa"],
                "superficial_velocity_prediction_pa": superficial_increment,
                "interface_error_pa": row["error_pa"],
                "superficial_velocity_error_pa": superficial_increment
                - raw["measured_wet_increment_pa"],
                "wetting_profile_valid": valid,
                "critical_advancing_angle_deg": critical_angle(
                    float(wetting["momentum_force_n_m"]), sigma
                )
                if valid
                else None,
                "curvature_ratio": float(wetting["curvature_ratio"]),
            }
        )
    dry_sensitivity = challenge["roughness_sensitivity"]["0.0"]["dry_series_sensitivity"]
    return {
        "summary": {
            "windows": len(rows),
            "interface_mae_pa": float(np.mean([abs(r["interface_error_pa"]) for r in rows])),
            "superficial_velocity_mae_pa": float(
                np.mean([abs(r["superficial_velocity_error_pa"]) for r in rows])
            ),
            "superficial_within_4pa": sum(
                abs(r["superficial_velocity_error_pa"]) <= 4 for r in rows
            ),
            "valid_wetting_profiles": sum(r["wetting_profile_valid"] for r in rows),
            "critical_advancing_angle_range_deg": [
                min(r["critical_advancing_angle_deg"] for r in rows if r["wetting_profile_valid"]),
                max(r["critical_advancing_angle_deg"] for r in rows if r["wetting_profile_valid"]),
            ]
            if any(r["wetting_profile_valid"] for r in rows)
            else None,
            "dry_series_common_windows": dry_sensitivity["common_wet_windows"],
            "dry_series_max_spread_pa": dry_sensitivity["maximum_dry_series_spread_pa"],
            "dry_series_mae_pa": {
                k: v["mean_absolute_error_pa"] for k, v in dry_sensitivity["dry_series"].items()
            },
        },
        "cases": rows,
        "source_sha256": baseline["source_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "surface_tension_assumption_n_m": sigma,
        "fit_performed": False,
        "material_contact_angles_measured": False,
        "experimental_qualification": False,
        "notes": [
            "Critical angle is conditional on the solved full-film profile and assumed surface tension; it neither measures wetted fraction nor proves dry patches.",
            "Superficial-velocity variant is a reference-only sensitivity, not a production model or a newly validated correlation.",
            "Dry measured-versus-friction difference is not scored as a failure: pressure-line hydrostatic correction and local losses must be identified first.",
            "Dry-series spread is sensitivity, not an uncertainty interval; original sample selection and experimental gate are unchanged.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(ROOT / "tests/data/surface/hannover_countercurrent")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
