"""Conditional reconstruction of the dry/wet Hannover measurement span.

No fitting. Paper contraction coefficient and expansion-area ratio inferred
from the author's dry velocities. Equal inlet/outlet chambers and common gas
impulse-line density are assumptions, so this is not an identified full rig.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from audit_hannover_air_reference import author_summary
from validate_annular_film_experiment import validate
from validate_multiphase_experiments import ROOT, run

from surface.annular_film import countercurrent_film
from surface.pressure_observation import PressureTapSpan


def run_span(data):
    metadata, author = author_summary(data / "air_reference.xlsx")
    diameter = metadata["declared_geometry"]["pipe_diameter_m"]
    lower_diameter = metadata["declared_geometry"]["lower_diameter_m"]
    area = math.pi * diameter**2 / 4
    ratios = [r["author_lower_velocity_m_s"] / r["author_pipe_velocity_m_s"] for r in author]
    if max(ratios) - min(ratios) > 1e-10 or not 0 < min(ratios) < 1:
        raise ValueError("Inconsistent author area ratios")
    ratio = (diameter / lower_diameter) ** 2
    if not math.isclose(ratios[0], ratio, rel_tol=1e-10):
        raise ValueError("Author geometry disagrees with author velocities")
    chamber_area = math.pi * lower_diameter**2 / 4
    # Eq. 1 and Eq. 4. These are prescribed values, not fitted to pressure.
    contraction = 0.32
    expansion = (1 - ratio) ** 2

    def model(rho):
        return PressureTapSpan(
            3.0,
            diameter,
            chamber_area,
            chamber_area,
            3.0,
            contraction + expansion,
            -rho * 9.80665 * 3,
        )

    dry = []
    for r in author:
        result = model(r["rho_kg_m3"]).predict(r["q_m3_h"] / 3600, r["rho_kg_m3"], r["mu_pa_s"])
        prediction = float(result["sensor_difference_pa"])
        dry.append(
            {
                **r,
                "pressure_terms_pa": {k: float(v) for k, v in result.items()},
                "prediction_pa": prediction,
                "cfd_difference_pa": prediction - r["author_simulation_pa"],
                "cfd_relative_difference": prediction / r["author_simulation_pa"] - 1,
            }
        )
    baseline = run(data)
    reference = validate(data, shear_velocity="superficial")
    key = lambda r: (r["sheet"], r["first_row"], r["last_row"])
    inputs = {key(r): r for r in baseline["roughness_sensitivity"]["0.0"]["cases"]}
    wet = []
    for r in reference["roughness_sensitivity"]["0.0"]["cases"]:
        raw = inputs[key(r)]
        q = raw["gas_volume_flow_m3_h"] / 3600
        rho = raw["absolute_pressure_pa"] / (287.05 * raw["temperature_k"])
        film = countercurrent_film(
            q / area, raw["liquid_mass_flow_kg_s"], rho, shear_velocity="superficial"
        )
        pipe_friction = 3 * (-film["pressure_gradient_pa_m"] - rho * 9.80665)
        dry_prediction = model(rho).predict(q, rho, 1.81e-5)
        wet_prediction = model(rho).predict(q, rho, 1.81e-5, distributed_friction_pa=pipe_friction)
        increment = float(
            wet_prediction["sensor_difference_pa"] - dry_prediction["sensor_difference_pa"]
        )
        residual = raw["measured_wet_increment_pa"] - increment
        wet.append(
            {
                "window": key(r),
                "measured_increment_pa": raw["measured_wet_increment_pa"],
                "predicted_increment_pa": increment,
                "required_change_in_local_losses_pa": residual,
                "positive_additional_loss_can_match": residual >= 0,
                "wet_pressure_terms_pa": {k: float(v) for k, v in wet_prediction.items()},
            }
        )
    return {
        "source": metadata,
        "area_ratio_from_declared_geometry": ratio,
        "area_ratio_crosschecked_against_author_velocities": True,
        "contraction_k": contraction,
        "expansion_k": expansion,
        "dry": dry,
        "wet": wet,
        "summary": {
            "dry_points": len(dry),
            "wet_points": len(wet),
            "dry_max_cfd_relative_difference": max(abs(r["cfd_relative_difference"]) for r in dry),
            "wet_increment_mae_pa": float(
                np.mean([abs(r["required_change_in_local_losses_pa"]) for r in wet])
            ),
            "required_local_loss_change_range_pa": [
                min(r["required_change_in_local_losses_pa"] for r in wet),
                max(r["required_change_in_local_losses_pa"] for r in wet),
            ],
            "positive_additional_loss_can_match_count": sum(
                r["positive_additional_loss_can_match"] for r in wet
            ),
        },
        "assumptions": [
            "Outlet chamber area equals lower inlet chamber area declared in the author workbook.",
            "Constant gas density, unit kinetic-energy correction, gas-filled impulse lines cancel gas hydrostatics.",
            "Dry contraction/expansion coefficients are unchanged by liquid application; this is not measured.",
        ],
        "missing_identification": [
            "Upper tap geometry and detailed distributor flow path",
            "Impulse-line fill and elevations",
            "Wet modification of local losses",
            "Film coverage, thickness and entrainment",
        ],
        "fit_performed": False,
        "fully_identified_rig": False,
        "experimentally_qualified": False,
        "interpretation": "Positive extra injection loss cannot fix an overprediction. Negative required changes do not prove actual losses decrease or identify wetting.",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "src/surface/pressure_observation.py",
                ROOT / "src/surface/annular_film.py",
            ]
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_span(ROOT / "tests/data/surface/hannover_countercurrent")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
