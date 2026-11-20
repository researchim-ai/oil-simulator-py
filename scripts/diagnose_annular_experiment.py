"""Separate numerical implementation error from the external pressure mismatch.

Independent scalar radial quadrature and Brent root solve at measured inputs.
The forward check does not use measured pressure. The separate inverse
diagnostic uses it explicitly to identify required shear, never to fit or
replace the production closure.
SciPy is reference-only; production remains Torch.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from reference_pr_annular_source import resistance
from scipy.optimize import brentq
from validate_annular_film_experiment import validate
from validate_multiphase_experiments import ROOT, run


def implied_interface(jg, ml, rg, friction_gradient):
    """Conditional inverse of the fully wetted Stokes model, without Wallis.

    friction_gradient is positive gas friction loss per length AFTER removing
    hydrostatics and local losses. Neither a wet fraction nor an empirical
    coefficient can be identified by this calculation.
    """
    if any(not math.isfinite(v) or v <= 0 for v in (jg, ml, rg, friction_gradient)):
        raise ValueError("Positive finite flow, density and distributed friction required")
    diameter, mu, rho, gravity = 0.085, 0.001002, 998.2, 9.80665
    if rg >= rho or ml / (math.pi * diameter * mu) > 400:
        raise ValueError("Outside declared laminar liquid/density domain")
    area = math.pi * diameter**2 / 4
    gradient = -rg * gravity - friction_gradient

    def state(alpha):
        eta, beta, wall, perimeter = resistance(alpha, diameter, mu)
        radius = diameter / 2 * math.sqrt(1 - alpha)
        tau = friction_gradient * radius / 2
        ul = -ml / (rho * area * alpha)
        residual = -area * alpha * (gradient + rho * gravity) + perimeter * eta * tau - wall * ul
        return residual, tau, eta * ul + beta * tau

    grid = np.geomspace(1e-5, 0.3, 160)
    for left, right in zip(grid[:-1], grid[1:]):
        if state(left)[0] >= 0 and state(right)[0] <= 0:
            alpha = brentq(lambda a: state(a)[0], left, right, xtol=1e-14)
            residual, tau, ui = state(alpha)
            ug = jg / (1 - alpha)
            if tau * (ug - ui) < 0:
                raise ValueError("Implied shear is not passive")
            delta = diameter / 2 * alpha / (1 + math.sqrt(1 - alpha))
            wallis = 0.005 * (1 + 300 * delta / diameter)
            fanning = 2 * tau / (rg * jg**2)
            return {
                "liquid_holdup": alpha,
                "film_thickness_m": delta,
                "implied_shear_pa": tau,
                "implied_superficial_fanning_factor": fanning,
                "wallis_superficial_fanning_factor": wallis,
                "implied_to_wallis_ratio": fanning / wallis,
                "liquid_force_residual_n_m": residual,
            }
    raise ValueError("No implied thin-film root in declared domain")


def pressure_inversion(raw, external):
    """Keep every window and both dry baselines; failures stay explicit."""
    import torch

    from surface.pr_active_pipe_friction import darcy_factor

    identity = lambda r: (r["sheet"], r["first_row"], r["last_row"])
    inputs = {identity(r): r for r in raw}
    rows = []
    for variant in external["roughness_sensitivity"].values():
        roughness = variant["roughness_m"]
        for case in variant["cases"]:
            r = inputs[identity(case)]
            jg = r["gas_volume_flow_m3_h"] / 3600 / (math.pi * 0.085**2 / 4)
            rg = r["absolute_pressure_pa"] / (287.05 * r["temperature_k"])
            re = torch.tensor(rg * jg * 0.085 / 1.81e-5, dtype=torch.float64)
            dry = float(darcy_factor(re, roughness / 0.085)) * rg * jg**2 / (2 * 0.085)
            # Same observation assumptions as the forward challenge. Offsets
            # are the existing instrumental-only bound, not fitted tolerances.
            outcomes = []
            for offset in (-4.0, 0.0, 4.0):
                friction = dry + (case["measured_wet_increment_pa"] + offset) / 3
                try:
                    result = implied_interface(jg, r["liquid_mass_flow_kg_s"], rg, friction)
                    result["identified_within_assumptions"] = True
                except ValueError as error:
                    result = {"identified_within_assumptions": False, "reason": str(error)}
                outcomes.append({"instrument_offset_pa": offset, **result})
            rows.append({"window": identity(case), "roughness_m": roughness, "outcomes": outcomes})
    summary = {}
    for roughness in sorted({row["roughness_m"] for row in rows}):
        outcomes = [
            outcome
            for row in rows
            if row["roughness_m"] == roughness
            for outcome in row["outcomes"]
        ]
        ratios = [
            r["implied_to_wallis_ratio"] for r in outcomes if r["identified_within_assumptions"]
        ]
        summary[str(roughness)] = {
            "evaluated_states": len(outcomes),
            "identified_states": len(ratios),
            "implied_to_wallis_range": [min(ratios), max(ratios)] if ratios else None,
            "all_identified_below_wallis": bool(ratios) and max(ratios) < 1,
        }
    return {
        "cases": rows,
        "roughness_summary": summary,
        "pressure_used_as_input": True,
        "fit_performed": False,
        "replaces_runtime_closure": False,
        "experimental_validation": False,
        "assumptions": [
            "steady fully wetted laminar annulus",
            "wet/dry local losses cancel",
            "dry distributed loss follows each declared Darcy baseline",
            "constant properties and no entrainment",
        ],
        "interpretation": "Conditional shear identification, not an independently measured shear or validation. "
        "Pressure alone cannot distinguish shear, wetting and local-loss errors.",
    }


def dry_density_sensitivity(baseline):
    """Conditional fixed-volume-flow density scaling of measured dry losses.

    Local inertial and gas hydrostatic terms scale with density, while Darcy's
    Reynolds dependence is neglected here and remains an explicit limitation.
    This diagnostic never changes the archived raw-pressure challenge score.
    """
    dry = baseline["dry_baseline"]
    q = np.array([r["q"] for r in dry])
    dp = np.array([r["dp"] for r in dry])
    rho = np.array([r["absolute_pressure_pa"] / (287.05 * r["temperature_k"]) for r in dry])
    cases = []
    for r in baseline["roughness_sensitivity"]["0.0"]["cases"]:
        wet_rho = r["absolute_pressure_pa"] / (287.05 * r["temperature_k"])
        reference = float(np.interp(r["gas_volume_flow_m3_h"], q, dp / rho) * wet_rho)
        cases.append(
            {
                "window": [r["sheet"], r["first_row"], r["last_row"]],
                "original_dry_baseline_pa": r["measured_dry_baseline_pa"],
                "density_scaled_dry_baseline_pa": reference,
                "change_in_wet_increment_pa": r["measured_dry_baseline_pa"] - reference,
            }
        )
    return {
        "cases": cases,
        "maximum_absolute_increment_change_pa": max(
            abs(r["change_in_wet_increment_pa"]) for r in cases
        ),
        "model": "conditional density-proportional dry loss at matched volume flow",
        "viscosity_and_reynolds_correction_applied": False,
        "replaces_experimental_gate": False,
        "fit_performed": False,
    }


def independent_film(jg, ml, rg, *, shear_velocity="interface"):
    if shear_velocity not in ("interface", "superficial"):
        raise ValueError("Unknown diagnostic shear velocity")
    diameter, mu, rho, gravity = 0.085, 0.001002, 998.2, 9.80665
    area = math.pi * diameter**2 / 4

    def evaluate(alpha):
        eta, beta, wall, perimeter = resistance(alpha, diameter, mu)
        radius = diameter / 2 * math.sqrt(1 - alpha)
        delta = diameter / 2 - radius
        ul, ug = -ml / (rho * area * alpha), jg / (1 - alpha)
        drive = ug - eta * ul
        re = rg * abs(drive) * 2 * radius / 1.81e-5
        blend = min(1.0, max(0.0, (re - 2000) / 2000))
        blend = blend**2 * (3 - 2 * blend)
        linear = (1 - blend) * 4 * 1.81e-5 / radius
        quadratic = blend * 0.5 * 0.005 * (1 + 300 * delta / diameter) * rg
        # Solve interface slip independently rather than reuse runtime formula.
        slip = brentq(
            lambda s: s + beta * (linear * s + quadratic * s * abs(s)) - drive,
            0.0,
            drive,
            xtol=1e-13,
        )
        tau = linear * slip + quadratic * slip * abs(slip)
        if shear_velocity == "superficial":
            # Diagnostic variant only: use total-area gas velocity in Wallis.
            tau = 0.5 * 0.005 * (1 + 300 * delta / diameter) * rg * jg**2
        gradient = -rg * gravity - 2 * tau / radius
        residual = -area * alpha * (gradient + rho * gravity) + perimeter * eta * tau - wall * ul
        return residual, gradient

    grid = np.geomspace(1e-5, 0.3, 160)
    previous, left = evaluate(grid[0])[0], grid[0]
    for right in grid[1:]:
        current = evaluate(right)[0]
        if previous >= 0 and current <= 0:
            alpha = brentq(lambda a: evaluate(a)[0], left, right, xtol=1e-14)
            return alpha, evaluate(alpha)[1]
        previous, left = current, right
    raise ValueError("No independent thin-film root")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = ROOT / "tests/data/surface/hannover_countercurrent"
    external = validate(data, shear_velocity="interface")
    baseline = run(data)
    raw = baseline["roughness_sensitivity"]["0.0"]["cases"]
    identity = lambda r: (r["sheet"], r["first_row"], r["last_row"])
    inputs = {identity(r): r for r in raw}
    rows = []
    for expected in external["roughness_sensitivity"]["0.0"]["cases"]:
        r = inputs[identity(expected)]
        jg = r["gas_volume_flow_m3_h"] / 3600 / (math.pi * 0.085**2 / 4)
        rg = r["absolute_pressure_pa"] / (287.05 * r["temperature_k"])
        alpha, gradient = independent_film(jg, r["liquid_mass_flow_kg_s"], rg)
        # Recover production gradient from its film momentum balance, independently
        # of Darcy subtraction (which is common to wet-minus-dry reporting).
        from surface.annular_film import countercurrent_film

        production = countercurrent_film(
            jg, r["liquid_mass_flow_kg_s"], rg, shear_velocity="interface"
        )
        rows.append(
            {
                "window": identity(r),
                "holdup_difference": abs(alpha - float(production["liquid_holdup"])),
                "pressure_span_difference_pa": 3
                * abs(gradient - float(production["pressure_gradient_pa_m"])),
                "experimental_error_pa": expected["error_pa"],
            }
        )
    result = {
        "cases": rows,
        "numerical_budgets": {"holdup": 1e-8, "pressure_span_pa": 1e-5},
        "numerical_check_passed": all(
            r["holdup_difference"] <= 1e-8 and r["pressure_span_difference_pa"] <= 1e-5
            for r in rows
        ),
        "experimental_check_passed": external["accepted_as_validated_for_annular_countercurrent"],
        "qualification_blockers": external["qualification_blockers"],
        "fit_performed": False,
        "conditional_pressure_inversion": pressure_inversion(raw, external),
        "dry_density_sensitivity": dry_density_sensitivity(baseline),
        "scope": "Steady fully-wetted laminar film; not transient or partial-wetting validation",
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "scripts/diagnose_annular_experiment.py",
                "scripts/reference_pr_annular_source.py",
            )
        },
    }
    (args.output_dir / "experiment.json").write_text(
        json.dumps(external, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "diagnosis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("cases", "conditional_pressure_inversion", "dry_density_sensitivity")
            },
            indent=2,
        )
    )
    print(json.dumps(result["conditional_pressure_inversion"]["roughness_summary"], indent=2))
    # This command diagnoses; success does not mean experimental qualification.
    return 0 if result["numerical_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
