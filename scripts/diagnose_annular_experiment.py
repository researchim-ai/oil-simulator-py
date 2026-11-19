"""Separate numerical implementation error from the external pressure mismatch.

Independent scalar radial quadrature and Brent root solve at measured inputs.
No measured pressure is used to solve film thickness or fit a coefficient.
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


def independent_film(jg, ml, rg):
    diameter, mu, rho, gravity = .085, .001002, 998.2, 9.80665
    area = math.pi * diameter**2 / 4

    def evaluate(alpha):
        eta, beta, wall, perimeter = resistance(alpha, diameter, mu)
        radius = diameter / 2 * math.sqrt(1 - alpha)
        delta = diameter / 2 - radius
        ul, ug = -ml / (rho * area * alpha), jg / (1 - alpha)
        drive = ug - eta * ul
        re = rg * abs(drive) * 2 * radius / 1.81e-5
        blend = min(1., max(0., (re - 2000) / 2000))
        blend = blend**2 * (3 - 2 * blend)
        linear = (1 - blend) * 4 * 1.81e-5 / radius
        quadratic = blend * .5 * .005 * (1 + 300 * delta / diameter) * rg
        # Solve interface slip independently rather than reuse runtime formula.
        slip = brentq(lambda s: s + beta * (linear*s + quadratic*s*abs(s)) - drive,
                      0., drive, xtol=1e-13)
        tau = linear * slip + quadratic * slip * abs(slip)
        gradient = -rg * gravity - 2 * tau / radius
        residual = -area * alpha * (gradient + rho * gravity) + perimeter * eta * tau - wall * ul
        return residual, gradient

    grid = np.geomspace(1e-5, .3, 160)
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
    raw = run(data)["roughness_sensitivity"]["0.0"]["cases"]
    identity = lambda r: (r["sheet"], r["first_row"], r["last_row"])
    inputs = {identity(r): r for r in raw}
    rows = []
    for expected in external["roughness_sensitivity"]["0.0"]["cases"]:
        r = inputs[identity(expected)]
        jg = r["gas_volume_flow_m3_h"] / 3600 / (math.pi * .085**2 / 4)
        rg = r["absolute_pressure_pa"] / (287.05 * r["temperature_k"])
        alpha, gradient = independent_film(jg, r["liquid_mass_flow_kg_s"], rg)
        # Recover production gradient from its film momentum balance, independently
        # of Darcy subtraction (which is common to wet-minus-dry reporting).
        from surface.annular_film import countercurrent_film
        production = countercurrent_film(jg, r["liquid_mass_flow_kg_s"], rg,
                                        shear_velocity="interface")
        rows.append({"window": identity(r), "holdup_difference":
                     abs(alpha - float(production["liquid_holdup"])),
                     "pressure_span_difference_pa":
                     3 * abs(gradient - float(production["pressure_gradient_pa_m"])),
                     "experimental_error_pa": expected["error_pa"]})
    result = {"cases": rows,
              "numerical_budgets": {"holdup": 1e-8, "pressure_span_pa": 1e-5},
              "numerical_check_passed": all(r["holdup_difference"] <= 1e-8 and
                                            r["pressure_span_difference_pa"] <= 1e-5 for r in rows),
              "experimental_check_passed": external["accepted_as_validated_for_annular_countercurrent"],
              "qualification_blockers": external["qualification_blockers"],
              "fit_performed": False,
              "scope": "Steady fully-wetted laminar film; not transient or partial-wetting validation",
              "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                  ("scripts/diagnose_annular_experiment.py", "scripts/reference_pr_annular_source.py")}}
    (args.output_dir / "experiment.json").write_text(json.dumps(external, indent=2, allow_nan=False)+"\n")
    (args.output_dir / "diagnosis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    # This command diagnoses; success does not mean experimental qualification.
    return 0 if result["numerical_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
