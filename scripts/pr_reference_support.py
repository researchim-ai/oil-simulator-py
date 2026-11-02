"""Reference-only PR convention adapters and explicit comparison tolerances.

Never used by production solvers. CoolProp's rounded PR constants are matched
only in the adapter lane; a separate lane checks the unchanged exact constants.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import PengRobinsonFluid, PengRobinsonMixture, single_phase_properties
from surface.peng_robinson import OMEGA_A, OMEGA_B


class CoolPropConventionPR(PengRobinsonFluid):
    @property
    def b(self):
        return super().b * 0.07780 / OMEGA_B

    @property
    def attraction_coefficient(self):
        return super().attraction_coefficient * 0.45724 / OMEGA_A


ROOT_TOLERANCES = {
    "density_kg_m3": (2e-9, 1e-8, 0.001, 1e-5),
    "compressibility_factor": (2e-9, 1e-9, 0.001, 1e-5),
    "molar_volume_m3_mol": (2e-9, 1e-14, 0.001, 1e-10),
    "molar_enthalpy_j_mol": (2e-9, 2e-6, 0.002, 0.2),
    "molar_entropy_j_mol_k": (2e-9, 2e-8, 0.002, 0.002),
    "cp_j_mol_k": (2e-8, 2e-7, 0.003, 0.002),
    "cv_j_mol_k": (2e-8, 2e-7, 0.003, 0.002),
    "sound_speed_m_s": (2e-8, 2e-5, 0.002, 0.01),
    "molar_volume_pressure_derivative": (2e-8, 1e-16, 0.005, 1e-16),
    "molar_volume_temperature_derivative": (2e-8, 1e-13, 0.005, 1e-11),
}


def fluid_for(data, case, matched):
    cls = CoolPropConventionPR if matched else PengRobinsonFluid
    components = tuple(cls(**data["components"][i]) for i in case["component_indices"])
    return components[0] if len(components) == 1 else PengRobinsonMixture(components, case["kij"])


def root_values(fluid, case):
    p, t = case["pressure_pa"], case["temperature_k"]
    if isinstance(fluid, PengRobinsonMixture):
        q = single_phase_properties(fluid, p, t, case["mole_fractions"])
        root = fluid.state(p, t, case["mole_fractions"], phase=case["phase"])
        if not math.isclose(q["density_kg_m3"], root["density_kg_m3"], rel_tol=1e-8):
            raise ValueError("Reference root is not the stable single-phase branch")
        return q
    q = fluid.state(p, t, phase=case["phase"])
    mw = fluid.molar_mass_kg_mol
    return {
        "density_kg_m3": q["density_kg_m3"],
        "compressibility_factor": q["compressibility_factor"],
        "molar_volume_m3_mol": mw / q["density_kg_m3"],
        "molar_enthalpy_j_mol": q["enthalpy_j_kg"] * mw,
        "molar_entropy_j_mol_k": q["entropy_j_kg_k"] * mw,
        "cp_j_mol_k": q["cp_j_kg_k"] * mw,
        "cv_j_mol_k": q["cv_j_kg_k"] * mw,
        "sound_speed_m_s": q["sound_speed_m_s"],
        "molar_volume_pressure_derivative": q["derivatives"]["specific_volume_m3_kg"]["pressure_pa"]
        * mw,
        "molar_volume_temperature_derivative": q["derivatives"]["specific_volume_m3_kg"][
            "temperature_k"
        ]
        * mw,
        "log_fugacity_coefficients": [q["log_fugacity_coefficient"]],
    }


def metric(actual, expected, relative, absolute):
    actual, expected = float(actual), float(expected)
    if not all(math.isfinite(x) for x in (actual, expected)):
        raise ValueError("Nonfinite reference or computed value")
    error = abs(actual - expected)
    tolerance = absolute + relative * abs(expected)
    return {
        "actual": actual,
        "expected": expected,
        "absolute_error": error,
        "relative_error": error / max(abs(expected), 1e-30),
        "tolerance": tolerance,
        "passed": error <= tolerance,
    }


def comparisons(kind, case, fluid, matched, method=None):
    if kind == "homogeneous":
        actual = root_values(fluid, case)
        expected = case["expected"]
        metrics = {
            name: metric(
                actual[name], expected[name], tol[0 if matched else 2], tol[1 if matched else 3]
            )
            for name, tol in ROOT_TOLERANCES.items()
        }
        for i, (a, b) in enumerate(
            zip(actual["log_fugacity_coefficients"], expected["log_fugacity_coefficients"])
        ):
            metrics[f"log_phi_{i}"] = metric(
                a, b, 2e-8 if matched else 0.002, 1e-8 if matched else 0.0002
            )
        return metrics
    if kind == "vle":
        q = fluid.flash_pt(case["pressure_pa"], case["temperature_k"], case["mole_fractions"])
    elif kind == "inverse":
        key = "molar_enthalpy_j_mol" if method == "ph" else "molar_entropy_j_mol_k"
        value = case.get(key, case.get("expected", {}).get(key))
        if isinstance(fluid, PengRobinsonMixture):
            q = getattr(fluid, "flash_" + method)(
                case["pressure_pa"], value, case["mole_fractions"]
            )
        else:
            q = getattr(fluid, "flash_" + method)(
                case["pressure_pa"], value / fluid.molar_mass_kg_mol
            )
        metrics = {
            "temperature_k": metric(
                q["temperature_k"], case["temperature_k"], 0, 2e-5 if matched else 0.05
            )
        }
        if "vapor_molar_fraction" in case:
            if q.get("phase_count") != 2:
                raise ValueError("Reference inverse VLE has not returned two phases")
            metrics["vapor_molar_fraction"] = metric(
                q["vapor_molar_fraction"],
                case["vapor_molar_fraction"],
                0,
                2e-6 if matched else 0.0015,
            )
        return metrics
    else:
        raise ValueError("Unknown reference comparison kind")
    if q["phase_count"] != 2:
        raise ValueError("Reference VLE did not return two phases")
    metrics = {
        "vapor_molar_fraction": metric(
            q["vapor_molar_fraction"], case["vapor_molar_fraction"], 0, 3e-7 if matched else 0.0005
        ),
        "density_kg_m3": metric(
            q["density_kg_m3"], case["density_kg_m3"], 3e-7 if matched else 0.001, 1e-7
        ),
        "molar_enthalpy_j_mol": metric(
            q["molar_enthalpy_j_mol"],
            case["molar_enthalpy_j_mol"],
            2e-8 if matched else 0.002,
            2e-6 if matched else 0.2,
        ),
        "molar_entropy_j_mol_k": metric(
            q["molar_entropy_j_mol_k"],
            case["molar_entropy_j_mol_k"],
            2e-8 if matched else 0.002,
            2e-7 if matched else 0.002,
        ),
    }
    for phase, name in zip(q["phases"], ("liquid", "vapor")):
        for i, (a, b) in enumerate(zip(phase["mole_fractions"], case[name + "_mole_fractions"])):
            metrics[f"{name}_x_{i}"] = metric(a, b, 0, 3e-7 if matched else 0.0005)
    return metrics


def cases(data):
    for matched in (False, True):
        for kind, key in [("homogeneous", "homogeneous_cases"), ("vle", "vle_cases")]:
            for case in data[key]:
                yield kind, case, matched, None
        for case in [data["vle_cases"][i] for i in (0, 7, 12, 23, 24, 31)] + data[
            "homogeneous_cases"
        ][:3]:
            for method in ("ph", "ps"):
                yield "inverse", case, matched, method


def validate_one(data, kind, case, matched, method=None):
    result = {
        "kind": kind,
        "case": case["id"],
        "convention": "matched_coolprop" if matched else "production_exact",
        "method": method,
    }
    try:
        metrics = comparisons(kind, case, fluid_for(data, case, matched), matched, method)
        result.update(metrics=metrics, passed=all(x["passed"] for x in metrics.values()))
    except (RuntimeError, ValueError, KeyError, ZeroDivisionError) as error:
        result.update(passed=False, error=str(error))
    return result
