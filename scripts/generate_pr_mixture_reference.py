"""Regenerate independent VLE fixtures with optional CoolProp 7.2.0.

Install the reference package separately; it is not a simulator dependency.
The PQ input pair is deliberate: homogeneous PT roots do not validate VLE.
"""

import argparse
import json
import math
from pathlib import Path


def main():
    import CoolProp
    import CoolProp.CoolProp as CP

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/surface_thermal_mixture.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if CoolProp.__version__ != "7.2.0":
        parser.error("Reference convention is pinned to CoolProp 7.2.0")
    data = json.loads(args.config.read_text())
    expected_names = ["Methane", "Ethane"]
    components = data["fluid"]["components"]
    if [c["name"] for c in components] != expected_names:
        parser.error("Reference generator requires Methane/Ethane in that order")
    for name, component in zip(expected_names, components):
        ref = CP.AbstractState("PR", name)
        for field, actual in (
            ("critical_temperature_k", ref.T_critical()),
            ("critical_pressure_pa", ref.p_critical()),
            ("acentric_factor", ref.acentric_factor()),
            ("molar_mass_kg_mol", ref.molar_mass()),
        ):
            if not math.isclose(component[field], actual, rel_tol=1e-12):
                parser.error(f"Component {name} {field} differs from reference data")
    rows = []
    for kij in [0, -0.03]:
        state = CP.AbstractState("PR", "Methane&Ethane")
        state.set_mole_fractions([0.5, 0.5])
        state.set_binary_interaction_double(0, 1, "kij", kij)
        for p in [1e6, 3e6]:
            for q in [0.1, 0.5, 0.9]:
                state.update(CP.PQ_INPUTS, p, q)
                rows.append(
                    {
                        "pressure_pa": p,
                        "temperature_k": state.T(),
                        "kij": kij,
                        "vapor_molar_fraction": q,
                        "density_kg_m3": state.rhomass(),
                        "liquid_mole_fractions": state.mole_fractions_liquid(),
                        "vapor_mole_fractions": state.mole_fractions_vapor(),
                    }
                )
    result = {
        "source": "https://coolprop.org/coolprop/Cubics.html",
        "version": CoolProp.__version__,
        "backend": "PR",
        "inputs": "PQ flash, not homogeneous PT root; constant kij explicitly set",
        "components": data["fluid"]["components"],
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
