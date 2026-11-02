"""Generate independent PR caloric/VLE fixtures with optional CoolProp 7.2.0.

Only CoolProp and explicit ideal-gas conventions generate the expected values.
This script never imports the simulator. Runtime solvers do not require CoolProp.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

VERSION = "7.2.0"
SOURCE = "https://coolprop.org/coolprop/Cubics.html"


def generate():
    import CoolProp
    import CoolProp.CoolProp as CP

    if CoolProp.__version__ != VERSION:
        raise ValueError(f"Reference requires CoolProp {VERSION}, got {CoolProp.__version__}")
    gas_constant = 8.31446261815324
    specs = []
    for name, cp0 in [
        ("Methane", 35.7),
        ("Ethane", 52.5),
        ("Propane", 73.6),
        ("CarbonDioxide", 37.0),
    ]:
        state = CP.AbstractState("PR", name)
        specs.append(
            {
                "name": name,
                "critical_temperature_k": state.T_critical(),
                "critical_pressure_pa": state.p_critical(),
                "acentric_factor": state.acentric_factor(),
                "molar_mass_kg_mol": state.molar_mass(),
                "ideal_cp_j_mol_k": cp0,
                "min_temperature_k": max(100.0, 0.55 * state.T_critical()),
                "max_temperature_k": 600.0,
                "reference_temperature_k": 298.15,
                "reference_pressure_pa": 1e5,
            }
        )

    def state_for(indices, z, kij, phase=None):
        state = CP.AbstractState("PR", "&".join(specs[i]["name"] for i in indices))
        if len(indices) > 1:
            state.set_mole_fractions(z)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    state.set_binary_interaction_double(i, j, "kij", kij[i][j])
        if phase is not None:
            state.specify_phase(CP.iphase_liquid if phase == "liquid" else CP.iphase_gas)
        return state

    def properties(state, indices, z):
        p, t = state.p(), state.T()
        cp0 = sum(x * specs[i]["ideal_cp_j_mol_k"] for x, i in zip(z, indices))
        ideal_h = cp0 * (t - 298.15)
        ideal_s = (
            cp0 * math.log(t / 298.15)
            - gas_constant * math.log(p / 1e5)
            - gas_constant * sum(x * math.log(x) for x in z)
        )
        # CoolProp residual entropy is relative to ideal gas at the SAME
        # density. Add R ln(Z) to obtain departure at the same P,T.
        h = ideal_h + state.hmolar_residual()
        s = (
            ideal_s
            + state.smolar_residual()
            + gas_constant * math.log(state.compressibility_factor())
        )
        cp = cp0 + state.cpmolar() - state.cp0molar()
        cv = cp0 - gas_constant + state.cvmolar() - (state.cp0molar() - gas_constant)
        vm = 1 / state.rhomolar()
        vp = -state.isothermal_compressibility() * vm
        vt = state.isobaric_expansion_coefficient() * vm
        sound = math.sqrt(-vm * vm * cp / (state.molar_mass() * cv * vp))
        return {
            "density_kg_m3": state.rhomass(),
            "compressibility_factor": state.compressibility_factor(),
            "molar_volume_m3_mol": vm,
            "molar_enthalpy_j_mol": h,
            "molar_entropy_j_mol_k": s,
            "cp_j_mol_k": cp,
            "cv_j_mol_k": cv,
            "sound_speed_m_s": sound,
            "molar_volume_pressure_derivative": vp,
            "molar_volume_temperature_derivative": vt,
            "log_fugacity_coefficients": [
                math.log(state.fugacity_coefficient(i)) for i in range(len(indices))
            ],
        }

    roots = []
    for i, c in enumerate(specs):
        for p, t, phase in [
            (1e5, 350, "vapor"),
            (8e6, 350, "vapor"),
            (8e6, 0.8 * c["critical_temperature_k"], "liquid"),
        ]:
            state = state_for([i], [1], [[0]], phase)
            state.update(CP.PT_INPUTS, p, t)
            roots.append(
                {
                    "id": f"pure-{i}-{p:g}-{t:g}",
                    "component_indices": [i],
                    "mole_fractions": [1.0],
                    "kij": [[0.0]],
                    "pressure_pa": p,
                    "temperature_k": t,
                    "phase": phase,
                    "expected": properties(state, [i], [1]),
                }
            )
    for indices, z, kij in [
        ([0, 1], [0.5, 0.5], [[0, 0], [0, 0]]),
        ([0, 1], [0.3, 0.7], [[0, 0.03], [0.03, 0]]),
        ([0, 1, 2], [0.3, 0.3, 0.4], [[0, 0.01, 0.02], [0.01, 0, 0.005], [0.02, 0.005, 0]]),
    ]:
        for p, t, phase in [
            (1e5, 350, "vapor"),
            (1e6, 400, "vapor"),
            (8e6, 350, "vapor"),
            (8e6, 220, "liquid"),
        ]:
            state = state_for(indices, z, kij, phase)
            state.update(CP.PT_INPUTS, p, t)
            roots.append(
                {
                    "id": f"mix-{len(roots)}",
                    "component_indices": indices,
                    "mole_fractions": z,
                    "kij": kij,
                    "pressure_pa": p,
                    "temperature_k": t,
                    "phase": phase,
                    "expected": properties(state, indices, z),
                }
            )
    vle = []

    def add_vle(indices, z, kij, p, beta):
        state = state_for(indices, z, kij)
        state.update(CP.PQ_INPUTS, p, beta)
        t = state.T()
        liquid = state.mole_fractions_liquid()
        vapor = state.mole_fractions_vapor()
        phases = []
        for composition, phase in [(liquid, "liquid"), (vapor, "vapor")]:
            branch = state_for(indices, composition, kij, phase)
            branch.update(CP.PT_INPUTS, p, t)
            phases.append(properties(branch, indices, composition))
        # Some CoolProp 7.2 PQ states close fugacity tightly but leave a
        # measurable feed-composition residual. Preserve the requested feed
        # and raw density, and define the comparison state by its independently
        # computed equilibrium phases and the exact material lever rule.
        effective_z = [(1 - beta) * x + beta * y for x, y in zip(liquid, vapor)]
        closure = [actual - requested for actual, requested in zip(effective_z, z)]
        volume = (1 - beta) * phases[0]["molar_volume_m3_mol"] + beta * phases[1][
            "molar_volume_m3_mol"
        ]
        effective_mw = sum(x * specs[i]["molar_mass_kg_mol"] for x, i in zip(effective_z, indices))
        vle.append(
            {
                "id": f"vle-{len(vle)}",
                "component_indices": indices,
                "mole_fractions": effective_z,
                "requested_mole_fractions": z,
                "coolprop_feed_residual": closure,
                "coolprop_native_density_kg_m3": state.rhomass(),
                "kij": kij,
                "pressure_pa": p,
                "temperature_k": t,
                "vapor_molar_fraction": beta,
                "density_kg_m3": effective_mw / volume,
                "liquid_mole_fractions": liquid,
                "vapor_mole_fractions": vapor,
                "phase_properties": phases,
                "molar_enthalpy_j_mol": (1 - beta) * phases[0]["molar_enthalpy_j_mol"]
                + beta * phases[1]["molar_enthalpy_j_mol"],
                "molar_entropy_j_mol_k": (1 - beta) * phases[0]["molar_entropy_j_mol_k"]
                + beta * phases[1]["molar_entropy_j_mol_k"],
            }
        )

    for z in ([0.3, 0.7], [0.5, 0.5]):
        for k in (0.0, -0.03, 0.03):
            for p in (1e6, 3e6):
                for beta in (0.2, 0.8):
                    add_vle([0, 1], z, [[0, k], [k, 0]], p, beta)
    for z in ([0.3, 0.3, 0.4], [0.5, 0.25, 0.25]):
        for p in (3e6, 5e6):
            for beta in (0.2, 0.8):
                add_vle(
                    [0, 1, 2], z, [[0, 0.01, 0.02], [0.01, 0, 0.005], [0.02, 0.005, 0]], p, beta
                )
    return {
        "schema": 1,
        "source": SOURCE,
        "generator_version": VERSION,
        "coolprop_git_revision": CoolProp.__gitrevision__,
        "backend": "PR",
        "reference_omega_a": 0.45724,
        "reference_omega_b": 0.07780,
        "gas_constant_j_mol_k": gas_constant,
        "convention": "Constant user Cp0 added to CoolProp residual calorics; entropy departure at equal P,T = smolar_residual + R ln Z; homogeneous roots are not VLE stability proofs.",
        "components": specs,
        "homogeneous_cases": roots,
        "vle_cases": vle,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
