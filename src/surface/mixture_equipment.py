"""Adiabatic equilibrium letdown with explicit multicomponent molar accounting."""

from dataclasses import dataclass

from .liquid import ConservationError, PropertyModelError, _name, _number
from .peng_robinson_mixture import PengRobinsonMixture


@dataclass(frozen=True)
class MixtureLetdown:
    name: str
    outlet_pressure_pa: float

    def __post_init__(self):
        _name(self.name)
        _number(self.outlet_pressure_pa, "outlet_pressure_pa", positive=True)

    def run(self, inlet_pressure_pa, inlet_temperature_k, mass_rate_kg_s, mole_fractions, fluid):
        """Bulk equilibrium PH outlet, no sizing/critical-flow or throat model."""
        if not isinstance(fluid, PengRobinsonMixture):
            raise TypeError("MixtureLetdown requires explicit PengRobinsonMixture")
        _number(mass_rate_kg_s, "mass_rate_kg_s", nonnegative=True)
        if self.outlet_pressure_pa > inlet_pressure_pa:
            raise PropertyModelError("Letdown cannot raise pressure")
        z = fluid._composition(mole_fractions)
        inlet = fluid.flash_pt(inlet_pressure_pa, inlet_temperature_k, z)
        if mass_rate_kg_s == 0:
            return {
                "inlet": inlet,
                "outlet": None,
                "mass_rate_kg_s": 0.0,
                "energy_error_w": 0.0,
                "entropy_generation_w_k": 0.0,
                "component_mass_rates_kg_s": [0.0] * len(z),
            }
        outlet = (
            inlet
            if self.outlet_pressure_pa == inlet_pressure_pa
            else fluid.flash_ph(self.outlet_pressure_pa, inlet["molar_enthalpy_j_mol"], z)
        )
        molar_mass = sum(x * c.molar_mass_kg_mol for x, c in zip(z, fluid.components))
        molar_rate = mass_rate_kg_s / molar_mass
        energy = molar_rate * (outlet["molar_enthalpy_j_mol"] - inlet["molar_enthalpy_j_mol"])
        entropy = molar_rate * (outlet["molar_entropy_j_mol_k"] - inlet["molar_entropy_j_mol_k"])
        if entropy < -1e-6:
            raise ConservationError("Adiabatic letdown decreases entropy")
        return {
            "inlet": inlet,
            "outlet": outlet,
            "mass_rate_kg_s": mass_rate_kg_s,
            "component_mass_rates_kg_s": [
                molar_rate * x * c.molar_mass_kg_mol for x, c in zip(z, fluid.components)
            ],
            "energy_error_w": energy,
            "entropy_generation_w_k": entropy,
        }
