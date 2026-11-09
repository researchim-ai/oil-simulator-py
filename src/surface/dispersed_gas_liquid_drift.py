"""Dilute creeping liquid droplets carried by gas, with finite surface energy.

Stokes or Hadamard--Rybczynski settling, mass-barycentric phase velocities.
No gas drift fit or oil/water carrier is reused. Entrainment from a wall film
is not inferred from this transport law; explicit finite injection/capture
volumes are required at morphology boundaries.
"""

import math
from dataclasses import dataclass, fields

import torch

from .dispersed_liquid_drift import DispersedOilWaterClosure
from .liquid import _number
from .peng_robinson import R


@dataclass(frozen=True, kw_only=True)
class DispersedGasLiquidClosure(DispersedOilWaterClosure):
    gas_viscosity_pa_s: float
    maximum_knudsen_number: float = 0.01
    minimum_gas_volume_fraction: float = 0.9
    maximum_gas_volume_fraction: float = 1.0
    maximum_gas_liquid_density_ratio: float = 0.2
    model: str = "dispersed_gas_liquid"

    def __post_init__(self):
        base = {f.name: getattr(self, f.name) for f in fields(DispersedOilWaterClosure)}
        base.update(
            model="dispersed_oil_water",
            minimum_gas_volume_fraction=1e-8,
            maximum_gas_volume_fraction=0.7,
        )
        DispersedOilWaterClosure(**base)
        _number(self.gas_viscosity_pa_s, "gas viscosity", positive=True)
        if not 0 < self.maximum_knudsen_number <= 0.01:
            raise ValueError("Continuum mist requires Kn <= .01")
        if (
            self.model != "dispersed_gas_liquid"
            or self.distribution_parameter != 1.0
            or self.drift_velocity_m_s != 0.0
        ):
            raise ValueError("Gas mist uses its own settling law without a second gas drift fit")
        if not 0.9 <= self.minimum_gas_volume_fraction < self.maximum_gas_volume_fraction <= 1.0:
            raise ValueError("Dilute gas carrier requires gas fraction >=.9")

    @property
    def population_carrier(self):
        return "gas"

    def population_masks(self, roles):
        gas, oil, water = roles
        return (oil if self.dispersed_liquid == "oil" else water), gas

    def population_viscosities(self):
        return self.gas_viscosity_pa_s, (
            self.oil_viscosity_pa_s if self.dispersed_liquid == "oil" else self.water_viscosity_pa_s
        )

    def population_slip(self, roles, densities, axial_gravity, gravity, diameters):
        drop, gas = self.population_masks(roles)
        rg, rd = (densities * gas).sum(), (densities * drop).sum()
        mu, md = self.population_viscosities()
        mobility = diameters**2 / (18 * mu)
        if self.interface == "clean":
            mobility = mobility * 3 * (mu + md) / (2 * mu + 3 * md)
        contrast = rd - rg
        slip = -contrast * axial_gravity * mobility
        speed = contrast.abs() * gravity * mobility
        re = rg * speed * diameters / mu
        eo = contrast.abs() * gravity * diameters**2 / self.surface_tension_n_m
        return slip, re, eo

    def pressure_valid(self, composition, pressure, droplet_diameter_m=None, phase_roles=None):
        roles, _ = self.roles(composition) if phase_roles is None else phase_roles
        drop, gas = self.population_masks(roles)
        present = (drop.sum() > 0) & (gas.sum() > 0)
        d = self.droplet_diameter_m if droplet_diameter_m is None else droplet_diameter_m
        return (~present) | (
            4 * self.surface_tension_n_m / d <= self.maximum_capillary_pressure_fraction * pressure
        )

    def continuum_valid(
        self, composition, densities, temperature, molar_masses, diameters, phase_roles=None
    ):
        roles, _ = self.roles(composition) if phase_roles is None else phase_roles
        drop, gas = self.population_masks(roles)
        present = (drop.sum() > 0) & (gas.sum() > 0)
        mass = ((composition @ molar_masses) * gas).sum().clamp_min(1e-300)
        rho = (densities * gas).sum().clamp_min(1e-300)
        mean_free_path = (
            self.gas_viscosity_pa_s / rho * torch.sqrt(math.pi * mass / (2 * R * temperature))
        )
        return torch.where(present, mean_free_path / diameters <= self.maximum_knudsen_number, True)

    def velocities(
        self,
        phase_volumes,
        phase_densities,
        bulk_velocity,
        *,
        diameter_m,
        axial_gravity_m_s2,
        phase_mole_fractions=None,
        gravity_m_s2=None,
        droplet_diameter_m=None,
        phase_roles=None,
    ):
        roles, valid = self.roles(phase_mole_fractions) if phase_roles is None else phase_roles
        drop, gas = self.population_masks(roles)
        other = roles[2] if self.dispersed_liquid == "oil" else roles[1]
        alpha = phase_volumes / phase_volumes.sum()
        ad, ag = (alpha * drop).sum(), (alpha * gas).sum()
        rg, rd = (phase_densities * gas).sum(), (phase_densities * drop).sum()
        d = self.droplet_diameter_m if droplet_diameter_m is None else droplet_diameter_m
        gravity = abs(axial_gravity_m_s2) if gravity_m_s2 is None else gravity_m_s2
        slip, re, eo = self.population_slip(roles, phase_densities, axial_gravity_m_s2, gravity, d)
        present = drop.sum() > 0
        slip = torch.where(present, slip, slip * 0)
        fraction = (alpha * phase_densities * drop).sum() / (alpha * phase_densities).sum()
        vg = bulk_velocity - fraction * slip
        values = vg + drop * slip
        domain = (
            (ag >= self.minimum_gas_volume_fraction)
            & (ag <= self.maximum_gas_volume_fraction)
            & (ad <= self.maximum_dispersed_liquid_fraction)
            & (other.sum() == 0)
            & (gas.sum() == 1)
        )
        domain = domain & torch.where(
            present,
            (rd > rg)
            & (rg <= self.maximum_gas_liquid_density_ratio * rd)
            & (re <= self.maximum_reynolds_number)
            & (eo <= self.maximum_eotvos_number)
            & (d / diameter_m <= self.maximum_confinement_ratio),
            True,
        )
        return values, valid & domain & torch.isfinite(values).all()
