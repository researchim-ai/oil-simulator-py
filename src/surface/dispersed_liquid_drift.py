"""Dilute oil/water creeping-drop slip plus an explicit gas drift law.

Phase roles use configured composition markers, never density ordering alone.
Stokes (immobile interface) and Hadamard--Rybczynski (clean interface) are
restricted to spherical, weakly confined, dilute droplets at small Reynolds
number. Drop size is prescribed; this is not population-balance kinetics.
"""

from dataclasses import dataclass

import torch

from .drift_flux_closure import ZuberFindlayClosure
from .liquid import _number


@dataclass(frozen=True)
class DispersedOilWaterClosure:
    gas_component_index: int
    oil_component_index: int
    water_component_index: int
    droplet_diameter_m: float
    oil_viscosity_pa_s: float
    water_viscosity_pa_s: float
    surface_tension_n_m: float
    dispersed_liquid: str = "oil"
    interface: str = "immobile"
    distribution_parameter: float = 1.0
    drift_velocity_m_s: float = 0.0
    minimum_marker_fraction: float = 0.5
    maximum_dispersed_liquid_fraction: float = 0.1
    maximum_reynolds_number: float = 0.5
    maximum_eotvos_number: float = 0.1
    maximum_confinement_ratio: float = 0.05
    maximum_capillary_pressure_fraction: float = 0.02
    minimum_gas_volume_fraction: float = 1e-8
    maximum_gas_volume_fraction: float = 0.7
    maximum_gas_liquid_density_ratio: float = 0.5
    model: str = "dispersed_oil_water"

    def __post_init__(self):
        indices = (self.gas_component_index, self.oil_component_index, self.water_component_index)
        if len(set(indices)) != 3 or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("Three distinct nonnegative component marker indices required")
        if self.model != "dispersed_oil_water" or self.interface not in ("immobile", "clean"):
            raise ValueError("Explicit immobile or clean droplet interface required")
        if self.dispersed_liquid not in ("oil", "water"):
            raise ValueError("Explicit oil-in-water or water-in-oil dispersion required")
        for key in (
            "droplet_diameter_m",
            "oil_viscosity_pa_s",
            "water_viscosity_pa_s",
            "surface_tension_n_m",
        ):
            _number(getattr(self, key), key, positive=True)
        for key, upper in (
            ("maximum_dispersed_liquid_fraction", 0.1),
            ("maximum_reynolds_number", 0.5),
            ("maximum_eotvos_number", 0.1),
            ("maximum_confinement_ratio", 0.05),
            ("maximum_capillary_pressure_fraction", 0.05),
        ):
            value = _number(getattr(self, key), key, positive=True)
            if value > upper:
                raise ValueError(f"{key} exceeds the dilute creeping-drop domain")
        if not 0.5 <= self.minimum_marker_fraction <= 1:
            raise ValueError("Composition markers require a dominant fraction >= 0.5")
        ZuberFindlayClosure(
            self.distribution_parameter,
            self.drift_velocity_m_s,
            self.minimum_gas_volume_fraction,
            self.maximum_gas_volume_fraction,
            self.maximum_gas_liquid_density_ratio,
        )

    def roles(self, x):
        scores = x[
            :, [self.gas_component_index, self.oil_component_index, self.water_component_index]
        ]
        role = scores.argmax(-1)
        masks = torch.nn.functional.one_hot(role, 3).to(x.dtype)
        valid = (scores.max(-1).values >= self.minimum_marker_fraction).all() & (
            masks.sum(0) <= 1
        ).all()
        return masks.unbind(-1), valid

    def pressure_valid(self, composition, pressure):
        if len(composition) == 1:
            return pressure.new_ones((), dtype=torch.bool)
        (_gas, oil, water), _ = self.roles(composition)
        both = (oil.sum() > 0) & (water.sum() > 0)
        laplace_pressure = 4 * self.surface_tension_n_m / self.droplet_diameter_m
        return (~both) | (laplace_pressure <= self.maximum_capillary_pressure_fraction * pressure)

    def liquid_slip(self, oil_density, water_density, axial_gravity_m_s2, gravity_m_s2=None):
        contrast = water_density - oil_density
        gravity = abs(axial_gravity_m_s2) if gravity_m_s2 is None else gravity_m_s2
        if self.dispersed_liquid == "oil":
            carrier_mu, drop_mu, carrier_density = (
                self.water_viscosity_pa_s,
                self.oil_viscosity_pa_s,
                water_density,
            )
        else:
            carrier_mu, drop_mu, carrier_density = (
                self.oil_viscosity_pa_s,
                self.water_viscosity_pa_s,
                oil_density,
            )
        mobility = self.droplet_diameter_m**2 / (18 * carrier_mu)
        if self.interface == "clean":
            mobility = mobility * (3 * (carrier_mu + drop_mu) / (2 * carrier_mu + 3 * drop_mu))
        velocity = contrast * axial_gravity_m_s2 * mobility
        terminal_speed = contrast.abs() * gravity * mobility
        reynolds = carrier_density * terminal_speed * self.droplet_diameter_m / carrier_mu
        eotvos = contrast.abs() * gravity * self.droplet_diameter_m**2 / self.surface_tension_n_m
        return velocity, reynolds, eotvos

    def velocities(
        self,
        phase_volumes,
        phase_densities,
        bulk_velocity,
        *,
        phase_mole_fractions,
        diameter_m,
        axial_gravity_m_s2,
        gravity_m_s2=None,
    ):
        if len(phase_volumes) == 1:
            return bulk_velocity.expand_as(phase_volumes), bulk_velocity.new_ones(
                (), dtype=torch.bool
            )
        (gas, oil, water), valid = self.roles(phase_mole_fractions)
        alpha = phase_volumes / phase_volumes.sum()
        ag, ao, aw = [(alpha * role).sum() for role in (gas, oil, water)]
        rg, ro, rw = [(phase_densities * role).sum() for role in (gas, oil, water)]
        both_liquids = (oil.sum() > 0) & (water.sum() > 0)
        slip, re, eo = self.liquid_slip(ro, rw, axial_gravity_m_s2, gravity_m_s2)
        slip = torch.where(both_liquids, slip, slip * 0)
        al = (1 - ag).clamp_min(1e-30)
        # Zero-volume-average liquid bias retains j = alpha_g*v_g+alpha_l*v_l.
        bias = (oil * aw - water * ao) * slip / al
        mean = (alpha * phase_densities).sum()
        liquid_density = ((alpha * phase_densities * (1 - gas)).sum() / al).clamp_min(
            phase_densities.min() * 1e-12
        )
        bias_momentum = (alpha * phase_densities * bias).sum()
        c0, vd = self.distribution_parameter, self.drift_velocity_m_s
        denominator = liquid_density + ag * c0 * (rg - liquid_density)
        safe_denominator = torch.where(
            denominator.abs() > 1e-20,
            denominator,
            torch.where(
                denominator < 0, denominator.new_tensor(-1e-20), denominator.new_tensor(1e-20)
            ),
        )
        j = (
            mean * bulk_velocity - bias_momentum - ag * (rg - liquid_density) * vd
        ) / safe_denominator
        vg = c0 * j + vd
        vl = (mean * bulk_velocity - bias_momentum - ag * rg * vg) / (al * liquid_density)
        values = gas * vg + (1 - gas) * vl + bias
        gas_valid = (
            (ag >= self.minimum_gas_volume_fraction)
            & (ag <= self.maximum_gas_volume_fraction)
            & (rg / liquid_density <= self.maximum_gas_liquid_density_ratio)
        )
        liquid_valid = (
            (rw > 0)
            & (ro > 0)
            & (
                (ao if self.dispersed_liquid == "oil" else aw) / al
                <= self.maximum_dispersed_liquid_fraction
            )
            & (re <= self.maximum_reynolds_number)
            & (eo <= self.maximum_eotvos_number)
            & (self.droplet_diameter_m / diameter_m <= self.maximum_confinement_ratio)
        )
        valid = (
            valid
            & (len(phase_volumes) <= 3)
            & (denominator > 0)
            & torch.where(ag > 0, gas_valid, True)
            & torch.where(both_liquids, liquid_valid, True)
            & torch.isfinite(values).all()
        )
        return values, valid
