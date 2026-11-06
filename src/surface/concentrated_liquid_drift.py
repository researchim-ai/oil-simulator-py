"""Explicit creeping, immobile-drop dense dispersion and morphology blend.

Richardson--Zaki settling and Krieger--Dougherty wall viscosity are distinct
constitutive inputs. The carrier transition is supplied by the user; it is
not a thermodynamic prediction of inversion or a deformable-emulsion model.
"""

from dataclasses import dataclass, fields

import torch

from .dispersed_liquid_drift import DispersedOilWaterClosure
from .liquid import _number


@dataclass(frozen=True, kw_only=True)
class ConcentratedOilWaterClosure(DispersedOilWaterClosure):
    phase_inversion_oil_fraction: float
    transition_half_width: float
    packing_fraction: float
    intrinsic_viscosity: float
    hindered_settling_exponent: float
    gas_viscosity_pa_s: float
    dispersed_liquid: str = "morphology_curve"
    maximum_dispersed_liquid_fraction: float = 0.58
    maximum_reynolds_number: float = 0.2
    model: str = "concentrated_oil_water"

    def __post_init__(self):
        inherited = {f.name: getattr(self, f.name) for f in fields(DispersedOilWaterClosure)}
        inherited.update(
            model="dispersed_oil_water",
            maximum_dispersed_liquid_fraction=0.1,
            dispersed_liquid="oil",
        )
        DispersedOilWaterClosure(**inherited)
        if (
            self.model != "concentrated_oil_water"
            or self.interface != "immobile"
            or self.dispersed_liquid != "morphology_curve"
        ):
            raise ValueError("Dense creeping model requires immobile spherical drops")
        for key in (
            "phase_inversion_oil_fraction",
            "transition_half_width",
            "packing_fraction",
            "intrinsic_viscosity",
            "hindered_settling_exponent",
            "gas_viscosity_pa_s",
        ):
            _number(getattr(self, key), key, positive=True)
        if not 0.5 < self.maximum_dispersed_liquid_fraction < self.packing_fraction < 1:
            raise ValueError("Dense fraction must remain below declared packing fraction")
        if self.intrinsic_viscosity > 10:
            raise ValueError("Intrinsic viscosity outside declared spherical-drop model")
        if self.maximum_reynolds_number > 0.2 or self.hindered_settling_exponent < 1:
            raise ValueError("Dense settling requires creeping isolated-drop Reynolds numbers")
        lo = self.phase_inversion_oil_fraction - self.transition_half_width
        hi = self.phase_inversion_oil_fraction + self.transition_half_width
        if (
            not 1 - self.maximum_dispersed_liquid_fraction
            <= lo
            < hi
            <= self.maximum_dispersed_liquid_fraction
        ):
            raise ValueError("Carrier transition must lie inside both morphology envelopes")

    def rheology(self, phase_volumes, composition, phase_roles=None):
        (gas, oil, water), valid = self.roles(composition) if phase_roles is None else phase_roles
        alpha = phase_volumes / phase_volumes.sum()
        ag, ao, aw = [(alpha * r).sum() for r in (gas, oil, water)]
        liquid = (ao + aw).clamp_min(1e-30)
        fraction = ao / liquid
        r = (
            (fraction - self.phase_inversion_oil_fraction + self.transition_half_width)
            / (2 * self.transition_half_width)
        ).clamp(0.0, 1.0)
        weight = r.square() * (3 - 2 * r)

        # Inactive branches stay finite; active packing limits are checked below.
        def viscosity(mu, dispersed):
            reduced = (dispersed / self.packing_fraction).clamp(max=1 - 1e-12)
            return mu * torch.exp(
                -self.intrinsic_viscosity * self.packing_fraction * torch.log1p(-reduced)
            )

        ow = viscosity(self.water_viscosity_pa_s, fraction)
        wo = viscosity(self.oil_viscosity_pa_s, 1 - fraction)
        mixed = torch.exp((1 - weight) * ow.log() + weight * wo.log())
        both = (oil.sum() > 0) & (water.sum() > 0)
        plain = (ao * self.oil_viscosity_pa_s + aw * self.water_viscosity_pa_s) / liquid
        liquid_mu = torch.where(both, mixed, plain)
        wall_mu = liquid * liquid_mu + ag * self.gas_viscosity_pa_s
        valid = valid & torch.where(
            both,
            ((weight >= 1) | (fraction <= self.maximum_dispersed_liquid_fraction))
            & ((weight <= 0) | ((1 - fraction) <= self.maximum_dispersed_liquid_fraction)),
            True,
        )
        valid = valid & torch.isfinite(wall_mu) & (wall_mu > 0)
        return {
            "viscosity_pa_s": wall_mu,
            "oil_continuous_weight": weight,
            "oil_liquid_fraction": fraction,
            "valid": valid,
            "roles": (gas, oil, water),
        }

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
        **context,
    ):
        props = self.rheology(phase_volumes, phase_mole_fractions, context.get("phase_roles"))
        if len(phase_volumes) == 1:
            return bulk_velocity.expand_as(phase_volumes), props["valid"]
        gas, oil, water = props["roles"]
        alpha = phase_volumes / phase_volumes.sum()
        ag, ao, aw = [(alpha * r).sum() for r in (gas, oil, water)]
        rg, ro, rw = [(phase_densities * r).sum() for r in (gas, oil, water)]
        both = (oil.sum() > 0) & (water.sum() > 0)
        phi = props["oil_liquid_fraction"]
        weight = props["oil_continuous_weight"]
        diameter = self.droplet_diameter_m
        contrast = rw - ro
        mobility = diameter**2 / 18
        # R-Z gives settling relative to the volume frame. Oil-water relative
        # speed has exponent n-1, since v_drop=(1-phi)*v_relative in that frame.
        ow = (
            mobility
            / self.water_viscosity_pa_s
            * (1 - phi).clamp_min(1e-30).pow(self.hindered_settling_exponent - 1)
        )
        wo = (
            mobility
            / self.oil_viscosity_pa_s
            * phi.clamp_min(1e-30).pow(self.hindered_settling_exponent - 1)
        )
        slip = contrast * axial_gravity_m_s2 * ((1 - weight) * ow + weight * wo)
        slip = torch.where(both, slip, slip * 0)
        al = (1 - ag).clamp_min(1e-30)
        bias = (oil * aw - water * ao) * slip / al
        mean = (alpha * phase_densities).sum()
        rl = ((alpha * phase_densities * (1 - gas)).sum() / al).clamp_min(1e-20)
        c0, vd = self.distribution_parameter, self.drift_velocity_m_s
        denominator = rl + ag * c0 * (rg - rl)
        safe = denominator.clamp_min(1e-20)
        bm = (alpha * phase_densities * bias).sum()
        j = (mean * bulk_velocity - bm - ag * (rg - rl) * vd) / safe
        vg = c0 * j + vd
        vl = (mean * bulk_velocity - bm - ag * rg * vg) / (al * rl)
        values = gas * vg + (1 - gas) * vl + bias
        gravity = abs(axial_gravity_m_s2) if gravity_m_s2 is None else gravity_m_s2
        isolated = contrast.abs() * gravity * mobility
        re_ow = rw * isolated * diameter / self.water_viscosity_pa_s**2
        re_wo = ro * isolated * diameter / self.oil_viscosity_pa_s**2
        re_valid = ((weight >= 1) | (re_ow <= self.maximum_reynolds_number)) & (
            (weight <= 0) | (re_wo <= self.maximum_reynolds_number)
        )
        eo = contrast.abs() * gravity * diameter**2 / self.surface_tension_n_m
        liquid_valid = (
            (ro > 0)
            & (rw > 0)
            & re_valid
            & (eo <= self.maximum_eotvos_number)
            & (diameter / diameter_m <= self.maximum_confinement_ratio)
        )
        gas_valid = (
            (ag >= self.minimum_gas_volume_fraction)
            & (ag <= self.maximum_gas_volume_fraction)
            & (rg / rl <= self.maximum_gas_liquid_density_ratio)
        )
        valid = (
            props["valid"]
            & (denominator > 0)
            & torch.where(both, liquid_valid, True)
            & torch.where(ag > 0, gas_valid, True)
            & torch.isfinite(values).all()
        )
        return values, valid
