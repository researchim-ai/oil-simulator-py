"""Explicit Zuber--Findlay gas/liquid closure with a declared validity envelope.

v_g = C0*j + Vgj. C0 and axial Vgj must be supplied for the intended regime.
No hidden regime classification or extrapolation beyond the specified envelope.
The non-gas phases share one liquid velocity (dispersed liquid mixture).
"""

from dataclasses import dataclass

import torch

from .liquid import _number


def _mass_average_velocities(volumes, densities, velocity, c0, vd, envelope):
    alpha = volumes / volumes.sum()
    gas = torch.nn.functional.one_hot(densities.argmin(), len(alpha)).to(alpha.dtype)
    liquid = 1 - gas
    ag = (alpha * gas).sum()
    rg = (densities * gas).sum()
    liquid_volume = (alpha * liquid).sum()
    rl = (alpha * densities * liquid).sum() / liquid_volume
    mean = (alpha * densities).sum()
    denominator = rl + ag * c0 * (rg - rl)
    j = (mean * velocity - ag * (rg - rl) * vd) / denominator
    vg = c0 * j + vd
    vl = (mean * velocity - ag * rg * vg) / (liquid_volume * rl)
    values = gas * vg + liquid * vl
    valid = (
        torch.isfinite(values).all()
        & (denominator > 0)
        & (ag >= envelope.minimum_gas_volume_fraction)
        & (ag <= envelope.maximum_gas_volume_fraction)
        & (rg / rl <= envelope.maximum_gas_liquid_density_ratio)
    )
    return values, valid


@dataclass(frozen=True)
class ZuberFindlayClosure:
    distribution_parameter: float
    drift_velocity_m_s: float
    minimum_gas_volume_fraction: float = 1e-8
    maximum_gas_volume_fraction: float = 0.7
    maximum_gas_liquid_density_ratio: float = 0.5

    def __post_init__(self):
        _number(self.distribution_parameter, "distribution parameter", positive=True)
        _number(self.drift_velocity_m_s, "axial drift velocity")
        if not 0 < self.minimum_gas_volume_fraction < self.maximum_gas_volume_fraction < 1:
            raise ValueError("Strict interior gas-fraction envelope required")
        if not 0 < self.maximum_gas_liquid_density_ratio < 1:
            raise ValueError("Density contrast envelope required")

    def velocities(self, phase_volumes, phase_densities, bulk_velocity, **context):
        """Return velocities and admissibility without any host synchronization.

        The lightest phase is explicitly treated as gas. Use this closure only
        for gas/liquid or gas/dispersed-oil-water states, not two liquid phases.
        """
        if len(phase_volumes) == 1:
            return bulk_velocity.expand_as(phase_volumes), bulk_velocity.new_ones(
                (), dtype=torch.bool
            )
        return _mass_average_velocities(
            phase_volumes,
            phase_densities,
            bulk_velocity,
            self.distribution_parameter,
            self.drift_velocity_m_s,
            self,
        )


@dataclass(frozen=True)
class RELAP7VerticalDriftClosure:
    """Explicit vertical gas/liquid regime, RELAP-7 theory eqs. 585--590.

    No automatic regime map or oil/water extension. Surface tension and
    liquid viscosity are prescribed constitutive inputs; EOS densities remain
    live Newton variables. Gravity direction is supplied by the pipe geometry.
    """

    regime: str
    surface_tension_n_m: float
    liquid_viscosity_pa_s: float
    minimum_gas_volume_fraction: float = 1e-8
    maximum_gas_volume_fraction: float = 0.2
    maximum_gas_liquid_density_ratio: float = 0.5
    model: str = "relap7_vertical_pipe"

    def __post_init__(self):
        if self.model != "relap7_vertical_pipe" or self.regime not in (
            "dispersed_bubbly",
            "cap_slug",
        ):
            raise ValueError("Explicit RELAP-7 dispersed_bubbly or cap_slug regime required")
        _number(self.surface_tension_n_m, "surface tension", positive=True)
        _number(self.liquid_viscosity_pa_s, "liquid viscosity", positive=True)
        ZuberFindlayClosure(
            1.0,
            0.0,
            self.minimum_gas_volume_fraction,
            self.maximum_gas_volume_fraction,
            self.maximum_gas_liquid_density_ratio,
        )
        if self.regime == "dispersed_bubbly" and self.maximum_gas_volume_fraction > 0.2:
            raise ValueError("Dispersed bubbly closure requires gas fraction <= 0.2")

    def parameters(
        self,
        phase_densities,
        *,
        diameter_m,
        axial_gravity_m_s2,
        phase_mole_fractions=None,
        gravity_m_s2=None,
    ):
        rg, rl = phase_densities.min(), phase_densities.max()
        gravity = abs(axial_gravity_m_s2)
        if gravity <= 0 or diameter_m <= 0:
            raise ValueError("Positive diameter and nonzero axial gravity required")
        contrast = (rl - rg).clamp_min(1e-12 * rl)
        capillary_length = torch.sqrt(self.surface_tension_n_m / (gravity * contrast))
        scale = (self.surface_tension_n_m * gravity * contrast / rl.square()).pow(0.25)
        c0 = 1.2 - 0.2 * torch.sqrt(rg / rl)
        if self.regime == "dispersed_bubbly":
            vd = 2**0.5 * scale
        else:
            dimensionless_diameter = (diameter_m / capillary_length).clamp_max(30.0)
            viscosity_number = self.liquid_viscosity_pa_s / torch.sqrt(
                rl * self.surface_tension_n_m * capillary_length
            )
            vd = (
                0.0019
                * dimensionless_diameter.pow(0.809)
                * (rg / rl).pow(-0.157)
                * viscosity_number.pow(-0.562)
                * scale
            )
        return c0, vd if axial_gravity_m_s2 > 0 else -vd

    def velocities(self, phase_volumes, phase_densities, bulk_velocity, **context):
        if len(phase_volumes) == 1:
            return bulk_velocity.expand_as(phase_volumes), bulk_velocity.new_ones(
                (), dtype=torch.bool
            )
        c0, vd = self.parameters(phase_densities, **context)
        values, valid = _mass_average_velocities(
            phase_volumes, phase_densities, bulk_velocity, c0, vd, self
        )
        return values, valid & (len(phase_volumes) == 2)


def drift_closure_from_dict(data):
    if data.get("model") == "concentrated_oil_water":
        from .concentrated_liquid_drift import ConcentratedOilWaterClosure

        return ConcentratedOilWaterClosure(**data)
    if data.get("model") == "relap7_vertical_transition":
        return RELAP7EquilibriumTransition(**data)
    if data.get("model") == "dispersed_oil_water":
        from .dispersed_liquid_drift import DispersedOilWaterClosure

        return DispersedOilWaterClosure(**data)
    if data.get("model") == "relap7_vertical_pipe":
        return RELAP7VerticalDriftClosure(**data)
    return ZuberFindlayClosure(**data)


@dataclass(frozen=True)
class RELAP7EquilibriumTransition:
    """RELAP-7 closures (2017), eqs 67--76, for the local-equilibrium limit.

    Interface and liquid temperatures coincide, so alpha_DB=0 and
    alpha_CS=0.1. No guessed 0.2--0.3 band or extrapolation to annular flow.
    """

    surface_tension_n_m: float
    liquid_viscosity_pa_s: float
    minimum_gas_volume_fraction: float = 1e-8
    maximum_gas_volume_fraction: float = 0.7
    maximum_gas_liquid_density_ratio: float = 0.5
    model: str = "relap7_vertical_transition"

    def __post_init__(self):
        if self.model != "relap7_vertical_transition" or self.maximum_gas_volume_fraction > 0.7:
            raise ValueError("Vertical bubbly/slug transition excludes annular/mist flow")
        ZuberFindlayClosure(
            1.0,
            0.0,
            self.minimum_gas_volume_fraction,
            self.maximum_gas_volume_fraction,
            self.maximum_gas_liquid_density_ratio,
        )
        object.__setattr__(
            self,
            "_bubbly",
            RELAP7VerticalDriftClosure(
                "dispersed_bubbly", self.surface_tension_n_m, self.liquid_viscosity_pa_s
            ),
        )
        object.__setattr__(
            self,
            "_slug",
            RELAP7VerticalDriftClosure(
                "cap_slug", self.surface_tension_n_m, self.liquid_viscosity_pa_s
            ),
        )

    def parameters(self, phase_volumes, phase_densities, **context):
        c0, db = self._bubbly.parameters(phase_densities, **context)
        _, cs = self._slug.parameters(phase_densities, **context)
        alpha = phase_volumes / phase_volumes.sum()
        gas = torch.nn.functional.one_hot(phase_densities.argmin(), len(alpha)).to(alpha.dtype)
        ag = (alpha * gas).sum()
        weight = (1 - ag / 0.1).clamp(0.0, 1.0)
        return c0, weight * db + (1 - weight) * cs, weight

    def velocities(self, phase_volumes, phase_densities, bulk_velocity, **context):
        if len(phase_volumes) == 1:
            return bulk_velocity.expand_as(phase_volumes), bulk_velocity.new_ones(
                (), dtype=torch.bool
            )
        c0, vd, _ = self.parameters(phase_volumes, phase_densities, **context)
        values, valid = _mass_average_velocities(
            phase_volumes, phase_densities, bulk_velocity, c0, vd, self
        )
        return values, valid & (len(phase_volumes) == 2)
