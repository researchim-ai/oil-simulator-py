"""Explicit Zuber--Findlay gas/liquid closure with a declared validity envelope.

v_g = C0*j + Vgj. C0 and axial Vgj must be supplied for the intended regime.
No hidden regime classification or extrapolation beyond the specified envelope.
The non-gas phases share one liquid velocity (dispersed liquid mixture).
"""

from dataclasses import dataclass

import torch

from .liquid import _number


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

    def velocities(self, phase_volumes, phase_densities, bulk_velocity):
        """Return velocities and admissibility without any host synchronization.

        The lightest phase is explicitly treated as gas. Use this closure only
        for gas/liquid or gas/dispersed-oil-water states, not two liquid phases.
        """
        if len(phase_volumes) == 1:
            return bulk_velocity.expand_as(phase_volumes), bulk_velocity.new_ones(
                (), dtype=torch.bool
            )
        alpha = phase_volumes / phase_volumes.sum()
        gas = torch.nn.functional.one_hot(phase_densities.argmin(), len(alpha)).to(alpha.dtype)
        liquid = 1 - gas
        ag = (alpha * gas).sum()
        rg = (phase_densities * gas).sum()
        liquid_volume = (alpha * liquid).sum()
        rl = (alpha * phase_densities * liquid).sum() / liquid_volume
        mean = (alpha * phase_densities).sum()
        c0 = self.distribution_parameter
        vd = self.drift_velocity_m_s
        denominator = rl + ag * c0 * (rg - rl)
        # Solve BOTH the drift law and the specified mass-average momentum.
        j = (mean * bulk_velocity - ag * (rg - rl) * vd) / denominator
        vg = c0 * j + vd
        vl = (mean * bulk_velocity - ag * rg * vg) / (liquid_volume * rl)
        values = gas * vg + liquid * vl
        valid = (
            torch.isfinite(values).all()
            & (denominator > 0)
            & (ag >= self.minimum_gas_volume_fraction)
            & (ag <= self.maximum_gas_volume_fraction)
            & (rg / rl <= self.maximum_gas_liquid_density_ratio)
        )
        return values, valid
