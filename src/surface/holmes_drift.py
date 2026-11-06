"""Continuous Holmes gas/liquid drift (Díaz 2004, equations 3--17, 29).

The correlation uses volume flux j, not the mass-average momentum velocity.
A device-local scalar inversion preserves that distinction and its derivatives.
No transverse stratified interface or experimental calibration is implied.
"""

import math
from dataclasses import dataclass

import torch

from .liquid import _number


@dataclass(frozen=True, kw_only=True)
class HolmesGasLiquidClosure:
    surface_tension_n_m: float
    profile_parameter: float
    profile_flattening_start: float
    velocity_sensitivity: float
    bubble_transition_start: float
    bubble_transition_end: float
    minimum_gas_volume_fraction: float = 1e-6
    maximum_gas_volume_fraction: float = 0.9999
    maximum_gas_liquid_density_ratio: float = 0.5
    maximum_deviation_deg: float = 80.0
    minimum_dimensionless_diameter: float = 4.0
    maximum_dimensionless_diameter: float = 200.0
    model: str = "holmes_gas_liquid"

    def __post_init__(self):
        if self.model != "holmes_gas_liquid":
            raise ValueError("Unknown Holmes model")
        for key in ("surface_tension_n_m", "velocity_sensitivity"):
            _number(getattr(self, key), key, positive=True)
        if not 1 <= self.profile_parameter <= 1.5 or not 0 <= self.profile_flattening_start <= 0.3:
            raise ValueError("Profile parameters outside monotone inversion envelope")
        if not 0 <= self.bubble_transition_start < self.bubble_transition_end < 1:
            raise ValueError("Ordered interior bubbly/film transition required")
        if not 0 < self.minimum_gas_volume_fraction < self.maximum_gas_volume_fraction < 1:
            raise ValueError("Interior gas fraction envelope required")
        if not 0 < self.maximum_gas_liquid_density_ratio < 1:
            raise ValueError("Gas/liquid density contrast required")
        if not 0 < self.maximum_deviation_deg <= 80:
            raise ValueError("Declared inclination range must stay within 80 degrees of vertical")
        if not 4 <= self.minimum_dimensionless_diameter < self.maximum_dimensionless_diameter:
            raise ValueError("Positive Kutateladze diameter envelope required")

    def geometry(self, diameter_m, axial_gravity_m_s2, gravity_m_s2):
        if not math.isfinite(diameter_m) or diameter_m <= 0 or gravity_m_s2 <= 0:
            raise ValueError("Holmes drift requires positive diameter and gravity")
        cosine = abs(axial_gravity_m_s2) / gravity_m_s2
        if not math.cos(math.radians(self.maximum_deviation_deg)) <= cosine <= 1 + 1e-12:
            raise ValueError("Holmes pipe inclination outside declared domain")
        cosine = min(1.0, cosine)
        factor = math.sqrt(cosine) * (1 + math.sqrt(max(0.0, 1 - cosine**2))) ** 2
        return (1.0 if axial_gravity_m_s2 > 0 else -1.0), factor

    def _scales(self, ag, rg, rl, diameter_m, gravity, inclination_factor):
        contrast = (rl - rg).clamp_min(1e-20)
        dd = diameter_m * (gravity * contrast / self.surface_tension_n_m).sqrt()
        knots = dd.new_tensor([2.0, 4.0, 10.0, 14.0, 20.0, 28.0, 50.0])
        heights = dd.new_tensor([0.0, 1.0, 2.1, 2.5, 2.8, 3.0, 3.2])
        # Sum of clipped ramps: exact piecewise-linear interpolation, with no
        # tensor indexing that would synchronize or fail in batched line search.
        fraction = ((dd[..., None] - knots[:-1]) / (knots[1:] - knots[:-1])).clamp(0, 1)
        ku = heights[0] + ((heights[1:] - heights[:-1]) * fraction).sum(-1)
        base = (self.surface_tension_n_m * gravity * contrast / rl.square()).pow(0.25)
        film = ku * (rl / rg.clamp_min(1e-20)).sqrt() * base
        transition = (
            (ag - self.bubble_transition_start)
            / (self.bubble_transition_end - self.bubble_transition_start)
        ).clamp(0, 1)
        return dd, ku, base * inclination_factor, film, transition

    def _relation(self, j, ag, rg, rl, ku, scale, film, transition):
        beta_velocity = self.velocity_sensitivity * ag * j / film.clamp_min(1e-30)
        beta = torch.maximum(ag, beta_velocity)
        reduced = (
            (beta - self.profile_flattening_start) / (1 - self.profile_flattening_start)
        ).clamp(0, 1)
        denominator = 1 + (self.profile_parameter - 1) * reduced.square()
        c0 = self.profile_parameter / denominator
        dc = (
            -2
            * self.profile_parameter
            * (self.profile_parameter - 1)
            * reduced
            / denominator.square()
        )
        dc = (
            dc
            * self.velocity_sensitivity
            * ag
            / (film.clamp_min(1e-30) * (1 - self.profile_flattening_start))
        )
        dc = torch.where((beta_velocity > ag) & (reduced > 0) & (reduced < 1), dc, 0.0)
        ratio = (rg / rl).clamp_min(1e-30).sqrt()
        a = 1 - ag * c0
        b = a + ratio * ag * c0
        ck = 1.53 * (1 - transition) + ku * transition * c0
        safe_b = b.clamp_min(1e-30)
        vd = scale * ck * a / safe_b
        dvd = scale * (ku * transition * a / safe_b - ck * ag * ratio / safe_b.square())
        vg = c0 * j + vd
        derivative = rl + ag * (rg - rl) * (c0 + (j + dvd) * dc)
        return vg, derivative, c0, vd

    def _monotone_bound(self, ag, rg, rl, ku, scale, film, transition):
        # A sufficient GLOBAL slope bound, not a check only at the chosen root.
        # C decreases on the profile-flattening interval. Outside it F is linear.
        cmax = self._relation(ag * 0, ag, rg, rl, ku, scale, film, transition)[2]
        ratio = (rg / rl).clamp_min(1e-30).sqrt()
        bmin = (1 - (1 - ratio) * ag * cmax).clamp_min(1e-30)
        ckmax = 1.53 * (1 - transition) + ku * transition * cmax
        dvd_min = -scale * ckmax * ag * ratio / bmin.square()
        j_min = (
            torch.maximum(ag, ag.new_tensor(self.profile_flattening_start))
            * film
            / (self.velocity_sensitivity * ag.clamp_min(1e-30))
        )
        peak = (
            min(1.0, 1 / math.sqrt(3 * (self.profile_parameter - 1)))
            if self.profile_parameter > 1
            else 1.0
        )
        dc_max = (
            2
            * self.profile_parameter
            * (self.profile_parameter - 1)
            * peak
            / (1 + (self.profile_parameter - 1) * peak**2) ** 2
            * self.velocity_sensitivity
            * ag
            / (film.clamp_min(1e-30) * (1 - self.profile_flattening_start))
        )
        return (
            rl - ag * (rl - rg) * cmax - ag * (rl - rg) * (-(j_min + dvd_min)).clamp_min(0) * dc_max
        )

    def _volume_flux(self, args, mass_flux):
        ag, rg, rl, *_ = args
        if self.profile_parameter == 1:
            vd = self._relation(mass_flux * 0, *args)[3]
            return (mass_flux - ag * (rg - rl) * vd) / (rl + ag * (rg - rl))
        # Detached safeguarded Newton seed. Forward AD is explicitly detached as
        # well; no_grad alone does not disable jacfwd tangents.
        with torch.no_grad():
            aa = tuple(v.detach() for v in args)
            av, gv, lv, kv, sv, _fv, _tv = aa
            flux = mass_flux.detach()
            cmax = self._relation(flux * 0, *aa)[2]
            dmin = (lv + av * (gv - lv) * cmax).clamp_min(1e-20)
            dmax = lv + av * (gv - lv)
            vmax = sv * torch.maximum(kv * cmax, kv.new_tensor(1.53))
            upper_flux = flux + av * (lv - gv) * vmax
            lower = torch.minimum(flux / dmin, flux / dmax)
            upper = torch.maximum(upper_flux / dmin, upper_flux / dmax)
            j = (flux + av * (lv - gv) * self._relation(flux * 0, *aa)[3]) / dmin
            for _ in range(8):
                vg, derivative, *_ = self._relation(j, *aa)
                residual = lv * j + av * (gv - lv) * vg - flux
                lower = torch.where(residual <= 0, j, lower)
                upper = torch.where(residual >= 0, j, upper)
                trial = j - residual / derivative.clamp_min(1e-20)
                j = torch.where((trial >= lower) & (trial <= upper), trial, (lower + upper) / 2)
        j = j.detach()
        for _ in range(2):
            vg, derivative, *_ = self._relation(j, *args)
            residual = rl * j + ag * (rg - rl) * vg - mass_flux
            j = j - residual / derivative.clamp_min(1e-20)
        return j

    def parameters(
        self,
        phase_volumes,
        phase_densities,
        bulk_velocity,
        *,
        diameter_m,
        axial_gravity_m_s2,
        gravity_m_s2,
        **context,
    ):
        direction, inclination = self.geometry(diameter_m, axial_gravity_m_s2, gravity_m_s2)
        alpha = phase_volumes / phase_volumes.sum()
        gas = (torch.arange(len(alpha), device=alpha.device) == phase_densities.argmin()).to(
            alpha.dtype
        )
        ag = (alpha * gas).sum()
        rg = (phase_densities * gas).sum()
        rl = (alpha * phase_densities * (1 - gas)).sum() / (1 - ag).clamp_min(1e-30)
        mean = (alpha * phase_densities).sum()
        mass_flux = mean * bulk_velocity * direction
        dd, ku, scale, film, transition = self._scales(
            ag, rg, rl, diameter_m, gravity_m_s2, inclination
        )
        args = (ag, rg, rl, ku, scale, film, transition)
        j = self._volume_flux(args, mass_flux)
        vg, derivative, c0, vd = self._relation(j, *args)
        residual = rl * j + ag * (rg - rl) * vg - mass_flux
        vl = (mean * bulk_velocity * direction - ag * rg * vg) / ((1 - ag) * rl).clamp_min(1e-30)
        velocities = direction * (gas * vg + (1 - gas) * vl)
        slope_bound = self._monotone_bound(*args)
        valid = (
            (slope_bound > 1e-8 * rl)
            & torch.isfinite(velocities).all()
            & (derivative > 1e-8 * rl)
            & (residual.abs() <= 1e-9 * (1 + mass_flux.abs()))
            & (1 - ag * c0 > 0)
            & (ag >= self.minimum_gas_volume_fraction)
            & (ag <= self.maximum_gas_volume_fraction)
            & (rg / rl <= self.maximum_gas_liquid_density_ratio)
            & (dd >= self.minimum_dimensionless_diameter)
            & (dd <= self.maximum_dimensionless_diameter)
        )
        return {
            "velocities": velocities,
            "valid": valid,
            "volume_flux_m_s": j * direction,
            "mass_flux_slope_lower_bound": slope_bound,
            "distribution_parameter": c0,
            "drift_velocity_m_s": vd * direction,
        }

    def velocities(self, phase_volumes, phase_densities, bulk_velocity, **context):
        if len(phase_volumes) == 1:
            return bulk_velocity.expand_as(phase_volumes), bulk_velocity.new_ones(
                (), dtype=torch.bool
            )
        result = self.parameters(phase_volumes, phase_densities, bulk_velocity, **context)
        return result["velocities"], result["valid"] & (len(phase_volumes) == 2)
