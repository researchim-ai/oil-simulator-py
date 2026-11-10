"""Dry-patch force criterion with explicit material wetting hysteresis.

Hartley--Murgatroyd's laminar gravity-film limit is recovered exactly as
delta/R -> 0. The current annular Stokes profile is integrated instead of
silently inserting a gravity-only film thickness under opposing gas shear.
This diagnoses whether full wetting is supported; it does not evolve a
contact line, rivulet geometry, wave field or a nucleation barrier.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .annular_film import annular_response, radial_pressure_mobility
from .liquid import _number


@dataclass(frozen=True)
class FilmWettingCriterion:
    surface_tension_n_m: float
    receding_contact_angle_deg: float
    advancing_contact_angle_deg: float
    require_full_wetting: bool = False
    maximum_curvature_ratio: float = 0.05
    model: str = "hartley_murgatroyd_profile_v1"

    def __post_init__(self):
        _number(self.surface_tension_n_m, "surface tension", positive=True)
        if not 0 <= self.receding_contact_angle_deg <= self.advancing_contact_angle_deg < 180:
            raise ValueError("Measured ordered contact angles in [0,180) required")
        if (
            not 0 < self.maximum_curvature_ratio <= 0.05
            or type(self.require_full_wetting) is not bool
        ):
            raise ValueError("Thin-film curvature ratio <=.05 and explicit wetting policy required")
        if self.model != "hartley_murgatroyd_profile_v1":
            raise ValueError("Unknown wetting criterion")

    def to_dict(self):
        return asdict(self)

    def diagnose(self, alpha, diameter, viscosity, density, liquid_velocity, shear):
        eta, beta, c, perimeter = annular_response(alpha, diameter, viscosity)
        s = torch.sqrt(1 - alpha)
        gap = alpha / (1 + s)
        radius = diameter / 2
        nodes = alpha.new_tensor(
            [
                0.0198550717512319,
                0.101666761293187,
                0.237233795041836,
                0.408282678752175,
                0.591717321247825,
                0.762766204958164,
                0.898333238706813,
                0.980144928248768,
            ]
        )
        weights = alpha.new_tensor(
            [
                0.0506142681451881,
                0.111190517226687,
                0.156853322938944,
                0.181341891689181,
                0.181341891689181,
                0.156853322938944,
                0.111190517226687,
                0.0506142681451881,
            ]
        )
        e = -gap[..., None] * (1 - nodes)
        w = radius**2 * radial_pressure_mobility(e, alpha[..., None])
        h = -radius * s[..., None] * torch.log1p(e)
        area = math.pi * radius**2 * alpha
        driving = (perimeter * eta * shear - c * liquid_velocity) / area
        velocity = (-w * driving[..., None] + h * shear[..., None]) / viscosity
        force = 0.5 * density * radius * gap * (weights * velocity.square() * (1 + e)).sum(-1)
        lower = self.surface_tension_n_m * (
            1 - math.cos(math.radians(self.receding_contact_angle_deg))
        )
        upper = self.surface_tension_n_m * (
            1 - math.cos(math.radians(self.advancing_contact_angle_deg))
        )
        interface_velocity = eta * liquid_velocity + beta * shear
        wall_derivative = (driving * radius * alpha / 2 - s * shear) / viscosity
        # The Stokes annular profile has at most one interior extremum.
        # Endpoint signs and the wall slope therefore detect narrow reverse
        # layers which quadrature sampling alone could miss.
        no_reversal = (
            (liquid_velocity * interface_velocity >= 0)
            & (liquid_velocity * wall_derivative <= 0)
            & ((liquid_velocity != 0) | (shear == 0))
        )
        valid = (gap <= self.maximum_curvature_ratio) & no_reversal & torch.isfinite(force)
        wet = valid & (force >= upper)
        dry = valid & (force < lower)
        return {
            "momentum_force_n_m": force,
            "receding_force_n_m": torch.full_like(force, lower),
            "advancing_force_n_m": torch.full_like(force, upper),
            "profile_valid": valid,
            "full_wetting_supported": wet,
            "dry_patch_supported": dry,
            "history_dependent": valid & (~wet) & (~dry),
            "curvature_ratio": gap,
        }


@dataclass(frozen=True)
class LongWaveWettingPotential:
    """Shared material wall energy formerly embedded in ThinFilmDynamics.

    The adsorption length describes actual liquid; it is not a mass floor.
    The default nonzero-angle law is singular at h=0. A finite dry material
    continuation is available only when explicitly selected.
    """

    surface_tension_n_m: float
    adsorption_thickness_m: float
    equilibrium_contact_angle_deg: float
    dry_extension: str | None = None

    def __post_init__(self):
        _number(self.surface_tension_n_m, "surface tension", positive=True)
        _number(self.adsorption_thickness_m, "material adsorption length", positive=True)
        angle = _number(self.equilibrium_contact_angle_deg, "contact angle", nonnegative=True)
        if angle > 20:
            raise ValueError("Long-wave wetting potential requires contact angle <=20 degrees")
        if self.dry_extension not in (None, "c2_half_adsorption"):
            raise ValueError("Unknown explicitly declared dry-state material extension")

    @property
    def scale(self):
        return (
            10
            / 3
            * self.surface_tension_n_m
            * (1 - math.cos(math.radians(self.equilibrium_contact_angle_deg)))
        )

    def potential(self, h):
        if self.scale == 0:
            return torch.zeros_like(h)
        safe = h if self.dry_extension is None else h.clamp_min(self.adsorption_thickness_m / 2)
        r = self.adsorption_thickness_m / safe
        original = self.scale * (-r.square() / 2 + r**5 / 5)
        if self.dry_extension is None:
            return original
        x = (h / (self.adsorption_thickness_m / 2)).clamp_max(1)
        # Explicit C2 Hermite material continuation: Phi(0)=Phi'(0)=Phi''(0)=0,
        # matching the original Phi/Phi'/Phi'' at h_ads/2. No liquid is stored
        # by this energy evaluation; h=0 is an actual absent liquid inventory.
        dry = self.scale * (x**3 * (1230 + x * (-2210 + 1002 * x)) / 5)
        return torch.where(h < self.adsorption_thickness_m / 2, dry, original)

    def _original_difference(self, a, b):
        ra, rb = self.adsorption_thickness_m / a, self.adsorption_thickness_m / b
        fifth = sum(ra**k * rb ** (4 - k) for k in range(5))
        return self.scale / self.adsorption_thickness_m * ra * rb * ((ra + rb) / 2 - fifth / 5)

    def _dry_difference(self, a, b):
        length = self.adsorption_thickness_m / 2
        x, y = a / length, b / length
        return (
            self.scale
            / length
            * sum(
                coefficient * sum(x**k * y ** (degree - 1 - k) for k in range(degree))
                for degree, coefficient in ((3, 246), (4, -442), (5, 200.4))
            )
        )

    def divided_difference(self, a, b):
        if self.scale == 0:
            return torch.zeros_like(a + b)
        if self.dry_extension is None:
            return self._original_difference(a, b)
        length = self.adsorption_thickness_m / 2
        low, high = torch.minimum(a, b), torch.maximum(a, b)
        below, above = low.clamp_max(length), high.clamp_min(length)
        left = self._dry_difference(below, high.clamp_max(length))
        right = self._original_difference(low.clamp_min(length), above)
        cross = (
            self._dry_difference(below, below.new_full(below.shape, length)) * (length - below)
            + self._original_difference(above.new_full(above.shape, length), above)
            * (above - length)
        ) / torch.where(high > low, high - low, torch.ones_like(high))
        return torch.where(high <= length, left, torch.where(low >= length, right, cross))

    def to_dict(self):
        result = {
            "adsorption_thickness_m": self.adsorption_thickness_m,
            "equilibrium_contact_angle_deg": self.equilibrium_contact_angle_deg,
        }
        if self.dry_extension is not None:
            result["dry_extension"] = self.dry_extension
        return result
