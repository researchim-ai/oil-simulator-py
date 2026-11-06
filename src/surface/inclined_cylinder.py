"""Horizontal cuts of a finite inclined circular cylinder, with live moments.

No fluid law lives here: gravity-equilibrated columns use these actual cut
volumes and vertical moments rather than an equivalent horizontal cylinder.
"""

import math
from dataclasses import dataclass

import torch

from .liquid import _number
from .separator_geometry import SeparatorGeometry
from .three_phase_properties import disk_area_moment


def disk_second_moment(height, diameter):
    """Integral of y**2 dA about the disk bottom, stable at both caps."""
    h = height.clamp(0, diameter)
    low = torch.where(h <= diameter / 2, h, diameter - h)
    x = low / diameter
    r = diameter / 2
    theta = torch.acos((1 - low.clamp_min(diameter * 1e-4) / r).clamp(-1 + 1e-15, 1 - 1e-15))
    regular = r**4 * (
        1.25 * theta
        - 0.5 * torch.sin(2 * theta)
        - 4 / 3 * torch.sin(theta) ** 3
        - torch.sin(4 * theta) / 16
    )
    small = (
        2
        * math.sqrt(diameter)
        * low.pow(3.5)
        * (2 / 7 - x / 9 - x * x / 44 - x**3 / 104 - x**4 / 192)
    )
    q = torch.where(x < 1e-4, small, regular)
    area, moment = disk_area_moment(low, diameter)
    total = 5 / 4 * r * r * math.pi * r * r
    return torch.where(h <= r, q, total - diameter**2 * area + 2 * diameter * moment - q)


@dataclass(frozen=True)
class InclinedCylinderGeometry(SeparatorGeometry):
    orientation: str = "inclined"
    inclination_rad: float = math.pi / 4

    def __post_init__(self):
        _number(self.diameter_m, "cylinder diameter", positive=True)
        _number(self.length_m, "cylinder length", positive=True)
        angle = _number(self.inclination_rad, "cylinder inclination")
        if self.orientation != "inclined" or abs(angle) > math.pi / 2:
            raise ValueError("Inclined cylinder requires angle in [-pi/2, pi/2]")

    @property
    def height_m(self):
        return self.length_m * abs(math.sin(self.inclination_rad)) + self.diameter_m * math.cos(
            self.inclination_rad
        )

    @property
    def axial_span_m(self):
        return self.length_m * abs(math.sin(self.inclination_rad))

    def _small_angle(self, h):
        # Short axial height span: avoid subtracting almost equal primitives.
        # Sixteen Gauss points suffice on smooth interior cuts. Cap intervals
        # are split before quadrature; a cosine map regularizes BOTH cap ends.
        from .gauss_rules import legendre_rule

        x, w = legendre_rule(16, h)
        a = abs(math.sin(self.inclination_rad))
        c = math.cos(self.inclination_rad)
        cuts = torch.stack(
            (
                h * 0,
                ((h - self.diameter_m * c) / a).clamp(0, self.length_m),
                (h / a).clamp(0, self.length_m),
                h * 0 + self.length_m,
            ),
            -1,
        )
        lo, hi = cuts[..., :-1], cuts[..., 1:]
        t = (x + 1) / 2
        s = lo[..., None] + (hi - lo)[..., None] * (1 - t.mul(math.pi).cos()) / 2
        weights = (hi - lo)[..., None] * w * (math.pi / 4) * t.mul(math.pi).sin()
        area, moment = disk_area_moment((h[..., None, None] - a * s) / c, self.diameter_m)
        volume = (weights * area).sum((-2, -1))
        first = (weights * (a * s * area + c * moment)).sum((-2, -1))
        chord_argument = (
            ((h[..., None, None] - a * s) / c).clamp(0, self.diameter_m)
            * (self.diameter_m - ((h[..., None, None] - a * s) / c).clamp(0, self.diameter_m))
        ).clamp_min(0)
        chord = 2 * torch.where(
            chord_argument > 0,
            torch.sqrt(chord_argument.clamp_min(self.diameter_m**2 * 1e-20)),
            torch.zeros_like(chord_argument),
        )
        slope = (weights * chord / c).sum((-2, -1))
        return volume, first, slope

    def cut_tensor(self, height):
        h = height.clamp(0, self.height_m)
        reflected = h > self.height_m / 2
        low = torch.where(reflected, self.height_m - h, h)
        a = abs(math.sin(self.inclination_rad))
        c = math.cos(self.inclination_rad)
        if a == 0:
            area, moment = disk_area_moment(low, self.diameter_m)
            v, m = area * self.length_m, moment * self.length_m
            slope = 2 * self.length_m * torch.sqrt((low * (self.diameter_m - low)).clamp_min(0))
        elif abs(self.inclination_rad) == math.pi / 2:
            slope = torch.ones_like(low) * math.pi * self.diameter_m**2 / 4
            v = slope * low
            m = v * low / 2
        elif self.axial_span_m < 1e-3 * self.diameter_m * c:
            v, m, slope = self._small_angle(low)
        else:
            yh = low / c
            yl = (low - self.axial_span_m) / c
            ah, mh = disk_area_moment(yh, self.diameter_m)
            al, ml = disk_area_moment(yl, self.diameter_m)
            jh, jl = yh * ah - mh, yl * al - ml
            v = c / a * (jh - jl)
            kh = 0.5 * (yh * yh * ah - 2 * yh * mh + disk_second_moment(yh, self.diameter_m))
            kl = 0.5 * (yl * yl * al - 2 * yl * ml + disk_second_moment(yl, self.diameter_m))
            m = low * v - c * c / a * (kh - kl)
            slope = (ah - al) / a
        return (
            torch.where(reflected, self.volume_m3 - v, v),
            torch.where(reflected, self.volume_m3 * self.height_m / 2 - self.height_m * v + m, m),
            slope,
        )

    def volume_tensor(self, height):
        return self.cut_tensor(height)[0]

    def moment_tensor(self, height):
        return self.cut_tensor(height)[1]

    def level_tensor(self, volume):
        # Detached monotone inversion followed by an implicit correction. The
        # geometric derivative therefore participates in the common Jacobian.
        interior = (volume > 0) & (volume < self.volume_m3)
        working = torch.where(interior, volume, volume * 0 + self.volume_m3 / 2)
        target = working.detach()
        lo = torch.zeros_like(target)
        hi = lo + self.height_m
        for _ in range(48):
            mid = (lo + hi) / 2
            below = self.volume_tensor(mid) < target
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        h = (lo + hi) / 2
        for _ in range(2):
            v, _, slope = self.cut_tensor(h)
            h = h + (working - v) / slope.clamp_min(self.volume_m3 / self.height_m * 1e-30)
        return torch.where(
            volume <= 0,
            torch.zeros_like(h),
            torch.where(volume >= self.volume_m3, torch.full_like(h, self.height_m), h),
        )

    def liquid_volume_m3(self, level_m):
        h = _number(level_m, "level", nonnegative=True)
        if h > self.height_m:
            raise ValueError("Level outside inclined cylinder")
        return float(self.volume_tensor(torch.tensor(h, dtype=torch.float64)))

    def level_m(self, liquid_volume_m3):
        v = _number(liquid_volume_m3, "liquid volume", nonnegative=True)
        if v > self.volume_m3:
            raise ValueError("Volume outside inclined cylinder")
        return float(self.level_tensor(torch.tensor(v, dtype=torch.float64)))


def geometry_from_dict(data):
    cls = InclinedCylinderGeometry if data.get("orientation") == "inclined" else SeparatorGeometry
    return cls(**data)
