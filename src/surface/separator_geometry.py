"""Flat-ended cylindrical separator geometry and bounded ideal level-flow control."""

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from .liquid import PropertyModelError, _number


@dataclass(frozen=True)
class SeparatorGeometry:
    diameter_m: float
    length_m: float
    orientation: str = "horizontal"

    def __post_init__(self):
        for key in ("diameter_m", "length_m"):
            _number(getattr(self, key), key, positive=True)
        if self.orientation not in ("horizontal", "vertical"):
            raise ValueError("Cylinder orientation must be horizontal or vertical")

    @property
    def volume_m3(self):
        return math.pi * self.diameter_m**2 / 4 * self.length_m

    @property
    def height_m(self):
        return self.diameter_m if self.orientation == "horizontal" else self.length_m

    def liquid_volume_m3(self, level_m):
        h = _number(level_m, "level_m", nonnegative=True)
        if h > self.height_m:
            raise ValueError("Liquid level outside separator")
        if self.orientation == "vertical":
            return h * math.pi * self.diameter_m**2 / 4
        r = self.diameter_m / 2
        if h > r:
            return self.volume_m3 - self.liquid_volume_m3(2 * r - h)
        theta = math.acos(max(-1, min(1, 1 - h / r)))
        segment = (
            theta**3 * (2 / 3 - 2 / 15 * theta**2 + 4 / 315 * theta**4)
            if theta < 1e-3
            else theta - math.sin(theta) * math.cos(theta)
        )
        return r * r * segment * self.length_m

    def level_m(self, liquid_volume_m3):
        v = _number(liquid_volume_m3, "liquid_volume_m3", nonnegative=True)
        if v > self.volume_m3 * (1 + 1e-10):
            raise PropertyModelError("Liquid volume exceeds separator geometry")
        v = min(v, self.volume_m3)
        if self.orientation == "vertical":
            return self.height_m * v / self.volume_m3
        if v == 0:
            return 0.0
        if v == self.volume_m3:
            return self.height_m
        return brentq(
            lambda h: self.liquid_volume_m3(h) - v, 0, self.height_m, xtol=1e-13 * self.height_m
        )

    def from_equilibrium(self, moles, equilibrium):
        if equilibrium["phase_count"] != 2:
            raise PropertyModelError("Separator level requires resolved liquid/vapour phases")
        q = equilibrium["phases"][0]
        volume = math.fsum(moles) * q["molar_fraction"] * q["molar_volume_m3_mol"]
        return self.level_m(volume)


@dataclass(frozen=True)
class LevelWithdrawal:
    setpoint_m: float
    gain_mol_s_m: float
    max_molar_rate_mol_s: float
    bias_mol_s: float = 0.0

    def __post_init__(self):
        for key in ("setpoint_m", "gain_mol_s_m", "bias_mol_s"):
            _number(getattr(self, key), key, nonnegative=True)
        _number(self.max_molar_rate_mol_s, "max_molar_rate_mol_s", positive=True)
        if self.bias_mol_s > self.max_molar_rate_mol_s:
            raise ValueError("Level flow bias exceeds capacity")

    def rate(self, level_m):
        _number(level_m, "liquid_level_m", nonnegative=True)
        return min(
            self.max_molar_rate_mol_s,
            max(0.0, self.bias_mol_s + self.gain_mol_s_m * (level_m - self.setpoint_m)),
        )
