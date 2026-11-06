"""Frozen immiscible oil/water/ideal-gas calorics, SI, float64 Torch.

These are phase inventories, not black-oil stock components. No Rs, flash,
boiling or dissolution is implied by the oil/water/gas names.
"""

import math
from dataclasses import asdict, dataclass

import torch

from .liquid import _number
from .operation import _digest


@dataclass(frozen=True)
class ThreePhaseFluid:
    oil_density_kg_m3: float
    water_density_kg_m3: float
    oil_heat_capacity_j_kg_k: float
    water_heat_capacity_j_kg_k: float
    gas_cv_j_kg_k: float
    gas_constant_j_kg_k: float
    reference_temperature_k: float = 300.0
    reference_pressure_pa: float = 100000.0
    min_temperature_k: float = 275.0
    max_temperature_k: float = 400.0
    gas_min_temperature_k: float = 100.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            _number(value, key, positive=True)
        if self.water_density_kg_m3 <= self.oil_density_kg_m3:
            raise ValueError("Three-phase stratification requires water denser than oil")
        if (
            not self.gas_min_temperature_k
            <= self.min_temperature_k
            <= self.reference_temperature_k
            <= self.max_temperature_k
        ):
            raise ValueError("Invalid three-phase temperature domain")
        if self.min_temperature_k == self.max_temperature_k:
            raise ValueError("Three-phase temperature interval must have positive width")

    @property
    def model_id(self):
        return _digest({"model": "frozen_immiscible_caloric_three_phase_v1", **asdict(self)})

    @property
    def heat_capacities(self):
        return (self.oil_heat_capacity_j_kg_k, self.water_heat_capacity_j_kg_k, self.gas_cv_j_kg_k)

    def phase_enthalpies(self, pressure, temperature):
        d = temperature - self.reference_temperature_k
        return torch.stack(
            (
                self.oil_heat_capacity_j_kg_k * d + pressure / self.oil_density_kg_m3,
                self.water_heat_capacity_j_kg_k * d + pressure / self.water_density_kg_m3,
                self.gas_cv_j_kg_k * d + self.gas_constant_j_kg_k * temperature,
            ),
            dim=-1,
        )

    def phase_entropies(self, pressure, temperature):
        logt = torch.log(temperature / self.reference_temperature_k)
        return torch.stack(
            (
                self.oil_heat_capacity_j_kg_k * logt,
                self.water_heat_capacity_j_kg_k * logt,
                (self.gas_cv_j_kg_k + self.gas_constant_j_kg_k) * logt
                - self.gas_constant_j_kg_k
                * torch.log(pressure.clamp_min(1e-30) / self.reference_pressure_pa),
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class ThreePhaseStream:
    phase_mass_rates_kg_s: tuple
    pressure_pa: float
    temperature_k: float
    fluid_id: str
    elevation_m: float = 0.0

    def __post_init__(self):
        values = tuple(
            _number(v, "phase_mass_rates_kg_s", nonnegative=True)
            for v in self.phase_mass_rates_kg_s
        )
        if len(values) != 3:
            raise ValueError("Expected oil/water/gas PHASE mass rates")
        object.__setattr__(self, "phase_mass_rates_kg_s", values)
        _number(self.pressure_pa, "pressure_pa", positive=True)
        _number(self.temperature_k, "temperature_k", positive=True)
        _number(self.elevation_m, "elevation_m")
        if not isinstance(self.fluid_id, str):
            raise TypeError("Missing three-phase fluid identity")

    @classmethod
    def create(cls, fluid, phase_mass_rates_kg_s, pressure_pa, temperature_k, elevation_m=0.0):
        if not isinstance(fluid, ThreePhaseFluid):
            raise TypeError("Expected ThreePhaseFluid")
        t = _number(temperature_k, "temperature_k", positive=True)
        if not fluid.min_temperature_k <= t <= fluid.max_temperature_k:
            raise ValueError("Feed outside fluid temperature domain")
        return cls(tuple(phase_mass_rates_kg_s), pressure_pa, t, fluid.model_id, elevation_m)


def disk_area_moment(height, diameter):
    """Area below height and its first moment about the disk bottom.

    Stable small-cap expansions avoid cancellation. Values outside the disk
    are geometrically clipped, not clipped fluid inventories.
    """
    h = height.clamp(0, diameter)
    low = torch.where(h <= diameter / 2, h, diameter - h)
    x = low / diameter
    regular = low.clamp_min(diameter * 1e-4)
    r = diameter / 2
    theta = torch.acos((1 - regular / r).clamp(-1 + 1e-15, 1 - 1e-15))
    area_regular = r * r * (theta - torch.sin(theta) * torch.cos(theta))
    area_small = (
        2
        * math.sqrt(diameter)
        * torch.where(low > 0, low.clamp_min(diameter * 1e-20).pow(1.5), torch.zeros_like(low))
        * (2 / 3 - x / 5 - x * x / 28 - x**3 / 72 - 5 * x**4 / 704)
    )
    area = torch.where(x < 1e-4, area_small, area_regular)
    moment_regular = r * area_regular - 2 / 3 * (regular * (diameter - regular)).clamp_min(0).pow(
        1.5
    )
    moment_small = (
        2
        * math.sqrt(diameter)
        * low.pow(2.5)
        * (2 / 5 - x / 7 - x * x / 36 - x**3 / 88 - 5 * x**4 / 832)
    )
    moment = torch.where(x < 1e-4, moment_small, moment_regular)
    total = math.pi * diameter**2 / 4
    return (
        torch.where(h <= r, area, total - area),
        torch.where(h <= r, moment, total * r - diameter * area + moment),
    )


def level_tensor(volume, geometry):
    """Geometric inverse with implicit first derivative, including in jacfwd."""
    if geometry.orientation == "inclined":
        return geometry.level_tensor(volume)
    if geometry.orientation == "vertical":
        return volume / geometry.volume_m3 * geometry.height_m
    fraction = (volume.detach() / geometry.volume_m3).clamp(0, 1)
    f = torch.minimum(fraction, 1 - fraction)
    y = (3 * math.pi * f / 16).clamp_min(1e-30).pow(2 / 3)
    for _ in range(8):
        area, _ = disk_area_moment(y * geometry.diameter_m, geometry.diameter_m)
        slope = (
            2
            * geometry.diameter_m**2
            * torch.sqrt((y * (1 - y)).clamp_min(1e-30))
            / (math.pi * geometry.diameter_m**2 / 4)
        )
        y = (y - (area / (math.pi * geometry.diameter_m**2 / 4) - f) / slope).clamp(1e-20, 0.5)
    h = torch.where(fraction <= 0.5, y, 1 - y) * geometry.diameter_m
    area, _ = disk_area_moment(h, geometry.diameter_m)
    cross = 2 * geometry.length_m * torch.sqrt((h * (geometry.diameter_m - h)).clamp_min(1e-30))
    corrected = h + (volume - area * geometry.length_m) / cross
    return torch.where(
        volume <= 0,
        torch.zeros_like(corrected),
        torch.where(
            volume >= geometry.volume_m3, torch.full_like(corrected, geometry.height_m), corrected
        ),
    )


def safe_sqrt(value):
    return torch.where(value > 0, torch.sqrt(value.clamp_min(1e-30)), torch.zeros_like(value))


def frozen_nozzle(
    pressure, temperature, back_pressure, weights, fluid, elevation_change, gravity, *, mixed=False
):
    """Homogeneous frozen isentropic nozzle, with gravity and choking.

    Isentropic dh=v dp, mass flux sqrt(2*(h0-h-g dz))/v.
    Reduces to incompressible orifice and NASA ideal-gas choke limits.
    """
    pressure = pressure.clamp_min(1e-6)
    back_pressure = back_pressure.clamp_min(1e-6)
    wo, ww, wg = weights.unbind(-1)
    cp = (
        wo * fluid.oil_heat_capacity_j_kg_k
        + ww * fluid.water_heat_capacity_j_kg_k
        + wg * (fluid.gas_cv_j_kg_k + fluid.gas_constant_j_kg_k)
    )
    rm = wg * fluid.gas_constant_j_kg_k
    cp_safe = cp.clamp_min(1.0)
    alpha = rm / cp_safe
    vl = wo / fluid.oil_density_kg_m3 + ww / fluid.water_density_kg_m3
    head = gravity * elevation_change

    def values(p):
        ts = temperature * (p / pressure).clamp_min(1e-20).pow(alpha)
        dh = cp * (temperature - ts) + vl * (pressure - p) - head
        vg = rm * ts / p
        return ts, dh, vl + vg, vg

    # Analytic pure-gas critical pressure; liquid has no sonic limit here.
    gas_cp = fluid.gas_cv_j_kg_k + fluid.gas_constant_j_kg_k
    gas_alpha = fluid.gas_constant_j_kg_k / gas_cp
    gas_tcrit = 2 * (1 - gas_alpha) / (2 - gas_alpha) * (temperature - head / gas_cp)
    gas_pcrit = pressure * (gas_tcrit / temperature).clamp_min(1e-20).pow(1 / gas_alpha)
    mixed_pcrit = gas_pcrit
    if mixed:
        low = pressure.detach() * 1e-10
        high = torch.maximum(pressure.detach() * 4, back_pressure.detach() * 4)
        c0, a0, l0, r0, t0, p0, h0 = (
            v.detach() if isinstance(v, torch.Tensor) else v
            for v in (cp, alpha, vl, rm, temperature, pressure, head)
        )
        for _ in range(40):
            mid = (low + high) / 2
            tm = t0 * (mid / p0).clamp_min(1e-20).pow(a0)
            dh = c0 * (t0 - tm) + l0 * (p0 - mid) - h0
            vg = r0 * tm / mid
            criterion = 2 * dh * (1 - a0) * vg / mid - (l0 + vg).square()
            low = torch.where(criterion > 0, mid, low)
            high = torch.where(criterion > 0, high, mid)
        mixed_pcrit = (low + high) / 2
    pcrit = torch.where(vl > 0, mixed_pcrit, gas_pcrit)
    throat = torch.maximum(back_pressure, torch.where(wg > 0, pcrit, back_pressure))
    ts, dh, v, _ = values(throat)
    flux = safe_sqrt(2 * dh) / torch.where(v > 0, v, torch.ones_like(v))
    flux = torch.where(cp > 0, flux, torch.zeros_like(flux))
    return flux, throat, ts, (wg > 0) & (pcrit > back_pressure)
