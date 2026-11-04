"""Imposed, density-ordered PR layers and finite circular pickup exposure.

Kinematic geometry only: no gravity pressure or sedimentation law. The PR
uniform-pressure model currently enables this closure only with g=0.
"""

import math
from dataclasses import dataclass

import torch

from .liquid import _name, _number
from .separator_geometry import SeparatorGeometry
from .three_phase_properties import disk_area_moment, level_tensor


@dataclass(frozen=True)
class EquilibriumLevelControl:
    component: str
    setpoint_m: float
    opening_gain_per_m: float
    opening_bias: float = 0.0
    minimum_mole_fraction: float = 0.5

    def __post_init__(self):
        _name(self.component)
        _number(self.setpoint_m, "setpoint_m", nonnegative=True)
        _number(self.opening_gain_per_m, "opening_gain_per_m", nonnegative=True)
        for key in ("opening_bias", "minimum_mole_fraction"):
            if not 0 <= _number(getattr(self, key), key) <= 1:
                raise ValueError(f"{key} outside [0,1]")

    def value(self, layout, mole_fractions, names):
        column = mole_fractions[:, names.index(self.component)]
        index = column.argmax()
        peak = column.max()
        present = (peak > 0) & (peak >= self.minimum_mole_fraction)
        selected = torch.gather(layout["phase_tops_m"], 0, index.reshape(1)).squeeze(0)
        level = torch.where(present, selected, column.new_zeros(()))
        opening = (self.opening_bias + self.opening_gain_per_m * (level - self.setpoint_m)).clamp(
            0, 1
        )
        return torch.where(present, opening, torch.zeros_like(opening)), level, present


def validate_pickup(geometry, height_m, diameter_m):
    if (height_m is None) != (diameter_m is None):
        raise ValueError("Geometric pickup needs height and diameter together")
    if height_m is None:
        return
    if not isinstance(geometry, SeparatorGeometry):
        raise TypeError("Geometric pickup requires vessel geometry")
    height = _number(height_m, "pickup height", nonnegative=True)
    diameter = _number(diameter_m, "pickup diameter", positive=True)
    if diameter > geometry.diameter_m:
        raise ValueError("Pickup diameter exceeds vessel diameter")
    if not diameter / 2 <= height <= geometry.height_m - diameter / 2:
        raise ValueError("Pickup aperture extends outside vessel height")


def phase_layout(phase_moles, properties, geometry):
    """Map EOS phases to bottom/top levels; phase indices remain unchanged.

    Density ordering is an imposed ideal-layer closure. Derivatives are
    piecewise valid away from density ties and active-phase changes.
    """
    volumes = phase_moles * properties["molar_volume_m3_mol"]
    order = torch.argsort(properties["density_kg_m3"], descending=True, stable=True)
    ordered = volumes.index_select(0, order)
    # Normalize only the trial geometry, not physical inventories. The full
    # Newton residual still enforces actual volume=geometric volume. This
    # keeps layout well defined for intermediate Newton iterates.
    fractions = ordered / ordered.sum().clamp_min(1e-300)
    cumulative = torch.cumsum(fractions, dim=0)
    top = torch.stack([level_tensor(v * geometry.volume_m3, geometry) for v in cumulative])
    bottom = torch.cat((top.new_zeros(1), top[:-1]))
    inverse = torch.argsort(order)
    density = properties["density_kg_m3"].index_select(0, order)
    resolved = ((density[:-1] - density[1:]) > 1e-10 * density.abs().max().clamp_min(1)).all()
    return {
        "density_order_resolved": resolved,
        "phase_order": order,
        "phase_bottoms_m": bottom.index_select(0, inverse),
        "phase_tops_m": top.index_select(0, inverse),
    }


def pickup_fractions(layout, height_m, diameter_m):
    """Exact circular aperture area occupied by each imposed phase layer."""
    bottom = height_m - diameter_m / 2
    high = layout["phase_tops_m"] - bottom
    low = layout["phase_bottoms_m"] - bottom
    area_high = disk_area_moment(high, diameter_m)[0]
    area_low = disk_area_moment(low, diameter_m)[0]
    return ((area_high - area_low) / (math.pi * diameter_m**2 / 4)).clamp(0, 1)
