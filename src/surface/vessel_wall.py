"""Finite, well-mixed wall heat storage and passive UA coupling (SI)."""

import math
from dataclasses import asdict, dataclass, replace

from .liquid import ConservationError, PropertyModelError, _number
from .operation import _digest


@dataclass(frozen=True)
class VesselWall:
    heat_capacity_j_k: float
    fluid_ua_w_k: float
    ambient_ua_w_k: float = 0.0
    ambient_temperature_k: float = 298.15
    reference_temperature_k: float = 298.15
    min_temperature_k: float = 1.0
    max_temperature_k: float = 2000.0
    max_temperature_change_k: float = 5.0

    def __post_init__(self):
        for key in (
            "heat_capacity_j_k",
            "ambient_temperature_k",
            "reference_temperature_k",
            "min_temperature_k",
            "max_temperature_k",
            "max_temperature_change_k",
        ):
            _number(getattr(self, key), key, positive=True)
        for key in ("fluid_ua_w_k", "ambient_ua_w_k"):
            _number(getattr(self, key), key, nonnegative=True)
        if (
            not self.min_temperature_k < self.max_temperature_k
            or not self.min_temperature_k <= self.reference_temperature_k <= self.max_temperature_k
        ):
            raise ValueError("Invalid wall temperature/reference interval")

    @property
    def model_id(self):
        return _digest(asdict(self))

    def energy(self, t):
        return self.heat_capacity_j_k * (t - self.reference_temperature_k)

    def entropy(self, t):
        return self.heat_capacity_j_k * math.log(t / self.reference_temperature_k)

    def attach(self, state, temperature_k):
        _number(temperature_k, "wall_temperature_k", positive=True)
        if not self.min_temperature_k <= temperature_k <= self.max_temperature_k:
            raise PropertyModelError("Wall temperature outside declared domain")
        return replace(
            state,
            wall_temperature_k=temperature_k,
            wall_energy_j=self.energy(temperature_k),
            wall_model_id=self.model_id,
        )

    def inspect(self, state):
        if (
            state.wall_model_id != self.model_id
            or state.wall_temperature_k is None
            or state.wall_energy_j is None
        ):
            raise PropertyModelError("Wall state/model mismatch")
        if not self.min_temperature_k <= state.wall_temperature_k <= self.max_temperature_k:
            raise PropertyModelError("Wall state outside declared temperature domain")
        if abs(state.wall_energy_j - self.energy(state.wall_temperature_k)) > 1e-6 + 1e-12 * max(
            1, abs(state.wall_energy_j)
        ):
            raise ConservationError("Wall energy does not match wall temperature")
        return self.entropy(state.wall_temperature_k)
