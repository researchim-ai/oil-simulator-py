"""Immutable piecewise-linear equipment maps with explicit extrapolation policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class EquipmentCurve:
    flow_m3_s: tuple
    values: tuple

    def __post_init__(self):
        x, y = tuple(self.flow_m3_s), tuple(self.values)
        if len(x) < 2 or len(x) != len(y):
            raise ValueError("Equipment curve requires matching arrays with at least two points")
        if any(
            isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in x + y
        ):
            raise ValueError("Equipment curve values must be finite numbers")
        if x[0] < 0 or any(b <= a for a, b in zip(x, x[1:])):
            raise ValueError(
                "Equipment curve flow axis must be nonnegative and strictly increasing"
            )
        object.__setattr__(self, "flow_m3_s", x)
        object.__setattr__(self, "values", y)

    def evaluate(self, flow_m3_s, extrapolate=False):
        if isinstance(flow_m3_s, bool) or not isinstance(flow_m3_s, Real):
            raise TypeError("Equipment curve query must be a number")
        q = float(flow_m3_s)
        if not math.isfinite(q):
            raise ValueError("Equipment curve query must be finite")
        x, y = self.flow_m3_s, self.values
        if not extrapolate and not x[0] <= q <= x[-1]:
            raise ValueError("Flow outside equipment map")
        i = min(max(int(np.searchsorted(x, q, side="right")) - 1, 0), len(x) - 2)
        slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        return y[i] + slope * (q - x[i]), slope
