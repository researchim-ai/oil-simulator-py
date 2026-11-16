"""Optional accepted-rate variation control, not a rigorous FIM time-error bound.

The endpoint-rate indicator is a cheap integration heuristic. Temporal accuracy
still needs refinement against the quantities of interest for each model.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FacilityTimeControl:
    rate_relative_tolerance: float = 0.01
    rate_absolute_tolerance_kg_s: float = 1e-8
    startup_step_days: float = 1.0
    safety_factor: float = 0.9
    maximum_growth: float = 2.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite {name} required")
        if self.rate_relative_tolerance >= 1 or self.safety_factor >= 1 or self.maximum_growth < 1:
            raise ValueError("Invalid rate/time control policy")
        if not math.isfinite(self.startup_step_days * 86400):
            raise ValueError("Startup step must be representable in seconds")

    def indicator(self, before, after):
        before, after = tuple(before), tuple(after)
        if any(not math.isfinite(v) or v < 0 for v in (*before, *after)):
            raise ValueError("Finite nonnegative component rates required")
        return max(
            0.5
            * abs(b - a)
            / (
                self.rate_absolute_tolerance_kg_s
                + self.rate_relative_tolerance * max(abs(a), abs(b))
            )
            for a, b in zip(before, after, strict=True)
        )

    def limit(self, proposed_s, dt_s=None, indicator=None, *, clipped=False):
        if not math.isfinite(proposed_s) or proposed_s <= 0:
            raise ValueError("Positive finite proposed step required")
        if indicator is None:
            return min(proposed_s, self.startup_step_days * 86400)
        if (
            not math.isfinite(indicator)
            or indicator < 0
            or dt_s is None
            or not math.isfinite(dt_s)
            or dt_s <= 0
        ):
            raise ValueError("Finite rate indicator and positive accepted duration required")
        allowed = proposed_s if indicator == 0 else dt_s * self.safety_factor / indicator
        if not clipped:
            allowed = min(allowed, dt_s * self.maximum_growth)
        return min(proposed_s, allowed)
