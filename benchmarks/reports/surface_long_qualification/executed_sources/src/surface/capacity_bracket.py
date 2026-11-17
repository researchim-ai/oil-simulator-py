"""Safeguarded scalar proposals; every candidate still needs a physical solve.

A bracket is only as global as its caller's feasible-domain sampling. Signed
margins accelerate proposals, never certify an unsolved candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CapacityBracket:
    low: float
    high: float
    tolerance: float
    low_margin: float | None = None
    high_margin: float | None = None
    slow_updates: int = 0
    low_margins: dict[str, float | None] | None = None
    high_margins: dict[str, float | None] | None = None

    def __post_init__(self):
        if not (0 < self.low < self.high <= 1 and 0 < self.tolerance < 1):
            raise ValueError("Ordered finite capacity bracket required")
        self.low_margin = self._margin(self.low_margin, True)
        self.high_margin = self._margin(self.high_margin, False)

    @staticmethod
    def _margin(value, feasible):
        if value is None or not math.isfinite(value):
            return None
        if (feasible and value > 0) or (not feasible and value <= 0):
            return None
        return float(value)

    @property
    def complete(self):
        return self.high - self.low <= self.tolerance

    def proposal(self):
        if self.complete:
            raise ValueError("Capacity bracket already resolved")
        middle = self.low + (self.high - self.low) / 2
        low, high = self.low_margin, self.high_margin
        # A max over unrelated constraints can have a long flat plateau.
        # Predict using the SAME violated constraint at both solved endpoints.
        pairs = [(self.low_margins[k], v) for k, v in (self.high_margins or {}).items()
                 if k in (self.low_margins or {}) and v is not None and math.isfinite(v) and v > 0
                 and self.low_margins[k] is not None
                 and math.isfinite(self.low_margins[k]) and self.low_margins[k] <= 0]
        if pairs:
            # First predicted crossing constrains a jointly feasible point.
            low, high = min(pairs, key=lambda p: -p[0] / (p[1] - p[0]))
        if self.slow_updates >= 2 or low is None or high is None:
            self.slow_updates = 0
            return middle
        ratio = -low / (high - low)
        candidate = self.low + ratio * (self.high - self.low)
        # Probe within one tolerance of an endpoint instead of repeatedly
        # revisiting a nearly identical secant root. Width still certifies stop.
        candidate = max(
            self.low + self.tolerance / 2, min(self.high - self.tolerance / 2, candidate)
        )
        return candidate if self.low < candidate < self.high else middle

    def update(self, candidate, feasible, margin, *, margins=None):
        if not self.low < candidate < self.high:
            raise ValueError("Candidate must be strictly inside its bracket")
        old_width = self.high - self.low
        if feasible:
            self.low, self.low_margin = candidate, self._margin(margin, True)
            self.low_margins = margins
        else:
            self.high, self.high_margin = candidate, self._margin(margin, False)
            self.high_margins = margins
        self.slow_updates = self.slow_updates + 1 if self.high - self.low > 0.8 * old_width else 0


def facility_constraint_margin(report):
    """Dimensionless signed measured constraints; positive means violated."""
    return max(facility_constraint_margins(report).values(), default=-1.0)


def facility_constraint_margins(report):
    """Stable constraint identities for proposal-only endpoint matching."""
    margins = {}
    for index, row in enumerate(report["limits"]):
        scale = max(abs(row.get("minimum") or 0), abs(row.get("maximum") or 0), 1e-12)
        if row.get("minimum") is not None:
            margins[f"limit:{index}:minimum"] = (row["minimum"] - row["value"]) / scale
        if row.get("maximum") is not None:
            margins[f"limit:{index}:maximum"] = (row["value"] - row["maximum"]) / scale
    valve = (report.get("gas_export") or {}).get("valve")
    if valve is not None and valve["status"] == "flowing":
        margins["gas_export.valve"] = valve["utilization"] - 1
    return margins
