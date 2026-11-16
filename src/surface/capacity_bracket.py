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
        if self.slow_updates >= 2 or self.low_margin is None or self.high_margin is None:
            self.slow_updates = 0
            return middle
        ratio = -self.low_margin / (self.high_margin - self.low_margin)
        candidate = self.low + ratio * (self.high - self.low)
        # Probe within one tolerance of an endpoint instead of repeatedly
        # revisiting a nearly identical secant root. Width still certifies stop.
        candidate = max(
            self.low + self.tolerance / 2, min(self.high - self.tolerance / 2, candidate)
        )
        return candidate if self.low < candidate < self.high else middle

    def update(self, candidate, feasible, margin):
        if not self.low < candidate < self.high:
            raise ValueError("Candidate must be strictly inside its bracket")
        old_width = self.high - self.low
        if feasible:
            self.low, self.low_margin = candidate, self._margin(margin, True)
        else:
            self.high, self.high_margin = candidate, self._margin(margin, False)
        self.slow_updates = self.slow_updates + 1 if self.high - self.low > 0.8 * old_width else 0


def facility_constraint_margin(report):
    """Dimensionless signed measured constraints; positive means violated."""
    margins = []
    for row in report["limits"]:
        scale = max(abs(row.get("minimum") or 0), abs(row.get("maximum") or 0), 1e-12)
        if row.get("minimum") is not None:
            margins.append((row["minimum"] - row["value"]) / scale)
        if row.get("maximum") is not None:
            margins.append((row["value"] - row["maximum"]) / scale)
    valve = (report.get("gas_export") or {}).get("valve")
    if valve is not None and valve["status"] == "flowing":
        margins.append(valve["utilization"] - 1)
    return max(margins, default=-1.0)
