"""Safeguarded scalar proposals; every candidate still needs a physical solve.

A bracket is only as global as its caller's feasible-domain sampling. Signed
margins accelerate proposals, never certify an unsolved candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def local_capacity_bracket(evaluate, fraction, slopes, minimum, tolerance):
    """Track a nearby operating branch using only evaluated endpoints.

    No global optimum is certified. Missing slopes, a remote root or four
    unsuccessful probes request the caller's full domain scan instead.
    """
    rows = []
    point = fraction
    lower, upper = max(minimum, .8*fraction), min(1., 1.2*fraction)
    for _ in range(4):
        if any(point == r['fraction'] for r in rows):
            return None
        row = evaluate(point)
        rows.append(row)
        feasible = [r for r in rows if r['feasible']]
        infeasible = [r for r in rows if not r['feasible']]
        if feasible and infeasible:
            low = max(feasible, key=lambda r:r['fraction'])
            high = min(infeasible, key=lambda r:r['fraction'])
            return (low, high) if low['fraction'] < high['fraction'] else None
        margins = row.get('constraint_margins') or {}
        if not margins or any(v is None or not math.isfinite(v) for v in margins.values()):
            return None
        if any(v > 0 and k not in slopes for k,v in margins.items()):
            return None
        roots = [point-v/slopes[k] for k,v in margins.items()
                 if k in slopes and math.isfinite(slopes[k]) and slopes[k] > 0]
        if not roots:
            return None
        root = min(roots)
        proposed = root + (tolerance/2 if row['feasible'] else -tolerance/2)
        if row['feasible']:
            proposed = max(point+tolerance/2, proposed)
        else:
            proposed = min(point-tolerance/2, proposed)
        if not lower <= proposed <= upper or proposed == point:
            return None
        point = proposed
    return None


def capacity_history_slopes(trace, fraction, tolerance):
    """Local secants from solved points; proposals only, never certificates.

    Exclude tiny final brackets (cancellation) and remote saturated caps.
    One previous search is sufficient; no reservoir tensors are retained.
    """
    rows = [r for r in trace if 'certified_from_trace_index' not in r
            and abs(r['fraction'] - fraction) <= .1 * fraction
            and r.get('constraint_margins')]
    slopes, distances = {}, {}
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            dx = b['fraction'] - a['fraction']
            if abs(dx) < 8 * tolerance:
                continue
            for key, va in a['constraint_margins'].items():
                vb = b['constraint_margins'].get(key)
                if va is None or vb is None or not all(math.isfinite(v) for v in (va, vb)):
                    continue
                slope = (vb - va) / dx
                distance = max(abs(a['fraction']-fraction), abs(b['fraction']-fraction))
                if math.isfinite(slope) and slope > 0 and distance < distances.get(key, math.inf):
                    slopes[key], distances[key] = slope, distance
    return slopes


def continuation_slopes(report, tolerance):
    """Prefer wide secants; fall back to the previous resolved bracket.

    The width tolerance controls search termination, not floating-point
    resolution. A narrow, physically solved bracket can still give a useful
    proposal. These slopes never certify feasibility or carry reservoir state.
    """
    slopes = capacity_history_slopes(report['trace'], report['fraction'], tolerance)
    if slopes:
        return slopes
    bracket = report.get('capacity_bracket')
    if not bracket:
        return {}
    low, high = bracket['feasible_fraction'], bracket['infeasible_fraction']
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high <= 1
            and low == report['fraction']):
        return {}
    width = high-low
    if width <= 32*max(math.ulp(low), math.ulp(high)):
        return {}
    points = {r['fraction']: r for r in report['trace']
              if 'certified_from_trace_index' not in r}
    a, b = points.get(low), points.get(high)
    if a is None or b is None or a.get('feasible') is not True or b.get('feasible') is not False:
        return {}
    for key, va in (a.get('constraint_margins') or {}).items():
        vb = (b.get('constraint_margins') or {}).get(key)
        if va is None or vb is None or not all(math.isfinite(v) for v in (va, vb)):
            continue
        signal = vb-va
        if signal <= 64*math.ulp(max(1., abs(va), abs(vb))):
            continue
        slope = signal/width
        if math.isfinite(slope) and slope > 0:
            slopes[key] = slope
    return slopes


def feasible_secant_proposal(trace, bracket):
    """One optional current-step proposal when the upper margin is unavailable.

    Uses only solved feasible points, never a prior step's equipment response.
    This extrapolation does not certify feasibility; the caller must solve it
    and limit its use to once per search before returning to normal safeguards.
    """
    if bracket.complete or bracket.high_margins:
        return None
    rows = {r['fraction']: r for r in trace
            if r.get('feasible') is True and 'certified_from_trace_index' not in r
            and r['fraction'] <= bracket.low and r.get('constraint_margins')}
    points = sorted(rows)
    if len(points) < 2 or points[-1] != bracket.low:
        return None
    left, right = points[-2:]
    if right-left <= 32*max(math.ulp(left), math.ulp(right)):
        return None
    a, b = [rows[x]['constraint_margins'] for x in (left, right)]
    if a.keys() != b.keys():
        return None
    roots = []
    for key, value in b.items():
        prior = a[key]
        if (value is None or prior is None
                or not all(math.isfinite(v) and v <= 0 for v in (value, prior))):
            return None
        signal = value-prior
        if signal > 64*math.ulp(max(1., abs(value), abs(prior))):
            roots.append(right-value*(right-left)/signal)
    if not roots:
        return None
    root = min(roots)
    # Avoid remote extrapolation from almost coincident points or tiny caps.
    if not bracket.low <= root <= min(2*right, right+2*(right-left)):
        return None
    candidate = max(bracket.low+bracket.tolerance/2,
                    min(bracket.high-bracket.tolerance/2, root))
    return candidate if bracket.low < candidate < bracket.high else None


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
    endpoint_probe_used: bool = False

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

    def proposal(self, *, allow_endpoint_probe=False):
        if self.complete:
            raise ValueError("Capacity bracket already resolved")
        middle = self.low + (self.high - self.low) / 2
        low, high = self.low_margin, self.high_margin
        # A max over unrelated constraints can have a long flat plateau.
        # Predict using the SAME violated constraint at both solved endpoints.
        pairs = [
            (self.low_margins[k], v)
            for k, v in (self.high_margins or {}).items()
            if k in (self.low_margins or {})
            and v is not None
            and math.isfinite(v)
            and v > 0
            and self.low_margins[k] is not None
            and math.isfinite(self.low_margins[k])
            and self.low_margins[k] <= 0
        ]
        if pairs:
            # First predicted crossing constrains a jointly feasible point.
            low, high = min(pairs, key=lambda p: -p[0] / (p[1] - p[0]))
        if low is None or high is None:
            self.slow_updates = 0
            return middle
        ratio = -low / (high - low)
        candidate = self.low + ratio * (self.high - self.low)
        # Probe within one tolerance of an endpoint instead of repeatedly
        # revisiting a nearly identical secant root. Width still certifies stop.
        candidate = max(
            self.low + self.tolerance / 2, min(self.high - self.tolerance / 2, candidate)
        )
        if self.slow_updates >= 2:
            # One bounded exception to bisection per search: a solved point
            # within the termination tolerance may close the bracket directly.
            # If it does not, the next stalled proposal still bisects.
            near_endpoint = min(candidate-self.low, self.high-candidate) <= self.tolerance
            if allow_endpoint_probe and not self.endpoint_probe_used and near_endpoint:
                self.endpoint_probe_used = True
            else:
                self.slow_updates = 0
                return middle
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

    def tangent_proposal(self, derivatives, high_derivatives=None):
        """Use the closer facet endpoint; physical solve still mandatory."""
        if self.complete or self.slow_updates >= 2:
            return None
        candidates = []
        for key, high in (self.high_margins or {}).items():
            low = (self.low_margins or {}).get(key)
            from_high = abs(high or 0) < abs(low or 0) and key in (high_derivatives or {})
            slope = ((high_derivatives or {}) if from_high else (derivatives or {})).get(key)
            if (high is None or low is None or slope is None
                    or not all(math.isfinite(x) for x in (high, low, slope))
                    or high <= 0 or low > 0 or slope <= 0):
                continue
            root = self.high - high / slope if from_high else self.low - low / slope
            if self.low <= root < self.high:
                candidates.append(root)
        if not candidates:
            return None
        root = min(candidates)
        return max(self.low + self.tolerance/2, min(self.high-self.tolerance/2, root))


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
