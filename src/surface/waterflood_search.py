"""Pressure-resolved waterflood brackets built from physically solved points."""

import math
from dataclasses import dataclass

from .capacity_bracket import CapacityBracket
from .liquid import HydraulicConvergenceError


@dataclass(frozen=True)
class WaterfloodDispatchControl:
    method: str = "safeguarded"
    commit: str = "state_restore"
    reuse_initial_cell_state: bool = True
    initial_cell_state_cache_bytes: int = 64 * 1024**2

    def __post_init__(self):
        if self.method not in ("safeguarded", "bisect"):
            raise ValueError("Waterflood dispatch method must be safeguarded or bisect")
        if self.commit not in ("state_restore", "replay"):
            raise ValueError("Waterflood commit must be state_restore or replay")
        if type(self.reuse_initial_cell_state) is not bool:
            raise ValueError("Initial-cell reuse must be boolean")
        if type(self.initial_cell_state_cache_bytes) is not int or self.initial_cell_state_cache_bytes <= 0:
            raise ValueError("Positive integer initial-cell cache budget required")


def pressure_span(a, b):
    left, right = a["well_bhp_pa"], b["well_bhp_pa"]
    return max((abs(left[k] - right[k]) for k in left.keys() & right.keys()), default=0.)


def _margin(data):
    values = data.get("constraint_margins")
    if not values or any(v is None or not math.isfinite(v) for v in values.values()):
        return None
    return max(values.values())


def resolve_boundary(evaluate, low, high, low_data, high_data, fraction_tolerance,
                     bhp_tolerance, method="safeguarded"):
    """Use existing facet/secant safeguards, with the original BHP stop test.

    Neither a secant nor a small margin certifies a solution. Both final
    endpoints have actually been solved at this same old state and duration.
    Unavailable facets cannot propose a root; unresolved constraints retain
    the existing bisection safeguards.
    """
    bracket = CapacityBracket(
        low, high, fraction_tolerance, _margin(low_data), _margin(high_data),
        low_margins=low_data.get("constraint_margins"),
        high_margins=high_data.get("constraint_margins"),
    )
    while (bracket.high - bracket.low > fraction_tolerance
           or pressure_span(low_data, high_data) > bhp_tolerance):
        width = bracket.high - bracket.low
        span = pressure_span(low_data, high_data)
        # This tolerance only positions the next proposal. Actual endpoint
        # BHP span, not this secant estimate, still determines acceptance.
        bracket.tolerance = min(fraction_tolerance, .25 * width)
        if span > bhp_tolerance:
            bracket.tolerance = min(bracket.tolerance, .25 * width * bhp_tolerance / span)
        point = (bracket.low + bracket.high) / 2 if method == "bisect" else bracket.proposal(
            allow_endpoint_probe=True
        )
        if not bracket.low < point < bracket.high:
            raise HydraulicConvergenceError("Waterflood bracket has no representable interior point")
        feasible, data = evaluate(point)
        bracket.update(point, feasible, _margin(data), margins=data.get("constraint_margins"))
        if feasible:
            low_data = data
        else:
            high_data = data
    return bracket.low, bracket.high, low_data, high_data
