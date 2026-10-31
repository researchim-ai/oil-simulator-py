"""Bounded production curtailment against steady facility constraints.

Uniformly scales the requested feed vector. A sampled contiguous feasible
interval and a local feasible/infeasible bracket are required. This is neither
well nodal analysis nor a global optimizer for nonmonotone disconnected domains.
Numerical solver failures and configuration errors are never capacity limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    InvalidPressureError,
    PropertyModelError,
    _number,
)


@dataclass(frozen=True)
class FacilityCapacityControl:
    min_fraction: float = 0.05
    max_fraction: float = 1.0
    fraction_tolerance: float = 1e-5
    scan_points: int = 17
    max_evaluations: int = 80

    def __post_init__(self):
        for name in ("min_fraction", "max_fraction", "fraction_tolerance"):
            _number(getattr(self, name), name, positive=True)
        if not self.min_fraction < self.max_fraction <= 1:
            raise ValueError("Capacity fractions require 0 < min < max <= 1")
        for name, minimum in [("scan_points", 3), ("max_evaluations", 3)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.max_evaluations < self.scan_points:
            raise ValueError("Evaluation budget must cover the scan")

    def solve(self, facility, initial=None, feed_scales=None, **controls):
        requested = (
            tuple(1.0 for _ in facility.feeds) if feed_scales is None else tuple(feed_scales)
        )
        if len(requested) != len(facility.feeds):
            raise ValueError("Requested feeds must match facility")
        for value in requested:
            if not 0 <= _number(value, "requested feed scale", nonnegative=True) <= 1:
                raise ValueError("Requested feed scales must be in [0, 1]")
        cache = {}
        trace = []
        guess = initial

        def evaluate(fraction):
            nonlocal guess
            if fraction in cache:
                return cache[fraction]
            if len(trace) >= self.max_evaluations:
                raise HydraulicConvergenceError("Facility capacity evaluation limit")
            scales = [value * fraction for value in requested]
            cold_retry = False
            try:
                try:
                    result = facility.run(initial=guess, feed_scales=scales, **controls)
                except HydraulicConvergenceError:
                    if guess is None:
                        raise
                    cold_retry = True
                    result = facility.run(feed_scales=scales, **controls)
                status = result["feasible"]
                reason = [
                    {
                        "name": v["name"],
                        "metric": v["metric"],
                        "violation": v["violation"],
                        "unit": v["unit"],
                    }
                    for v in result["violations"]
                ]
                # Local import avoids making the physical plant depend on control.
                guess = facility.network_initial(result)
                row = {
                    "fraction": fraction,
                    "feasible": status,
                    "reason": reason,
                    "cold_retry": cold_retry,
                }
            except PropertyModelError:
                raise
            except (EquipmentOperatingError, InvalidPressureError) as error:
                result = None
                status = False
                row = {
                    "fraction": fraction,
                    "feasible": False,
                    "reason": [{"physical_domain": str(error)}],
                    "cold_retry": cold_retry,
                }
            trace.append(row)
            cache[fraction] = (status, result, row)
            return cache[fraction]

        def finish(fraction, result, status, upper=None):
            requested_streams = [s.split(value)[0] for s, value in zip(facility.feeds, requested)]
            from .process import ComponentStream

            potential = ComponentStream.mix(requested_streams)
            actual = None if result is None else result["feed"]
            if actual is not None and any(
                actual[name] > value + 1e-12 * max(1.0, value)
                for name, value in asdict(potential).items()
            ):
                raise ConservationError("Accepted feed exceeds available requested feed")
            curtailed = (
                None
                if actual is None
                else {
                    name: max(0.0, value - actual[name])
                    for name, value in asdict(potential).items()
                }
            )
            return {
                "status": status,
                "fraction": fraction,
                "upper_infeasible_fraction": upper,
                "fraction_tolerance": self.fraction_tolerance,
                "requested_feed_scales": list(requested),
                "accepted_feed_scales": None
                if fraction is None
                else [v * fraction for v in requested],
                "curtailed_standard_rates_m3_s": curtailed,
                "evaluations": len(trace),
                "limiting_constraints": [] if upper is None else cache[upper][2]["reason"],
                "trace": trace,
                "facility": result,
                "scope": "uniform_feed_curtailment_contiguous_feasible_interval",
            }

        feasible, result, _ = evaluate(self.max_fraction)
        if feasible:
            return finish(
                self.max_fraction, result, "requested" if self.max_fraction == 1 else "max_allowed"
            )
        grid = [
            self.min_fraction + (self.max_fraction - self.min_fraction) * i / (self.scan_points - 1)
            for i in range(self.scan_points - 1)
        ] + [self.max_fraction]
        states = [evaluate(value)[0] for value in grid]
        indices = [i for i, ok in enumerate(states) if ok]
        if not indices:
            return finish(None, None, "no_feasible_sample")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise EquipmentOperatingError(
                "Disconnected sampled feasible regions require a different dispatch optimizer"
            )
        index = indices[-1]
        low, high = grid[index], grid[index + 1]
        result = cache[low][1]
        while high - low > self.fraction_tolerance:
            mid = (low + high) / 2
            feasible, trial, _ = evaluate(mid)
            if feasible:
                low, result = mid, trial
            else:
                high = mid
        return finish(low, result, "limited", high)
