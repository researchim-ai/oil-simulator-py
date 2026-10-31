"""Bounded steady pressure regulation for a pump or common-speed pump group."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .liquid import (
    EquipmentOperatingError,
    HydraulicConvergenceError,
    NetworkResult,
    _name,
    _number,
)


@dataclass(frozen=True)
class PressureControlResult:
    network: NetworkResult
    speed_ratio: float
    status: str
    pressure_error_pa: float
    evaluations: int

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PumpPressureControl:
    """One free-node pressure target and bounded positive common speed.

    Requires a continuous monotone response over the bracket. Decreasing
    suction-pressure response can be selected explicitly or inferred at ends.
    Equipment safety errors propagate; limits are never silently relaxed.
    A saturated speed returns an explicit status and the unmet pressure error.
    """

    pump: str
    node: str
    target_pressure_pa: float
    min_speed_ratio: float
    max_speed_ratio: float
    pressure_tolerance_pa: float = 1.0
    max_evaluations: int = 50
    method: str = "bisect"
    response_direction: str = "increasing"
    linked_pumps: tuple = ()

    def __post_init__(self):
        _name(self.pump)
        _name(self.node)
        linked = tuple(self.linked_pumps)
        if isinstance(self.linked_pumps, str) or len({self.pump, *linked}) != 1 + len(linked):
            raise ValueError("Linked pump names must be distinct")
        for name in linked:
            _name(name)
        object.__setattr__(self, "linked_pumps", linked)
        for name in (
            "target_pressure_pa",
            "min_speed_ratio",
            "max_speed_ratio",
            "pressure_tolerance_pa",
        ):
            _number(getattr(self, name), name, positive=True)
        if self.min_speed_ratio >= self.max_speed_ratio:
            raise ValueError("Pressure control requires min_speed_ratio < max_speed_ratio")
        if self.method not in {"bisect", "newton"}:
            raise ValueError("Pressure control method must be bisect or newton")
        if self.response_direction not in {"increasing", "decreasing", "auto"}:
            raise ValueError("Unknown pressure-control response direction")
        if (
            isinstance(self.max_evaluations, bool)
            or not isinstance(self.max_evaluations, int)
            or self.max_evaluations < 3
        ):
            raise ValueError("max_evaluations must be an integer >= 3")

    def solve(self, network, initial=None, **network_options):
        pumps = {p.name: p for p in network.pumps}
        nodes = {n.name: n for n in network.nodes}
        pump_names = (self.pump, *self.linked_pumps)
        if set(pump_names) - set(pumps) or self.node not in nodes:
            raise ValueError("Unknown pump or pressure-control node")
        if any(not pumps[name].open for name in pump_names):
            raise ValueError("Cannot regulate a manually closed pump")
        boundaries = network_options.get("boundary_pressures_pa") or {}
        if boundaries.get(self.node, nodes[self.node].pressure_pa) is not None:
            raise ValueError("Pressure-control target node must not have a fixed pressure")
        if "pump_speed_ratios" in network_options or "pump_status" in network_options:
            raise ValueError("Pressure regulator owns pump controls for this solve")
        count = 0

        def evaluate(speed, guess):
            nonlocal count
            count += 1
            result = network.solve(
                initial=guess, pump_speed_ratios={name: speed for name in pump_names}, **network_options
            )
            error = result.pressures_pa[self.node] - self.target_pressure_pa
            return result, error

        def finish(result, speed, error, status):
            return PressureControlResult(result, speed, status, direction * error, count)

        low, high = self.min_speed_ratio, self.max_speed_ratio
        lo, elo = evaluate(low, initial)
        hi, ehi = evaluate(high, lo)
        tol = self.pressure_tolerance_pa
        direction = (-1.0 if self.response_direction == "decreasing" else
                     -1.0 if self.response_direction == "auto" and ehi < elo else 1.0)
        elo, ehi = direction * elo, direction * ehi
        if ehi < elo - tol:
            raise EquipmentOperatingError("Pressure response decreases with pump speed")
        if abs(ehi - elo) <= tol and abs(elo) <= tol:
            return finish(lo, low, elo, "min_speed")
        if abs(elo) <= tol:
            return finish(lo, low, elo, "target")
        if abs(ehi) <= tol:
            return finish(hi, high, ehi, "target")
        if elo > 0:
            return finish(lo, low, elo, "min_speed")
        if ehi < 0:
            return finish(hi, high, ehi, "max_speed")
        while count < self.max_evaluations:
            mid = (low + high) / 2
            if self.method == "newton":
                speed, state, error = (low, lo, elo) if abs(elo) <= abs(ehi) else (high, hi, ehi)
                slope = direction * network.pump_speed_sensitivities(
                    state, pump_names, [self.node]).sum()
                if slope > 0:
                    proposed = speed - error / slope
                    if low < proposed < high:
                        mid = float(proposed)
            result, error = evaluate(mid, lo)
            error *= direction
            if error < elo - tol or error > ehi + tol:
                raise EquipmentOperatingError("Nonmonotone pressure-control response")
            if abs(error) <= tol:
                return finish(result, mid, error, "target")
            if error < 0:
                low, lo, elo = mid, result, error
            else:
                high, hi, ehi = mid, result, error
        raise HydraulicConvergenceError("Pressure regulator evaluation limit exceeded")

    def pressure_sensitivities(self, network, result, source_nodes, pressure_nodes):
        """Include implicit speed response at target; freeze speed at a bound."""
        if not isinstance(result, PressureControlResult):
            raise TypeError("Expected PressureControlResult")
        if result.status not in {"target", "min_speed", "max_speed"}:
            raise ValueError("Unknown pressure-control state")
        if result.status == "target":
            actual_error = result.network.pressures_pa[self.node] - self.target_pressure_pa
            if (abs(result.pressure_error_pa) > self.pressure_tolerance_pa
                    or abs(actual_error) > self.pressure_tolerance_pa):
                raise ValueError("Pressure-control result misses target")
            return network.pressure_sensitivities(
                result.network, source_nodes, pressure_nodes,
                regulated_pump=(self.pump, *self.linked_pumps), regulated_node=self.node)
        return network.pressure_sensitivities(result.network, source_nodes, pressure_nodes)
