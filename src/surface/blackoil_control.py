"""Steady pressure regulation by one controllable black-oil branch resistance.

The implicit tangent includes opening response. No actuator dynamics, general gas-liquid
choke correlation, zero-flow holdup or multiphase pump model is implied.
"""

import math
from dataclasses import asdict, dataclass

from .liquid import HydraulicConvergenceError, _name, _number


@dataclass(frozen=True)
class BlackOilPressureControl:
    valve: str
    node: str
    target_pressure_pa: float
    min_opening: float = 0.2
    max_opening: float = 1.0
    pressure_tolerance_pa: float = 1e-4
    max_evaluations: int = 50

    def __post_init__(self):
        _name(self.valve)
        _name(self.node)
        for key in ("target_pressure_pa", "min_opening", "max_opening", "pressure_tolerance_pa"):
            _number(getattr(self, key), key, positive=True)
        if not self.min_opening < self.max_opening <= 1:
            raise ValueError("Require 0 < min_opening < max_opening <= 1")
        if (
            isinstance(self.max_evaluations, bool)
            or not isinstance(self.max_evaluations, int)
            or self.max_evaluations < 3
        ):
            raise ValueError("max_evaluations must be an integer >= 3")

    def validate_network(self, network):
        controlled = {p.name for p in network.edges if p.loss_coefficient > 0}
        if self.valve not in controlled:
            raise ValueError("Unknown controllable production restriction")
        if self.node not in network.by_name or self.node == network.root:
            raise ValueError("Pressure regulator requires a free production node")
        # Only upstream nodes respond to this branch in a directed tree.
        node = self.node
        while node != network.root:
            pipe = network.outgoing[node]
            if pipe.name == self.valve:
                return
            node = pipe.end
        raise ValueError("Regulated node must be upstream of the controlled restriction")

    def solve(self, network, sources):
        self.validate_network(network)
        evaluations = 0

        def evaluate(opening):
            nonlocal evaluations
            if evaluations >= self.max_evaluations:
                raise HydraulicConvergenceError(
                    "Black-oil pressure regulator exhausted evaluation budget"
                )
            evaluations += 1
            result = network.with_controls({self.valve: float(opening)}).solve(sources)
            error = result["node_pressures_pa"][self.node] - self.target_pressure_pa
            return error, result

        low, high = self.min_opening, self.max_opening
        fl, rl = evaluate(low)
        fh, rh = evaluate(high)
        if abs(fl) <= self.pressure_tolerance_pa:
            opening, error, result, status = low, fl, rl, "min_opening"
        elif abs(fh) <= self.pressure_tolerance_pa:
            opening, error, result, status = high, fh, rh, "max_opening"
        elif fl * fh > 0:
            opening, error, result, status = (
                (low, fl, rl, "min_opening") if abs(fl) < abs(fh) else (high, fh, rh, "max_opening")
            )
        else:
            opening, error, result = (low, fl, rl) if abs(fl) < abs(fh) else (high, fh, rh)
            while abs(error) > self.pressure_tolerance_pa:
                slope = result["pressure_derivatives_valve_opening"][self.node][self.valve]
                proposal = (
                    opening - error / slope if math.isfinite(slope) and slope != 0 else math.nan
                )
                if not low < proposal < high:
                    proposal = (low + high) / 2
                opening = proposal
                error, result = evaluate(opening)
                if fl * error > 0:
                    low, fl = opening, error
                else:
                    high = opening
            status = "target"

        if status == "target":
            # Differentiate p_control(q, opening) = target. Use copies before
            # updating rows, since the regulated row participates in every row.
            row = result["pressure_derivatives_per_kg_s"][self.node]
            dp = {source: dict(components) for source, components in row.items()}
            db = result["pressure_derivatives_receiving_pressure"][self.node]
            slope = result["pressure_derivatives_valve_opening"][self.node][self.valve]
            if not math.isfinite(slope) or abs(slope) < 1e-12:
                raise HydraulicConvergenceError("Singular black-oil pressure-control tangent")
            control_row = dict(result["pressure_derivatives_valve_opening"][self.node])
            result["opening_derivatives_per_kg_s"] = {
                source: {c: -value / slope for c, value in components.items()}
                for source, components in dp.items()
            }
            result["opening_derivative_receiving_pressure"] = -db / slope
            adjusted_openings = {}
            for node in network.order:
                weight = result["pressure_derivatives_valve_opening"][node][self.valve] / slope
                for source, components in result["pressure_derivatives_per_kg_s"][node].items():
                    for component in components:
                        components[component] -= weight * dp[source][component]
                result["pressure_derivatives_receiving_pressure"][node] -= weight * db
                adjusted_openings[node] = {
                    valve: value - weight * control_row[valve]
                    for valve, value in result["pressure_derivatives_valve_opening"][node].items()
                    if valve != self.valve
                }
            result["fixed_opening_pressure_derivatives"] = result[
                "pressure_derivatives_valve_opening"
            ]
            result["pressure_derivatives_valve_opening"] = adjusted_openings
        result["pressure_control"] = {
            **asdict(self),
            "status": status,
            "opening": float(opening),
            "pressure_error_pa": float(error),
            "evaluations": evaluations,
        }
        return result
