"""Latched threshold trips with conservative event-time replay.

Soft trips act strictly inside hard model limits. No SIL certification,
continuous-time guarantee between sampled substeps, or automatic relief sizing.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .dynamic_vessel import VesselState
from .liquid import HydraulicConvergenceError, PropertyModelError, _name, _number


@dataclass(frozen=True)
class NetworkTrip:
    name: str
    vessel: str
    metric: str
    direction: str
    threshold: float
    value_tolerance: float
    close_connections: tuple = ()
    stop_feed_vessels: tuple = ()
    stop_heat_vessels: tuple = ()
    stop_heating_vessels: tuple = ()
    stop_cooling_vessels: tuple = ()
    stop_withdrawal_vessels: tuple = ()

    def __post_init__(self):
        _name(self.name)
        _name(self.vessel)
        if self.metric not in (
            "pressure_pa",
            "temperature_k",
            "wall_temperature_k",
            "liquid_volume_fraction",
            "liquid_level_m",
            "total_moles",
        ):
            raise ValueError("Unknown trip metric")
        if self.direction not in ("above", "below"):
            raise ValueError("Trip direction must be above or below")
        _number(self.threshold, "threshold", positive=True)
        _number(self.value_tolerance, "value_tolerance", positive=True)
        for key in (
            "close_connections",
            "stop_feed_vessels",
            "stop_heat_vessels",
            "stop_heating_vessels",
            "stop_cooling_vessels",
            "stop_withdrawal_vessels",
        ):
            seq = getattr(self, key)
            if not isinstance(seq, (tuple, list)) or len(set(seq)) != len(seq):
                raise ValueError("Trip actions must contain unique names")
            for name in seq:
                _name(name)
            object.__setattr__(self, key, tuple(sorted(seq)))

    def validate(self, network, fluid, require_action=False):
        vessels = dict(network.vessels)
        if self.vessel not in vessels:
            raise ValueError("Trip references an unknown vessel")
        if set(self.close_connections) - {c.name for c in network.connections} or (
            set(self.stop_feed_vessels)
            | set(self.stop_heat_vessels)
            | set(self.stop_heating_vessels)
            | set(self.stop_cooling_vessels)
            | set(self.stop_withdrawal_vessels)
        ) - set(vessels):
            raise ValueError("Trip action references unknown equipment")
        if require_action and not any(
            (
                self.close_connections,
                self.stop_feed_vessels,
                self.stop_heat_vessels,
                self.stop_heating_vessels,
                self.stop_cooling_vessels,
                self.stop_withdrawal_vessels,
            )
        ):
            raise ValueError("Latched protection requires an explicit shutdown action")
        v = vessels[self.vessel]
        if self.metric == "pressure_pa":
            lo, hi = v.min_pressure_pa, v.max_pressure_pa
        elif self.metric == "wall_temperature_k":
            wall = dict(network.walls).get(self.vessel)
            if wall is None:
                raise ValueError("Wall-temperature trip needs a configured wall")
            lo, hi = wall.min_temperature_k, wall.max_temperature_k
        elif self.metric == "temperature_k":
            lo, hi = (
                max(c.min_temperature_k for c in fluid.components),
                min(c.max_temperature_k for c in fluid.components),
            )
        elif self.metric == "liquid_level_m":
            geometry = dict(network.geometries).get(self.vessel)
            if geometry is None:
                raise ValueError("Liquid level trip needs separator geometry")
            lo, hi = 0.0, geometry.height_m
        elif self.metric == "liquid_volume_fraction":
            lo = v.min_liquid_volume_fraction if v.min_liquid_volume_fraction is not None else 0.0
            hi = v.max_liquid_volume_fraction if v.max_liquid_volume_fraction is not None else 1.0
        else:
            lo, hi = len(fluid.components) * v.minimum_component_moles, math.inf
        if not lo + self.value_tolerance < self.threshold < hi - self.value_tolerance:
            raise ValueError("Trip threshold and tolerance must lie strictly inside hard limits")

    def value(self, states, network, fluid):
        s = states[self.vessel]
        if self.metric in ("pressure_pa", "temperature_k", "wall_temperature_k"):
            return getattr(s, self.metric)
        if self.metric == "liquid_level_m":
            q = dict(network.vessels)[self.vessel].inspect(s, fluid)["equilibrium"]
            return dict(network.geometries)[self.vessel].from_equilibrium(s.component_moles, q)
        if self.metric == "total_moles":
            return math.fsum(s.component_moles)
        value = dict(network.vessels)[self.vessel].inspect(s, fluid)["liquid_volume_fraction"]
        if value is None:
            raise PropertyModelError("Liquid-volume trip requires resolved liquid/vapor phases")
        return value

    def gap(self, states, network, fluid):
        value = self.value(states, network, fluid)
        return self.threshold - value if self.direction == "above" else value - self.threshold

    def record(self, states, network, fluid, initial=False):
        return {
            "name": self.name,
            "vessel": self.vessel,
            "metric": self.metric,
            "direction": self.direction,
            "threshold": self.threshold,
            "value": self.value(states, network, fluid),
            "time_s": states[self.vessel].time_s,
            "initially_triggered": initial,
        }


def locate_network_event(
    network,
    fluid,
    current,
    new,
    record,
    step,
    feeds,
    withdrawals,
    heat,
    factors,
    rules,
    time_tolerance_s,
    ambient_temperatures=None,
    actuator_positions=None,
    actuator_modes=None,
):
    """Replay one implicit step from the SAME state for each time probe."""
    crossed = [
        r for r in rules if r.gap(current, network, fluid) > 0 and r.gap(new, network, fluid) <= 0
    ]
    if not crossed:
        return None
    cache = {0.0: (current, None), step: (new, record)}

    def state_at(dt):
        dt = float(dt)
        if dt not in cache:
            cache[dt] = network._step(
                current,
                fluid,
                dt,
                feeds,
                withdrawals,
                heat,
                factors,
                ambient_temperatures=ambient_temperatures,
                actuator_positions=actuator_positions,
                actuator_modes=actuator_modes,
            )
        return cache[dt]

    roots = []
    for rule in crossed:
        try:
            root = brentq(
                lambda t, rule=rule: rule.gap(state_at(t)[0], network, fluid),
                0.0,
                step,
                xtol=time_tolerance_s * 0.1,
                rtol=1e-13,
                maxiter=80,
            )
        except HydraulicConvergenceError:
            raise
        except RuntimeError as error:
            raise HydraulicConvergenceError("Event time root failed to converge") from error
        states, _ = state_at(root)
        gap = rule.gap(states, network, fluid)
        if abs(gap) > rule.value_tolerance:
            raise HydraulicConvergenceError("Event time root fails metric tolerance")
        roots.append((root, rule))
    root, selected = min(roots, key=lambda item: (item[0], item[1].name))
    states, step_record = state_at(root)
    triggered = [
        rule
        for time, rule in roots
        if abs(time - root) <= time_tolerance_s
        and abs(rule.gap(states, network, fluid)) <= rule.value_tolerance
    ]
    if selected not in triggered:
        triggered.append(selected)
    return root, states, step_record, sorted(triggered, key=lambda rule: rule.name)


def advance_protected_network(
    network,
    states,
    fluid,
    duration_s,
    rules,
    latched=(),
    *,
    feeds=None,
    withdrawals=None,
    heat=None,
    connection_factors=None,
    time_tolerance_s=1e-7,
    ambient_temperatures=None,
    actuator_positions=None,
    actuator_modes=None,
    **stepper,
):
    """Latch shutdown actions, then finish the exact requested interval."""
    _number(time_tolerance_s, "time_tolerance_s", positive=True)
    requested = _number(duration_s, "duration_s", positive=True)
    rules = tuple(rules)
    by_name = {rule.name: rule for rule in rules}
    if len(by_name) != len(rules) or set(latched) - set(by_name):
        raise ValueError("Duplicate or unknown protection latch")
    for rule in rules:
        rule.validate(network, fluid, require_action=True)
    current = dict(states)
    positions, modes = network.inspect_actuators(actuator_positions, actuator_modes)
    tripped = set(latched)
    elapsed = 0.0
    steps = []
    events = []
    failures = 0
    blocked = np.zeros(len(fluid.components))
    vessels = dict(network.vessels)
    if set(current) != set(vessels) or len({s.time_s for s in current.values()}) != 1:
        raise ValueError("Protected network states/clocks must match vessels")
    for name, vessel in network.vessels:
        vessel.inspect(current[name], fluid)
    budget = stepper.get("max_substeps", 10000)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("max_substeps must be a positive integer")
    while elapsed < requested:
        # Apply an already-active trip before checking flows that it shuts down
        # (e.g. an initially missing phase outlet or insufficient feed pressure).
        for rule in sorted(rules, key=lambda r: r.name):
            if rule.name not in tripped and rule.gap(current, network, fluid) <= 0:
                tripped.add(rule.name)
                events.append(rule.record(current, network, fluid, initial=True))
        active = [r for r in rules if r.name not in tripped]
        active_feeds = dict(feeds or {})
        active_withdrawals = dict(withdrawals or {})
        active_heat = dict(heat or {})
        factors = dict(connection_factors or {})
        for name in sorted(tripped):
            rule = by_name[name]
            for c in rule.close_connections:
                factors[c] = 0.0
            for node in rule.stop_feed_vessels:
                active_feeds.pop(node, None)
            for node in rule.stop_withdrawal_vessels:
                active_withdrawals.pop(node, None)
            for node in rule.stop_heat_vessels:
                active_heat.pop(node, None)
            for node in rule.stop_heating_vessels:
                if active_heat.get(node, {}).get("heat_w", 0.0) > 0:
                    active_heat.pop(node, None)
            for node in rule.stop_cooling_vessels:
                if active_heat.get(node, {}).get("heat_w", 0.0) < 0:
                    active_heat.pop(node, None)
        if len(steps) >= budget:
            raise HydraulicConvergenceError("Protected interval substep budget exhausted")
        local_stepper = {**stepper, "max_substeps": budget - len(steps)}
        result = network.advance(
            current,
            fluid,
            requested - elapsed,
            feeds=active_feeds,
            withdrawals=active_withdrawals,
            heat=active_heat,
            connection_factors=factors,
            stop_conditions=active,
            event_time_tolerance_s=time_tolerance_s,
            ambient_temperatures=ambient_temperatures,
            actuator_positions=positions,
            actuator_modes=modes,
            **local_stepper,
        )
        dt = result["duration_s"]
        for node, incoming in (feeds or {}).items():
            if node not in active_feeds:
                for stream in incoming:
                    blocked += dt * np.array(stream.component_molar_rates_mol_s)
        steps.extend(result["steps"])
        failures += result["failed_steps"]
        current = {name: VesselState(**s) for name, s in result["states"].items()}
        positions = result["actuator_positions"]
        detected = result.get("events", [])
        if not dt and not detected:
            raise HydraulicConvergenceError("Protection loop made no time or latch progress")
        elapsed = math.fsum([elapsed, dt])
        for event in detected:
            if event["name"] in tripped:
                raise HydraulicConvergenceError("Already latched trip fired again")
            tripped.add(event["name"])
            events.append(event)
        if len(events) > len(rules):
            raise HydraulicConvergenceError("Protection event budget exhausted")
    return {
        "states": {name: s.to_dict() for name, s in current.items()},
        "steps": steps,
        "failed_steps": failures,
        "duration_s": requested,
        "events": events,
        "latched_trips": sorted(tripped),
        "blocked_feed_component_moles": blocked.tolist(),
        "actuator_positions": positions,
    }
