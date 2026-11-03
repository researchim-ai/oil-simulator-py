"""Sampled, delayed and latched liquid trips above a persistent PID clock.

Trips remove drive torque, never mass or stored energy. Relief flow is an
independent physical branch, not a numerical pressure clamp. No SIL claim or
guarantee of detecting excursions between protection samples.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .controlled_liquid import ControlledLiquidState, ControlledLiquidStation
from .liquid import HydraulicConvergenceError, _name, _number
from .liquid_relief import inspect_liquid_pumps
from .operation import _digest


@dataclass(frozen=True)
class LiquidTrip:
    name: str
    equipment: str
    metric: str
    direction: str
    threshold: float
    stop_pumps: tuple = ()
    trip_station: bool = False
    delay_s: float = 0.0
    startup_grace_s: float = 0.0
    min_speed_ratio: float = 0.6
    reset_margin: float = 0.0

    def __post_init__(self):
        _name(self.name)
        _name(self.equipment)
        if self.metric not in (
            "pressure_pa",
            "pump_flow_m3_s",
            "npsh_margin_m",
        ) or self.direction not in ("above", "below"):
            raise ValueError("Unknown liquid trip metric/direction")
        _number(self.threshold, "threshold", positive=True)
        for k in ("delay_s", "startup_grace_s", "min_speed_ratio", "reset_margin"):
            _number(getattr(self, k), k, nonnegative=True)
        pumps = tuple(self.stop_pumps)
        if len(pumps) != len(set(pumps)):
            raise ValueError("Duplicate liquid trip action")
        for n in pumps:
            _name(n)
        object.__setattr__(self, "stop_pumps", pumps)
        if not isinstance(self.trip_station, bool) or not (self.trip_station or pumps):
            raise ValueError("Trip requires an explicit drive shutdown action")

    def violating(self, value):
        return value >= self.threshold if self.direction == "above" else value <= self.threshold

    def reset_safe(self, value):
        return (
            value < self.threshold - self.reset_margin
            if self.direction == "above"
            else value > self.threshold + self.reset_margin
        )


@dataclass(frozen=True)
class LiquidTripMemory:
    armed_since_s: float | None = None
    violating_since_s: float | None = None
    tripped_at_s: float | None = None

    def __post_init__(self):
        for value in (self.armed_since_s, self.violating_since_s, self.tripped_at_s):
            if value is not None:
                _number(value, "trip_memory_time_s", nonnegative=True)


@dataclass(frozen=True)
class ProtectedLiquidState:
    controlled: ControlledLiquidState
    trips: tuple
    external_unavailable: tuple
    operator_tripped: bool
    clock_origin_s: float
    next_tick: int
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "trips", tuple(sorted(dict(self.trips).items())))
        object.__setattr__(self, "external_unavailable", tuple(self.external_unavailable))
        if len(self.external_unavailable) != len(set(self.external_unavailable)):
            raise ValueError("Duplicate external unavailability")
        _number(self.clock_origin_s, "clock_origin_s", nonnegative=True)
        if (
            not isinstance(self.operator_tripped, bool)
            or isinstance(self.next_tick, bool)
            or not isinstance(self.next_tick, int)
            or self.next_tick < 0
        ):
            raise ValueError("Invalid protection clock/operator latch")

    def to_dict(self):
        return {
            "controlled": self.controlled.to_dict(),
            "trips": {n: asdict(m) for n, m in self.trips},
            "external_unavailable": list(self.external_unavailable),
            "operator_tripped": self.operator_tripped,
            "clock_origin_s": self.clock_origin_s,
            "next_tick": self.next_tick,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["controlled"] = ControlledLiquidState.from_dict(data["controlled"])
        data["trips"] = {n: LiquidTripMemory(**v) for n, v in data["trips"].items()}
        return cls(**data)


class ProtectedLiquidStation:
    def __init__(self, controller, rules, protection_interval_s=0.01):
        if not isinstance(controller, ControlledLiquidStation):
            raise TypeError("Protection requires ControlledLiquidStation")
        self.controller = controller
        self.rules = tuple(r if isinstance(r, LiquidTrip) else LiquidTrip(**r) for r in rules)
        if not self.rules or len({r.name for r in self.rules}) != len(self.rules):
            raise ValueError("Nonempty distinct protection rules required")
        self.protection_interval_s = _number(
            protection_interval_s, "protection_interval_s", positive=True
        )
        physics = controller.physics
        for rule in self.rules:
            if set(rule.stop_pumps) - set(controller.names):
                raise ValueError("Unknown pump shutdown action")
            if rule.metric == "pressure_pa":
                if rule.equipment not in physics._nodes:
                    raise ValueError("Unknown pressure instrument")
                lo = max(
                    physics.network.fluid.vapor_pressure_pa,
                    physics.reference_pressure_pa
                    - physics.bulk_modulus_pa * physics.max_compression_fraction,
                )
                hi = min(
                    physics.max_pressure_pa,
                    physics.reference_pressure_pa
                    + physics.bulk_modulus_pa * physics.max_compression_fraction,
                )
                if not lo < rule.threshold < hi:
                    raise ValueError(
                        "Soft pressure trip must lie strictly inside hard model domain"
                    )
            else:
                if rule.equipment not in controller.names:
                    raise ValueError("Unknown pump instrument")
                pump = next(p for p in physics.network.pumps if p.name == rule.equipment)
                if rule.metric == "npsh_margin_m" and (
                    pump.inlet_diameter_m is None
                    or (pump.npsh_required_m is None and pump.npsh_curve is None)
                ):
                    raise ValueError(
                        "NPSH margin protection requires inlet diameter and required NPSH"
                    )

    @property
    def model_id(self):
        return _digest({"model": "sampled_latched_liquid_trips_v1", **self.to_dict()})

    def to_dict(self):
        return {
            "controller": self.controller.to_dict(),
            "rules": [asdict(r) for r in self.rules],
            "protection_interval_s": self.protection_interval_s,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["controller"] = ControlledLiquidStation.from_dict(data["controller"])
        return cls(**data)

    def initialize(self, controller_initial=None):
        controlled = self.controller.initialize(**(controller_initial or {}))
        state = ProtectedLiquidState(
            controlled,
            {r.name: LiquidTripMemory() for r in self.rules},
            controlled.physical.unavailable,
            controlled.physical.tripped,
            controlled.physical.time_s,
            0,
            self.model_id,
        )
        self._check(state)
        return state

    def _check(self, state):
        if not isinstance(state, ProtectedLiquidState) or state.model_id != self.model_id:
            raise ValueError("Protection state/model mismatch")
        self.controller._check(state.controlled)
        now = state.controlled.physical.time_s
        next_time = state.clock_origin_s + state.next_tick * self.protection_interval_s
        tol = 16 * max(math.ulp(now), math.ulp(next_time))
        if (
            not math.isfinite(next_time)
            or state.clock_origin_s > now
            or next_time < now - tol
            or set(dict(state.trips)) != {r.name for r in self.rules}
            or set(state.external_unavailable) - set(self.controller.names)
        ):
            raise ValueError("Protection clock/rule memory mismatch")
        for _, memory in state.trips:
            if not isinstance(memory, LiquidTripMemory) or any(
                t is not None and t > now for t in asdict(memory).values()
            ):
                raise ValueError("Invalid protection timer history")
            if (
                memory.violating_since_s is not None
                and (
                    memory.armed_since_s is None or memory.violating_since_s < memory.armed_since_s
                )
                or memory.tripped_at_s is not None
                and (
                    memory.violating_since_s is None
                    or memory.tripped_at_s < memory.violating_since_s
                )
            ):
                raise ValueError("Protection timer order mismatch")
        latched = [r for r in self.rules if dict(state.trips)[r.name].tripped_at_s is not None]
        blocked = set(state.external_unavailable).union(*(set(r.stop_pumps) for r in latched))
        station_trip = state.operator_tripped or any(r.trip_station for r in latched)
        if (
            blocked != set(state.controlled.physical.unavailable)
            or station_trip != state.controlled.physical.tripped
        ):
            raise ValueError("Protection action/latch provenance mismatch")

    def _measure(self, current, rules, blocked, station_off, boundary_pressures_pa):
        physics = self.controller.physics
        boundary, _ = physics._controls(boundary_pressures_pa)
        pressures = {**boundary, **dict(current.physical.pressures_pa)}
        pumps = (
            inspect_liquid_pumps(physics, current.physical, boundary_pressures_pa)
            if any(r.metric != "pressure_pa" for r in rules)
            else {}
        )
        out = {}
        for rule in rules:
            if rule.metric == "pressure_pa":
                out[rule.name] = (pressures[rule.equipment], True)
            else:
                instrument = pumps[rule.equipment]
                armed = (
                    not station_off
                    and rule.equipment in current.selected
                    and rule.equipment not in blocked
                    and instrument["speed_ratio"] >= rule.min_speed_ratio
                )
                value = (
                    instrument["flow_m3_s"]
                    if rule.metric == "pump_flow_m3_s"
                    else instrument["npsh_margin_m"]
                )
                out[rule.name] = (value, armed and value is not None)
        return out

    def advance(
        self,
        state,
        duration_s,
        mode="auto",
        unavailable=None,
        reset_trip=False,
        reset_trips=(),
        max_protection_samples=100000,
        max_substeps=100000,
        max_control_samples=100000,
        **commands,
    ):
        self._check(state)
        duration = _number(duration_s, "duration_s", positive=True)
        end = state.controlled.physical.time_s + duration
        if not math.isfinite(end) or end <= state.controlled.physical.time_s:
            raise ValueError("Protection report cannot advance time")
        if (
            mode not in ("auto", "manual", "torque", "off", "trip")
            or not isinstance(reset_trip, bool)
            or (mode == "trip" and reset_trip)
        ):
            raise ValueError("Invalid protection mode/reset")
        for limit in (max_protection_samples, max_substeps, max_control_samples):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("Positive integer work budgets required")
        permitted = {
            "hydraulic_journal",
            "hydraulic_journal_branches",
            "_on_accepted_step",
            "hydraulic_state_transfer",
            "pressure_setpoint_pa",
            "speed_ratio",
            "running_pumps",
            "motor_torques_nm",
            "boundary_pressures_pa",
            "valve_openings",
            "actuator_modes",
            "redispatch",
        }
        if set(commands) - permitted:
            raise ValueError("Unknown protected controller command")
        external = state.external_unavailable if unavailable is None else tuple(unavailable)
        resets = tuple(reset_trips)
        if (
            set(external) - set(self.controller.names)
            or len(external) != len(set(external))
            or set(resets) - {r.name for r in self.rules}
            or len(resets) != len(set(resets))
        ):
            raise ValueError("Unknown/duplicate reset or external unavailable unit")
        current = state.controlled
        memories = dict(state.trips)
        operator = (state.operator_tripped and not reset_trip) or mode == "trip"
        events, measurements, control_samples, dispatches, steps = [], [], [], [], []
        retries = {}
        tick = state.next_tick

        def actions():
            latched = [r for r in self.rules if memories[r.name].tripped_at_s is not None]
            blocked = set(external).union(*(set(r.stop_pumps) for r in latched))
            return blocked, operator or any(r.trip_station for r in latched)

        if resets:
            blocked, station_trip = actions()
            values = self._measure(
                current,
                [r for r in self.rules if r.name in resets],
                blocked,
                station_trip or mode == "off",
                commands.get("boundary_pressures_pa"),
            )
            for rule in self.rules:
                if rule.name not in resets:
                    continue
                value, armed = values[rule.name]
                safe = not armed or rule.reset_safe(value)
                events.append(
                    {
                        "type": "reset" if safe else "reset_denied",
                        "name": rule.name,
                        "time_s": current.physical.time_s,
                        "value": value,
                    }
                )
                if safe:
                    memories[rule.name] = LiquidTripMemory()
        pending_redispatch = commands.pop("redispatch", False)
        if not isinstance(pending_redispatch, bool):
            raise TypeError("redispatch must be boolean")
        first = True
        while current.physical.time_s < end:
            now = current.physical.time_s
            next_time = state.clock_origin_s + tick * self.protection_interval_s
            tol = 4 * max(math.ulp(now), math.ulp(next_time), math.ulp(end))
            terminal = abs(now - next_time) <= tol and end - now <= tol
            blocked, station_trip = actions()
            if abs(now - next_time) <= tol and not terminal:
                if len(measurements) >= max_protection_samples:
                    raise HydraulicConvergenceError("Protection sample budget exhausted")
                values = self._measure(
                    current,
                    self.rules,
                    blocked,
                    station_trip or mode == "off",
                    commands.get("boundary_pressures_pa"),
                )
                sample = {"time_s": now, "tick": tick, "rules": {}}
                for rule in self.rules:
                    memory = memories[rule.name]
                    value, armed = values[rule.name]
                    if memory.tripped_at_s is None:
                        if not armed:
                            memory = LiquidTripMemory()
                        else:
                            since = now if memory.armed_since_s is None else memory.armed_since_s
                            qualified = now - since + tol >= rule.startup_grace_s
                            violating = qualified and rule.violating(value)
                            bad_since = (
                                (
                                    now
                                    if memory.violating_since_s is None
                                    else memory.violating_since_s
                                )
                                if violating
                                else None
                            )
                            trigger = (
                                bad_since is not None and now - bad_since + tol >= rule.delay_s
                            )
                            memory = LiquidTripMemory(since, bad_since, now if trigger else None)
                            if trigger:
                                events.append(
                                    {
                                        "type": "trip",
                                        "name": rule.name,
                                        "time_s": now,
                                        "value": value,
                                        "metric": rule.metric,
                                        "equipment": rule.equipment,
                                        "stop_pumps": list(rule.stop_pumps),
                                        "trip_station": rule.trip_station,
                                    }
                                )
                        memories[rule.name] = memory
                    sample["rules"][rule.name] = {"value": value, "armed": armed, **asdict(memory)}
                measurements.append(sample)
                tick += 1
                next_time = state.clock_origin_s + tick * self.protection_interval_s
                if next_time <= now:
                    raise ValueError("Protection clock cannot advance")
                blocked, station_trip = actions()
            until = end if terminal else min(end, next_time)
            if until <= now:
                raise ValueError("Protection clock skipped or regressed")
            budget = max_substeps - len(steps) - sum(retries.values())
            control_budget = max_control_samples - len(control_samples)
            if budget < 1:
                raise HydraulicConvergenceError("Protected station work budget exhausted")
            result = self.controller.advance(
                current,
                until - now,
                mode="trip" if station_trip else ("off" if mode == "trip" else mode),
                unavailable=tuple(n for n in self.controller.names if n in blocked),
                reset_trip=not station_trip and current.physical.tripped,
                redispatch=pending_redispatch if first else False,
                max_substeps=budget,
                max_control_samples=max(1, control_budget),
                **commands,
            )
            if len(result["control_samples"]) > control_budget:
                raise HydraulicConvergenceError("Protected station control sample budget exhausted")
            first = False
            for reason, count in result["retry_reasons"].items():
                retries[reason] = retries.get(reason, 0) + count
            steps.extend(result["steps"])
            control_samples.extend(result["control_samples"])
            dispatches.extend(result["dispatches"])
            current = ControlledLiquidState.from_dict(result["state"])
        final = ProtectedLiquidState(
            current, memories, external, operator, state.clock_origin_s, tick, self.model_id
        )
        self._check(final)
        return {
            "state": final.to_dict(),
            "properties": self.controller.physics.inspect(current.physical),
            "steps": steps,
            "control_samples": control_samples,
            "protection_samples": measurements,
            "dispatches": dispatches,
            "events": events,
            "retry_reasons": retries,
            "failed_steps": sum(retries.values()),
            "latched_trips": [
                r.name for r in self.rules if memories[r.name].tripped_at_s is not None
            ],
            "pressure_error_pa": result["pressure_error_pa"],
            "duration_s": duration,
        }
