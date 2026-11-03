"""Caloric liquid recycle with sampled temperature trips and bounded utilities.

Hydraulics retains its explicit constant-property acoustic approximation.
Temperature drives real shutdown/cooling commands, not artificial state clamps.
"""

import math
from dataclasses import asdict, dataclass, replace

from .liquid import HydraulicConvergenceError, _name, _number
from .liquid_heat import LiquidHeatNetwork, LiquidHeatState
from .liquid_protection import LiquidTripMemory
from .liquid_recycle import RecyclingLiquidState, RecyclingLiquidStation
from .operation import _digest
from .pid import PIDController, PIDState


@dataclass(frozen=True)
class LiquidTemperatureTrip:
    name: str
    location: str
    threshold_k: float
    direction: str = "above"
    metric: str = "fluid_temperature_k"
    delay_s: float = 0.0
    reset_margin_k: float = 0.0
    stop_pumps: tuple = ()
    trip_station: bool = False

    def __post_init__(self):
        _name(self.name)
        _name(self.location)
        for key in ("threshold_k", "delay_s", "reset_margin_k"):
            _number(
                getattr(self, key),
                key,
                positive=key == "threshold_k",
                nonnegative=key != "threshold_k",
            )
        if self.direction not in ("above", "below") or self.metric not in (
            "fluid_temperature_k",
            "wall_temperature_k",
        ):
            raise ValueError("Unknown temperature trip direction/metric")
        pumps = tuple(self.stop_pumps)
        if len(pumps) != len(set(pumps)):
            raise ValueError("Duplicate temperature trip pump")
        object.__setattr__(self, "stop_pumps", pumps)
        if not isinstance(self.trip_station, bool) or not (pumps or self.trip_station):
            raise ValueError("Temperature trip needs an explicit shutdown action")

    def violating(self, t):
        return t >= self.threshold_k if self.direction == "above" else t <= self.threshold_k

    def reset_safe(self, t):
        return (
            t < self.threshold_k - self.reset_margin_k
            if self.direction == "above"
            else t > self.threshold_k + self.reset_margin_k
        )


@dataclass(frozen=True)
class LiquidThermostat:
    name: str
    location: str
    setpoint_k: float
    controller: PIDController

    def __post_init__(self):
        _name(self.name)
        _name(self.location)
        _number(self.setpoint_k, "setpoint_k", positive=True)
        if not isinstance(self.controller, PIDController) or self.controller.direction != 1:
            raise ValueError("Thermostat PID requires positive direction and duty in W")


@dataclass(frozen=True)
class ThermalLiquidState:
    recycling: RecyclingLiquidState
    heat: LiquidHeatState
    trips: tuple
    pids: tuple
    held_duties_w: tuple
    external_unavailable: tuple
    operator_tripped: bool
    clock_origin_s: float
    next_tick: int
    model_id: str

    def __post_init__(self):
        for key in ("trips", "pids", "held_duties_w"):
            object.__setattr__(self, key, tuple(sorted(dict(getattr(self, key)).items())))
        object.__setattr__(self, "external_unavailable", tuple(self.external_unavailable))
        if len(self.external_unavailable) != len(set(self.external_unavailable)) or not isinstance(
            self.operator_tripped, bool
        ):
            raise ValueError("Invalid external thermal-station command memory")
        if (
            isinstance(self.next_tick, bool)
            or not isinstance(self.next_tick, int)
            or self.next_tick < 0
        ):
            raise ValueError("Invalid thermal clock")
        _number(self.clock_origin_s, "clock_origin_s", nonnegative=True)
        for _, v in self.held_duties_w:
            _number(v, "held_duty_w")

    def to_dict(self):
        return {
            "recycling": self.recycling.to_dict(),
            "heat": self.heat.to_dict(),
            "trips": {n: asdict(v) for n, v in self.trips},
            "pids": {n: v.to_dict() for n, v in self.pids},
            "held_duties_w": dict(self.held_duties_w),
            "external_unavailable": list(self.external_unavailable),
            "operator_tripped": self.operator_tripped,
            "clock_origin_s": self.clock_origin_s,
            "next_tick": self.next_tick,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["recycling"] = RecyclingLiquidState.from_dict(data["recycling"])
        data["heat"] = LiquidHeatState(**data["heat"])
        data["trips"] = {n: LiquidTripMemory(**v) for n, v in data["trips"].items()}
        data["pids"] = {n: PIDState(**v) for n, v in data["pids"].items()}
        return cls(**data)


class ThermalLiquidStation:
    def __init__(
        self,
        recycling,
        heat,
        temperature_trips=(),
        thermostats=(),
        control_interval_s=0.05,
        max_heating_w=None,
        max_cooling_w=None,
    ):
        if not isinstance(recycling, RecyclingLiquidStation) or not isinstance(
            heat, LiquidHeatNetwork
        ):
            raise TypeError("Thermal station requires recycling and caloric models")
        if recycling.physics.model_id != heat.physics.model_id:
            raise ValueError("Thermal/hydraulic model mismatch")
        self.recycling, self.heat = recycling, heat
        self.temperature_trips = tuple(
            r if isinstance(r, LiquidTemperatureTrip) else LiquidTemperatureTrip(**r)
            for r in temperature_trips
        )
        self.thermostats = tuple(thermostats)
        if any(not isinstance(r, LiquidThermostat) for r in self.thermostats):
            raise TypeError("Expected LiquidThermostat")
        for values in (self.temperature_trips, self.thermostats):
            if len({x.name for x in values}) != len(values):
                raise ValueError("Duplicate thermal rule/controller")
        if len({x.location for x in self.thermostats}) != len(self.thermostats):
            raise ValueError("Thermostats cannot share duty actuator")
        for rule in self.temperature_trips:
            if set(rule.stop_pumps) - set(recycling.physics.names):
                raise ValueError("Unknown temperature shutdown pump")
            names = heat.names if rule.metric == "fluid_temperature_k" else heat.wall_names
            if rule.location not in names:
                raise ValueError("Unknown temperature sensor")
            lo, hi = (
                (heat.min_temperature_k, heat.max_temperature_k)
                if rule.metric == "fluid_temperature_k"
                else (
                    heat.walls[rule.location].min_temperature_k,
                    heat.walls[rule.location].max_temperature_k,
                )
            )
            if not lo < rule.threshold_k < hi:
                raise ValueError("Temperature trip must lie inside hard domain")
        for unit in self.thermostats:
            if unit.location not in heat.names:
                raise ValueError("Unknown thermostat fluid inventory")
            heat._temperature(unit.setpoint_k)
        self.control_interval_s = _number(control_interval_s, "control_interval_s", positive=True)
        self.max_heating_w = max_heating_w
        self.max_cooling_w = max_cooling_w
        for value in (max_heating_w, max_cooling_w):
            if value is not None:
                _number(value, "thermal_capacity_w", nonnegative=True)

    def to_dict(self):
        config = self.heat.to_dict()
        config.pop("physics")
        return {
            "recycling": self.recycling.to_dict(),
            "heat": config,
            "temperature_trips": [asdict(r) for r in self.temperature_trips],
            "thermostats": [asdict(r) for r in self.thermostats],
            "control_interval_s": self.control_interval_s,
            "max_heating_w": self.max_heating_w,
            "max_cooling_w": self.max_cooling_w,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["recycling"] = RecyclingLiquidStation.from_dict(data["recycling"])
        data["heat"] = LiquidHeatNetwork(data["recycling"].physics, **data["heat"])
        data["thermostats"] = [
            LiquidThermostat(**{**x, "controller": PIDController(**x["controller"])})
            for x in data.get("thermostats", ())
        ]
        return cls(**data)

    @property
    def model_id(self):
        return _digest({"model": "thermal_liquid_recycle_station_v1", **self.to_dict()})

    def initialize(self, recycling_initial=None, temperatures_k=None, wall_temperatures_k=None):
        recycling = self.recycling.initialize(**dict(recycling_initial or {}))
        heat = self.heat.initialize(
            recycling.protected.controlled.physical, temperatures_k, wall_temperatures_k
        )
        return ThermalLiquidState(
            recycling,
            heat,
            {r.name: LiquidTripMemory() for r in self.temperature_trips},
            {
                r.name: r.controller.initialize(
                    dict(heat.temperatures_k)[r.location], setpoint=r.setpoint_k
                )
                for r in self.thermostats
            },
            {r.name: 0.0 for r in self.thermostats},
            recycling.protected.external_unavailable,
            recycling.protected.operator_tripped,
            heat.time_s,
            0,
            self.model_id,
        )

    def _actions(self, memories, external, operator):
        latched = [r for r in self.temperature_trips if memories[r.name].tripped_at_s is not None]
        return set(external).union(*(set(r.stop_pumps) for r in latched)), operator or any(
            r.trip_station for r in latched
        )

    def _check(self, state):
        if not isinstance(state, ThermalLiquidState) or state.model_id != self.model_id:
            raise ValueError("Thermal station state/model mismatch")
        self.recycling._check(state.recycling)
        self.heat._check(state.heat, state.recycling.protected.controlled.physical)
        if (
            set(dict(state.trips)) != {r.name for r in self.temperature_trips}
            or set(dict(state.pids)) != {r.name for r in self.thermostats}
            or set(dict(state.held_duties_w)) != {r.name for r in self.thermostats}
            or set(state.external_unavailable) - set(self.recycling.physics.names)
        ):
            raise ValueError("Thermal protection/control coordinate mismatch")
        for _, memory in state.trips:
            if not isinstance(memory, LiquidTripMemory) or any(
                v is not None and v > state.heat.time_s for v in asdict(memory).values()
            ):
                raise ValueError("Invalid temperature trip timer history")
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
                raise ValueError("Invalid temperature trip timer order")
        for unit in self.thermostats:
            if dict(state.pids)[unit.name].model_id != unit.controller.model_id:
                raise ValueError("Thermostat memory mismatch")
            if (
                not unit.controller.min_output
                <= dict(state.held_duties_w)[unit.name]
                <= unit.controller.max_output
            ):
                raise ValueError("Thermostat applied duty outside equipment limits")
        for cap, sign in ((self.max_heating_w, 1), (self.max_cooling_w, -1)):
            if (
                cap is not None
                and math.fsum(max(0.0, sign * v) for _, v in state.held_duties_w) > cap + 1e-9
            ):
                raise ValueError("Thermal utility capacity exceeded")
        blocked, tripped = self._actions(
            dict(state.trips), state.external_unavailable, state.operator_tripped
        )
        if (
            blocked != set(state.recycling.protected.external_unavailable)
            or tripped != state.recycling.protected.operator_tripped
        ):
            raise ValueError("Temperature latch action provenance mismatch")
        next_time = state.clock_origin_s + state.next_tick * self.control_interval_s
        tol = 16 * math.ulp(max(1.0, state.heat.time_s, next_time))
        if (
            not math.isfinite(next_time)
            or state.clock_origin_s > state.heat.time_s
            or next_time < state.heat.time_s - tol
            or next_time > state.heat.time_s + self.control_interval_s + tol
        ):
            raise ValueError("Thermal sampling clock mismatch")

    @staticmethod
    def _measure(rule, heat):
        return dict(
            heat.temperatures_k
            if rule.metric == "fluid_temperature_k"
            else heat.wall_temperatures_k
        )[rule.location]

    def advance(
        self,
        state,
        duration_s,
        mode="auto",
        unavailable=None,
        reset_trip=False,
        reset_temperature_trips=(),
        boundary_temperatures_k=None,
        ambient_temperatures_k=None,
        heat_duties_w=None,
        temperature_setpoints_k=None,
        temperature_control_modes=None,
        temperature_manual_duties_w=None,
        max_thermal_samples=100000,
        max_thermal_substeps=100000,
        max_substeps=100000,
        max_control_samples=100000,
        max_protection_samples=100000,
        max_recycle_samples=100000,
        **commands,
    ):
        self._check(state)
        duration = _number(duration_s, "duration_s", positive=True)
        if (
            mode not in ("auto", "manual", "torque", "off", "trip")
            or not isinstance(reset_trip, bool)
            or (mode == "trip" and reset_trip)
        ):
            raise ValueError("Invalid thermal station mode/reset")
        for value in (
            max_thermal_samples,
            max_thermal_substeps,
            max_substeps,
            max_control_samples,
            max_protection_samples,
            max_recycle_samples,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Positive integer work budgets required")
        external = state.external_unavailable if unavailable is None else tuple(unavailable)
        resets = tuple(reset_temperature_trips)
        if (
            set(external) - set(self.recycling.physics.names)
            or len(external) != len(set(external))
            or set(resets) - {r.name for r in self.temperature_trips}
            or len(resets) != len(set(resets))
        ):
            raise ValueError("Unknown/duplicate thermal reset or unavailable pump")
        targets = {r.name: r.setpoint_k for r in self.thermostats}
        targets.update(temperature_setpoints_k or {})
        modes = {r.name: "auto" for r in self.thermostats}
        modes.update(temperature_control_modes or {})
        manual = dict(temperature_manual_duties_w or {})
        duties = dict(heat_duties_w or {})
        names = {r.name for r in self.thermostats}
        if (
            set(targets) != names
            or set(modes) != names
            or set(manual) - names
            or any(v not in ("auto", "manual", "off") for v in modes.values())
        ):
            raise ValueError("Unknown thermostat command")
        if set(duties) & {r.location for r in self.thermostats} or set(duties) - set(
            self.heat.names
        ):
            raise ValueError("Unknown/conflicting external heat duty")
        for value in targets.values():
            self.heat._temperature(value)
        for value in duties.values():
            _number(value, "heat_duty_w")
        for unit in self.thermostats:
            if modes[unit.name] == "manual" and unit.name not in manual:
                raise ValueError("Manual thermal control requires duty")
            if (
                unit.name in manual
                and not unit.controller.min_output
                <= _number(manual[unit.name], "manual_duty_w")
                <= unit.controller.max_output
            ):
                raise ValueError("Manual duty outside equipment map")
        memories, pids, held = dict(state.trips), dict(state.pids), dict(state.held_duties_w)
        current, heat, tick = state.recycling, state.heat, state.next_tick
        operator = (state.operator_tripped and not reset_trip) or mode == "trip"
        events = []
        for rule in self.temperature_trips:
            if rule.name in resets:
                value = self._measure(rule, heat)
                safe = rule.reset_safe(value)
                events.append(
                    {
                        "type": "reset" if safe else "reset_denied",
                        "name": rule.name,
                        "time_s": heat.time_s,
                        "temperature_k": value,
                    }
                )
                if safe:
                    memories[rule.name] = LiquidTripMemory()
        end = heat.time_s + duration
        if not math.isfinite(end) or end <= heat.time_s:
            raise ValueError("Thermal station report cannot advance time")
        records = {
            k: []
            for k in (
                "steps",
                "control_samples",
                "protection_samples",
                "dispatches",
                "events",
                "recycle_samples",
            )
        }
        samples, thermal_records, thermal_audits = [], [], []
        retries = {}
        thermal_failed = 0
        first = True
        recycled = {x.valve: 0.0 for x in self.recycling.loops}
        while heat.time_s < end:
            now = heat.time_s
            next_time = state.clock_origin_s + tick * self.control_interval_s
            tol = 8 * math.ulp(max(1.0, now, end, next_time))
            terminal = 0 < end - now <= tol
            if now >= next_time - tol and not terminal:
                if len(samples) >= max_thermal_samples:
                    raise HydraulicConvergenceError("Thermal sample budget exhausted")
                sample = {"time_s": now, "tick": tick, "rules": {}, "controllers": {}}
                for rule in self.temperature_trips:
                    value = self._measure(rule, heat)
                    memory = memories[rule.name]
                    if memory.tripped_at_s is None:
                        since = now if memory.armed_since_s is None else memory.armed_since_s
                        bad = (
                            (now if memory.violating_since_s is None else memory.violating_since_s)
                            if rule.violating(value)
                            else None
                        )
                        trigger = bad is not None and now - bad + tol >= rule.delay_s
                        memory = LiquidTripMemory(since, bad, now if trigger else None)
                        if trigger:
                            events.append(
                                {
                                    "type": "temperature_trip",
                                    "name": rule.name,
                                    "time_s": now,
                                    "temperature_k": value,
                                    "stop_pumps": list(rule.stop_pumps),
                                    "trip_station": rule.trip_station,
                                }
                            )
                        memories[rule.name] = memory
                    sample["rules"][rule.name] = {"temperature_k": value, **asdict(memory)}
                requested = {}
                for unit in self.thermostats:
                    value = dict(heat.temperatures_k)[unit.location]
                    update = unit.controller.step(
                        pids[unit.name],
                        targets[unit.name],
                        value,
                        self.control_interval_s,
                        mode=modes[unit.name],
                        manual_output=manual.get(unit.name, 0.0),
                        applied_output=held[unit.name],
                    )
                    pids[unit.name] = PIDState(**update["state"])
                    requested[unit.name] = update["output"]
                    sample["controllers"][unit.name] = {
                        "temperature_k": value,
                        "setpoint_k": targets[unit.name],
                        "requested_duty_w": update["output"],
                        "mode": modes[unit.name],
                    }
                held = requested.copy()
                for cap, sign in ((self.max_heating_w, 1), (self.max_cooling_w, -1)):
                    total = math.fsum(max(0.0, sign * v) for v in held.values())
                    if cap is not None and total > cap:
                        for n, v in held.items():
                            if sign * v > 0:
                                held[n] = v * cap / total
                for n, v in held.items():
                    sample["controllers"][n]["applied_duty_w"] = v
                samples.append(sample)
                tick += 1
                next_time = state.clock_origin_s + tick * self.control_interval_s
                if next_time <= now:
                    raise ValueError("Thermal clock cannot advance")
            segment = dict(commands)
            if not first:
                for name in ("reset_trips", "redispatch"):
                    segment.pop(name, None)
            blocked, station_trip = self._actions(memories, external, operator)
            budgets = {
                "max_substeps": max_substeps - len(records["steps"]) - sum(retries.values()),
                "max_control_samples": max_control_samples - len(records["control_samples"]),
                "max_protection_samples": max_protection_samples
                - len(records["protection_samples"]),
                "max_recycle_samples": max_recycle_samples
                - len(records["recycle_samples"]) // len(self.recycling.loops),
            }
            if budgets["max_substeps"] < 1:
                raise HydraulicConvergenceError("Thermal station physical budget exhausted")
            result = self.recycling.advance(
                current,
                (end if terminal else min(end, next_time)) - now,
                mode="trip" if station_trip else mode,
                unavailable=tuple(n for n in self.recycling.physics.names if n in blocked),
                reset_trip=not station_trip and current.protected.operator_tripped,
                **{k: max(1, v) for k, v in budgets.items()},
                **segment,
            )
            for key, budget in (
                ("control_samples", "max_control_samples"),
                ("protection_samples", "max_protection_samples"),
            ):
                if len(result[key]) > budgets[budget]:
                    raise HydraulicConvergenceError(
                        "Thermal station nested sample budget exhausted"
                    )
            if (
                len(result["recycle_samples"]) // len(self.recycling.loops)
                > budgets["max_recycle_samples"]
            ):
                raise HydraulicConvergenceError("Thermal station recycle budget exhausted")
            applied = {**duties, **{r.location: held[r.name] for r in self.thermostats}}
            for record in result["steps"]:
                budget = max_thermal_substeps - len(thermal_records) - thermal_failed
                if budget < 1:
                    raise HydraulicConvergenceError("Thermal station heat budget exhausted")
                calc = self.heat.advance_record(
                    heat, record, boundary_temperatures_k, ambient_temperatures_k, applied, budget
                )
                heat = LiquidHeatState(**calc["state"])
                thermal_records.extend(calc["steps"])
                thermal_failed += calc["failed_steps"]
                thermal_audits.append(
                    {
                        "time_s": record["time_s"],
                        "dt_s": record["dt_s"],
                        "combined_energy_residual_j": calc["combined_energy_residual_j"],
                    }
                )
            for key, values in records.items():
                values.extend(result[key])
            for key, count in result["retry_reasons"].items():
                retries[key] = retries.get(key, 0) + count
            for n, value in result["recycled_volume_m3"].items():
                recycled[n] += value
            current = RecyclingLiquidState.from_dict(result["state"])
            # Accepted hydraulic records are independently anchored inside
            # reporting windows. Synchronize only endpoint floating roundoff,
            # never remove an unintegrated physical interval.
            physical_time = current.protected.controlled.physical.time_s
            if abs(heat.time_s - physical_time) > 32 * math.ulp(
                max(1.0, heat.time_s, physical_time)
            ):
                raise ValueError("Thermal window did not reach hydraulic endpoint")
            heat = replace(heat, time_s=physical_time)
            first = False
        final = ThermalLiquidState(
            current,
            heat,
            memories,
            pids,
            held,
            tuple(external),
            operator,
            state.clock_origin_s,
            tick,
            self.model_id,
        )
        self._check(final)
        return {
            "state": final.to_dict(),
            **records,
            "thermal_steps": thermal_records,
            "thermal_audits": thermal_audits,
            "temperature_samples": samples,
            "temperature_events": events,
            "latched_temperature_trips": [
                r.name for r in self.temperature_trips if memories[r.name].tripped_at_s is not None
            ],
            "latched_trips": result["latched_trips"],
            "recycled_volume_m3": recycled,
            "retry_reasons": retries,
            "failed_steps": sum(retries.values()),
            "failed_thermal_steps": thermal_failed,
            "thermal_properties": self.heat.inspect(heat),
            "properties": result["properties"],
            "pressure_error_pa": result["pressure_error_pa"],
            "duration_s": duration,
        }
