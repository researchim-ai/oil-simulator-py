"""Persistent sampled thermostats and latched temperature trips on Gauss4.

This controller holds commands between integer clock ticks. It does not
claim continuous event detection, pressure/speed dispatch or SIL protection.
Trip actions remove torque, never conserved stocks or shaft velocity.
"""

import math
from dataclasses import asdict, dataclass

from .liquid import HydraulicConvergenceError, _number
from .liquid_heat import LiquidHeatNetwork, LiquidHeatState
from .liquid_protection import LiquidTripMemory
from .liquid_thermal_collocation import ThermalLiquidCollocation
from .liquid_transient import LiquidTransientState
from .operation import _digest
from .pid import PIDController, PIDState
from .thermal_control import allocate_thermal_duties, sample_temperature_trip
from .thermal_liquid import LiquidTemperatureTrip, LiquidThermostat, ThermalLiquidStation


@dataclass(frozen=True)
class CollocationThermalState:
    physical: LiquidTransientState
    heat: LiquidHeatState
    pids: tuple
    trips: tuple
    held_duties_w: tuple
    external_unavailable: tuple
    operator_tripped: bool
    clock_origin_s: float
    next_tick: int
    next_step_s: float
    model_id: str

    def __post_init__(self):
        for key in ("pids", "trips", "held_duties_w"):
            raw = getattr(self, key)
            values = dict(raw)
            if len(values) != len(raw):
                raise ValueError("Duplicate controller memory key")
            object.__setattr__(self, key, tuple(sorted(values.items())))
        object.__setattr__(self, "external_unavailable", tuple(self.external_unavailable))
        if len(set(self.external_unavailable)) != len(self.external_unavailable):
            raise ValueError("Duplicate unavailable drive")
        if (
            type(self.operator_tripped) is not bool
            or type(self.next_tick) is not int
            or self.next_tick < 0
        ):
            raise ValueError("Invalid trip latch or controller tick")
        _number(self.clock_origin_s, "clock origin", nonnegative=True)
        _number(self.next_step_s, "next step", positive=True)
        for _, v in self.held_duties_w:
            _number(v, "held thermal duty")

    def to_dict(self):
        return {
            "physical": self.physical.to_dict(),
            "heat": self.heat.to_dict(),
            "pids": {n: v.to_dict() for n, v in self.pids},
            "trips": {n: asdict(v) for n, v in self.trips},
            "held_duties_w": dict(self.held_duties_w),
            "external_unavailable": list(self.external_unavailable),
            "operator_tripped": self.operator_tripped,
            "clock_origin_s": self.clock_origin_s,
            "next_tick": self.next_tick,
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["physical"] = LiquidTransientState(**data["physical"])
        data["heat"] = LiquidHeatState(**data["heat"])
        data["pids"] = {n: PIDState(**v) for n, v in data["pids"].items()}
        data["trips"] = {n: LiquidTripMemory(**v) for n, v in data["trips"].items()}
        return cls(**data)


class CollocationThermalStation:
    def __init__(
        self,
        heat,
        thermostats=(),
        temperature_trips=(),
        control_interval_s=0.05,
        max_heating_w=None,
        max_cooling_w=None,
        solver_options=None,
    ):
        if not isinstance(heat, LiquidHeatNetwork):
            raise TypeError("Expected LiquidHeatNetwork")
        self.heat, self.physics = heat, heat.physics
        self.thermostats = tuple(thermostats)
        self.temperature_trips = tuple(
            r if isinstance(r, LiquidTemperatureTrip) else LiquidTemperatureTrip(**r)
            for r in temperature_trips
        )
        if any(not isinstance(u, LiquidThermostat) for u in self.thermostats):
            raise TypeError("Expected LiquidThermostat controllers")
        for values in (self.thermostats, self.temperature_trips):
            if len({u.name for u in values}) != len(values):
                raise ValueError("Duplicate thermal controller or trip")
        if len({u.location for u in self.thermostats}) != len(self.thermostats):
            raise ValueError("Thermostats cannot share a duty actuator")
        for u in self.thermostats:
            if u.location not in heat.names:
                raise ValueError("Unknown thermostat inventory")
            heat._temperature(u.setpoint_k)
        for r in self.temperature_trips:
            names = heat.names if r.metric == "fluid_temperature_k" else heat.wall_names
            if r.location not in names or set(r.stop_pumps) - set(self.physics.names):
                raise ValueError("Unknown protection sensor or pump")
            lo, hi = (
                (heat.min_temperature_k, heat.max_temperature_k)
                if r.metric == "fluid_temperature_k"
                else (
                    heat.walls[r.location].min_temperature_k,
                    heat.walls[r.location].max_temperature_k,
                )
            )
            if not lo < r.threshold_k < hi:
                raise ValueError("Temperature trip must lie inside hard domain")
        self.control_interval_s = _number(control_interval_s, "control interval", positive=True)
        self.max_heating_w, self.max_cooling_w = max_heating_w, max_cooling_w
        for value in (max_heating_w, max_cooling_w):
            if value is not None:
                _number(value, "thermal capacity", nonnegative=True)
        self.solver_options = dict(solver_options or {})
        self.solver = ThermalLiquidCollocation(heat, **self.solver_options)
        self.minimum_dt_s = max(heat.minimum_dt_s, self.physics.minimum_dt_s)
        self.maximum_dt_s = min(heat.maximum_dt_s, self.physics.maximum_dt_s)
        if self.minimum_dt_s > self.maximum_dt_s:
            raise ValueError("Coupled timestep ranges do not overlap")
        self._built_model_id = self.model_id

    def to_dict(self):
        return {
            "heat": self.heat.to_dict(),
            "thermostats": [asdict(u) for u in self.thermostats],
            "temperature_trips": [asdict(r) for r in self.temperature_trips],
            "control_interval_s": self.control_interval_s,
            "max_heating_w": self.max_heating_w,
            "max_cooling_w": self.max_cooling_w,
            "solver_options": self.solver_options.copy(),
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["heat"] = LiquidHeatNetwork.from_dict(data["heat"])
        data["thermostats"] = [
            LiquidThermostat(**{**u, "controller": PIDController(**u["controller"])})
            for u in data["thermostats"]
        ]
        return cls(**data)

    @property
    def model_id(self):
        return _digest({"model": "sampled_collocation_thermal_station_v1", **self.to_dict()})

    def initialize(self, physical, heat):
        self.heat._check(heat, physical)
        state = CollocationThermalState(
            physical,
            heat,
            {
                u.name: u.controller.initialize(
                    dict(heat.temperatures_k)[u.location], setpoint=u.setpoint_k
                )
                for u in self.thermostats
            },
            {r.name: LiquidTripMemory() for r in self.temperature_trips},
            {u.name: 0.0 for u in self.thermostats},
            (),
            False,
            physical.time_s,
            0,
            max(self.minimum_dt_s, self.maximum_dt_s / 8),
            self.model_id,
        )
        self._check(state)
        return state

    def _check(self, state):
        if self.model_id != self._built_model_id:
            raise ValueError("Controlled collocation model changed after construction")
        if not isinstance(state, CollocationThermalState) or state.model_id != self.model_id:
            raise ValueError("Controlled collocation state/model mismatch")
        self.heat._check(state.heat, state.physical)
        if state.physical.tripped or state.physical.unavailable:
            raise ValueError("Physical trip flags must be managed by their owning controller")
        if (
            set(dict(state.pids)) != {u.name for u in self.thermostats}
            or set(dict(state.trips)) != {r.name for r in self.temperature_trips}
            or set(dict(state.held_duties_w)) != {u.name for u in self.thermostats}
            or set(state.external_unavailable) - set(self.physics.names)
        ):
            raise ValueError("Controlled thermal registry mismatch")
        for u in self.thermostats:
            pid = dict(state.pids)[u.name]
            if not isinstance(pid, PIDState) or pid.model_id != u.controller.model_id:
                raise ValueError("Thermostat memory mismatch")
            if not u.controller.min_output <= pid.previous_output <= u.controller.max_output:
                raise ValueError("PID output history outside controller limits")
            if (
                not u.controller.min_output
                <= dict(state.held_duties_w)[u.name]
                <= u.controller.max_output
            ):
                raise ValueError("Applied duty outside controller limits")
        for memory in dict(state.trips).values():
            if not isinstance(memory, LiquidTripMemory):
                raise TypeError("Invalid trip memory")
            if any(v is not None and v > state.heat.time_s for v in asdict(memory).values()):
                raise ValueError("Protection timer lies in the future")
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
        for cap, sign in ((self.max_heating_w, 1), (self.max_cooling_w, -1)):
            if (
                cap is not None
                and math.fsum(max(0.0, sign * v) for _, v in state.held_duties_w) > cap + 1e-9
            ):
                raise ValueError("Thermal utility capacity exceeded")
        due = self._tick_time(state.clock_origin_s, state.next_tick)
        now = state.physical.time_s
        tolerance = 16 * math.ulp(max(1.0, now, due))
        if (
            not math.isfinite(due)
            or state.clock_origin_s > now
            or due < now - tolerance
            or due > now + self.control_interval_s + tolerance
            or due + self.control_interval_s <= due
        ):
            raise ValueError("Persistent controller clock mismatch")
        if not self.minimum_dt_s <= state.next_step_s <= self.maximum_dt_s:
            raise ValueError("Invalid next physical step")

    def _tick_time(self, origin, tick):
        try:
            due = origin + tick * self.control_interval_s
        except OverflowError as exc:
            raise ValueError("Controller clock is not representable") from exc
        if not math.isfinite(due) or due + self.control_interval_s <= due:
            raise ValueError("Controller clock is not representable")
        return due

    def advance(
        self,
        state,
        duration_s,
        *,
        motor_torques_nm=None,
        boundary_pressures_pa=None,
        valve_openings=None,
        boundary_temperatures_k=None,
        ambient_temperatures_k=None,
        heat_duties_w=None,
        temperature_setpoints_k=None,
        temperature_control_modes=None,
        temperature_manual_duties_w=None,
        mode="auto",
        unavailable=None,
        reset_trip=False,
        reset_temperature_trips=(),
        max_samples=10000,
        max_attempts=10000,
    ):
        self._check(state)
        duration = _number(duration_s, "duration", positive=True)
        end = state.physical.time_s + duration
        if not math.isfinite(end) or end <= state.physical.time_s:
            raise ValueError("Controller window cannot advance representable time")
        if (
            mode not in ("auto", "off", "trip")
            or type(reset_trip) is not bool
            or (mode == "trip" and reset_trip)
        ):
            raise ValueError("Invalid station mode/reset")
        if any(type(v) is not int or v < 1 for v in (max_samples, max_attempts)):
            raise ValueError("Positive integer work budgets required")
        blocked = state.external_unavailable if unavailable is None else tuple(unavailable)
        if len(set(blocked)) != len(blocked) or set(blocked) - set(self.physics.names):
            raise ValueError("Unknown or duplicate unavailable drive")
        command = {n: 0.0 for n in self.physics.names}
        supplied = dict(motor_torques_nm or {})
        if set(supplied) - set(command):
            raise ValueError("Unknown motor command")
        command.update(supplied)
        for n, value in command.items():
            _number(value, "motor torque")
            if (
                not -self.physics.drives[n].max_braking_torque_nm
                <= value
                <= self.physics.drives[n].max_motor_torque_nm
            ):
                raise ValueError("Motor torque outside drive limits")
        targets = {u.name: u.setpoint_k for u in self.thermostats}
        targets.update(temperature_setpoints_k or {})
        modes = {u.name: "auto" for u in self.thermostats}
        modes.update(temperature_control_modes or {})
        manual = dict(temperature_manual_duties_w or {})
        names = {u.name for u in self.thermostats}
        if (
            set(targets) != names
            or set(modes) != names
            or set(manual) - names
            or any(v not in ("auto", "manual", "off") for v in modes.values())
        ):
            raise ValueError("Unknown thermostat control")
        for value in targets.values():
            self.heat._temperature(value)
        for u in self.thermostats:
            if modes[u.name] == "manual" and u.name not in manual:
                raise ValueError("Manual thermal control requires duty")
            if (
                u.name in manual
                and not u.controller.min_output
                <= _number(manual[u.name], "manual duty")
                <= u.controller.max_output
            ):
                raise ValueError("Manual duty outside controller limits")
        duties = dict(heat_duties_w or {})
        if set(duties) - set(self.heat.names) or set(duties) & {
            u.location for u in self.thermostats
        }:
            raise ValueError("Unknown or conflicting external heat duty")
        for value in duties.values():
            _number(value, "heat duty")
        resets = tuple(reset_temperature_trips)
        if len(set(resets)) != len(resets) or set(resets) - {
            r.name for r in self.temperature_trips
        }:
            raise ValueError("Unknown or duplicate temperature reset")
        physical, heat, tick, step = state.physical, state.heat, state.next_tick, state.next_step_s
        pids, memories, held = dict(state.pids), dict(state.trips), dict(state.held_duties_w)
        operator = (state.operator_tripped and not reset_trip) or mode == "trip"
        events = []
        if mode == "trip" and not state.operator_tripped:
            events.append({"type": "operator_trip", "time_s": physical.time_s})
        if state.operator_tripped and reset_trip:
            events.append({"type": "operator_reset", "time_s": physical.time_s})
        if set(blocked) != set(state.external_unavailable):
            events.append(
                {
                    "type": "availability_command",
                    "time_s": physical.time_s,
                    "unavailable": sorted(blocked),
                }
            )
        for rule in self.temperature_trips:
            if rule.name in resets:
                value = ThermalLiquidStation._measure(rule, heat)
                safe = rule.reset_safe(value)
                events.append(
                    {
                        "type": "reset" if safe else "reset_denied",
                        "name": rule.name,
                        "time_s": physical.time_s,
                        "temperature_k": value,
                    }
                )
                if safe:
                    memories[rule.name] = LiquidTripMemory()
        segments, samples = [], []
        attempts = 0
        while physical.time_s < end:
            now = physical.time_s
            due = self._tick_time(state.clock_origin_s, tick)
            tolerance = 8 * math.ulp(max(1.0, now, end, due))
            if now >= due - tolerance and due < end:
                if len(samples) >= max_samples:
                    raise HydraulicConvergenceError("Controller sample budget exhausted")
                sample = {"time_s": now, "tick": tick, "rules": {}, "controllers": {}}
                for rule in self.temperature_trips:
                    value = ThermalLiquidStation._measure(rule, heat)
                    memory, trigger = sample_temperature_trip(
                        rule, memories[rule.name], value, now, tolerance
                    )
                    memories[rule.name] = memory
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
                    sample["rules"][rule.name] = {"temperature_k": value, **asdict(memory)}
                requested = {}
                for u in self.thermostats:
                    value = dict(heat.temperatures_k)[u.location]
                    update = u.controller.step(
                        pids[u.name],
                        targets[u.name],
                        value,
                        self.control_interval_s,
                        mode=modes[u.name],
                        manual_output=manual.get(u.name, 0.0),
                        applied_output=held[u.name],
                    )
                    pids[u.name] = PIDState(**update["state"])
                    requested[u.name] = update["output"]
                    sample["controllers"][u.name] = {
                        "temperature_k": value,
                        "setpoint_k": targets[u.name],
                        "requested_duty_w": update["output"],
                        "mode": modes[u.name],
                    }
                held = allocate_thermal_duties(requested, self.max_heating_w, self.max_cooling_w)
                for n, value in held.items():
                    sample["controllers"][n]["applied_duty_w"] = value
                samples.append(sample)
                tick += 1
                due = self._tick_time(state.clock_origin_s, tick)
                if due <= now:
                    raise ValueError("Controller clock cannot advance")
            cutoff = min(end, due)
            if cutoff <= now:
                raise ValueError("Non-increasing control segment")
            remaining = max_attempts - attempts
            if remaining < 1:
                raise HydraulicConvergenceError("Station physical attempt budget exhausted")
            latched = [
                r for r in self.temperature_trips if memories[r.name].tripped_at_s is not None
            ]
            stopped = set(blocked).union(*(set(r.stop_pumps) for r in latched))
            all_off = operator or mode == "off" or any(r.trip_station for r in latched)
            applied = {
                n: (0.0 if all_off or n in stopped else value) for n, value in command.items()
            }
            heat_duties = {**duties, **{u.location: held[u.name] for u in self.thermostats}}
            result = self.solver.advance(
                physical,
                heat,
                cutoff - now,
                motor_torques_nm=applied,
                boundary_pressures_pa=boundary_pressures_pa,
                valve_openings=valve_openings,
                boundary_temperatures_k=boundary_temperatures_k,
                ambient_temperatures_k=ambient_temperatures_k,
                heat_duties_w=heat_duties,
                initial_step_s=step,
                max_attempts=remaining,
            )
            attempts += result["hydraulics"]["attempts"]
            physical = LiquidTransientState(**result["state"]["hydraulics"])
            heat = LiquidHeatState(**result["state"]["heat"])
            step = result["hydraulics"]["next_step_s"]
            segments.append(result)
        final = CollocationThermalState(
            physical,
            heat,
            pids,
            memories,
            held,
            blocked,
            operator,
            state.clock_origin_s,
            tick,
            step,
            self.model_id,
        )
        self._check(final)
        return {
            "state": final.to_dict(),
            "samples": samples,
            "events": events,
            "segments": segments,
            "attempts": attempts,
            "duration_s": duration,
            "command": {
                "mode": mode,
                "motor_torques_nm": command,
                "unavailable": list(blocked),
                "operator_reset": reset_trip,
            },
            "combined_energy_residual_j": math.fsum(
                s["combined_energy_residual_j"] for s in segments
            ),
        }
