"""Sampled pressure/speed control over conservative liquid/shaft dynamics.

The integer controller clock survives report boundaries and restart. Steady
dispatch is a planning calculation only: measurements and actuator tracking
always come from the transient network, never from the steady candidate.
"""

import math
from dataclasses import asdict, dataclass

from .dynamic_station import DynamicPumpStation
from .liquid import HydraulicConvergenceError, _number
from .liquid_transient import LiquidTransientNetwork, LiquidTransientState
from .operation import _digest
from .pid import PIDController, PIDState
from .station import PumpStationControl


@dataclass(frozen=True)
class ControlledLiquidState:
    physical: LiquidTransientState
    pressure_pid: PIDState
    speed_pids: tuple
    held_torques_nm: tuple
    torque_integrals_nm_s: tuple
    feedback_duration_s: float
    selected: tuple
    clock_origin_s: float
    next_tick: int
    last_switch_time_s: float
    dispatch_signature: str
    dispatch_status: str
    pending_redispatch: bool
    model_id: str

    def __post_init__(self):
        for key in ("speed_pids", "held_torques_nm", "torque_integrals_nm_s"):
            object.__setattr__(self, key, tuple(sorted(dict(getattr(self, key)).items())))
        object.__setattr__(self, "selected", tuple(self.selected))
        if len(self.selected) != len(set(self.selected)):
            raise ValueError("Duplicate selected pump")
        for key in ("held_torques_nm", "torque_integrals_nm_s"):
            for _, value in getattr(self, key):
                _number(value, key)
        for key in ("feedback_duration_s", "clock_origin_s", "last_switch_time_s"):
            _number(getattr(self, key), key, nonnegative=True)
        if (
            isinstance(self.next_tick, bool)
            or not isinstance(self.next_tick, int)
            or self.next_tick < 0
        ):
            raise ValueError("Controller tick must be a nonnegative integer")
        if not isinstance(self.pending_redispatch, bool):
            raise TypeError("pending_redispatch must be boolean")

    def to_dict(self):
        return {
            "physical": self.physical.to_dict(),
            "pressure_pid": self.pressure_pid.to_dict(),
            "speed_pids": {n: p.to_dict() for n, p in self.speed_pids},
            "held_torques_nm": dict(self.held_torques_nm),
            "torque_integrals_nm_s": dict(self.torque_integrals_nm_s),
            "selected": list(self.selected),
            **{
                k: getattr(self, k)
                for k in (
                    "feedback_duration_s",
                    "clock_origin_s",
                    "next_tick",
                    "last_switch_time_s",
                    "dispatch_signature",
                    "dispatch_status",
                    "pending_redispatch",
                    "model_id",
                )
            },
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["physical"] = LiquidTransientState(**data["physical"])
        data["pressure_pid"] = PIDState(**data["pressure_pid"])
        data["speed_pids"] = {n: PIDState(**v) for n, v in data["speed_pids"].items()}
        return cls(**data)


class ControlledLiquidStation:
    def __init__(
        self,
        physics,
        pressure_node,
        pressure_controller,
        speed_controllers,
        dispatch_policy=None,
        control_interval_s=0.05,
        minimum_stage_dwell_s=0.5,
        dispatch_power_deadband_w=10.0,
        planning_valve_openings=None,
    ):
        if not isinstance(physics, LiquidTransientNetwork) or not physics.names:
            raise ValueError("Controlled station requires a liquid transient network with pumps")
        if pressure_node not in physics.free:
            raise ValueError("Pressure feedback must be a dynamic storage node")
        if (
            not isinstance(pressure_controller, PIDController)
            or pressure_controller.min_output != 0
        ):
            raise ValueError("Pressure PID must output nonnegative speed ratio")
        if set(speed_controllers) != set(physics.names):
            raise ValueError("Every shaft requires a local PID")
        for n, pid in speed_controllers.items():
            d = physics.drives[n]
            if (
                not isinstance(pid, PIDController)
                or pid.min_output < -d.max_braking_torque_nm
                or pid.max_output > d.max_motor_torque_nm
            ):
                raise ValueError("Local PID limits must fit its drive")
        if dispatch_policy is not None and (
            not isinstance(dispatch_policy, PumpStationControl)
            or set(dispatch_policy.pumps) != set(physics.names)
            or dispatch_policy.node != pressure_node
            or dispatch_policy.max_speed_ratio > pressure_controller.max_output
        ):
            raise ValueError("Dispatch policy must match pumps, feedback and PID range")
        self.physics, self.pressure_node = physics, pressure_node
        self.pressure_controller = pressure_controller
        self.speed_controllers = dict(speed_controllers)
        self.dispatch_policy = dispatch_policy
        self.control_interval_s = _number(control_interval_s, "control_interval_s", positive=True)
        self.minimum_stage_dwell_s = _number(
            minimum_stage_dwell_s, "minimum_stage_dwell_s", nonnegative=True
        )
        self.dispatch_power_deadband_w = _number(
            dispatch_power_deadband_w, "dispatch_power_deadband_w", nonnegative=True
        )
        self.planning_valve_openings = dict(planning_valve_openings or {})
        physics._controls(valve_openings=self.planning_valve_openings)
        self.names = physics.names
        self._planner = DynamicPumpStation(
            physics.network,
            physics.drives,
            physics.nominal_speeds_rad_s,
            pressure_node=pressure_node,
            dispatch_policy=dispatch_policy,
            no_flow_power_w_at_nominal=physics.no_flow_power_w_at_nominal,
            max_station_electrical_draw_w=physics.max_station_electrical_draw_w,
            max_station_electrical_export_w=physics.max_station_electrical_export_w,
            dispatch_power_deadband_w=dispatch_power_deadband_w,
        )

        if physics.backend == "torch":
            from .torch_steady import TorchNetworkView

            self._planner.network = TorchNetworkView(self._planner.network, physics.device)
            self._planner._trial = TorchNetworkView(self._planner._trial, physics.device)

    def to_dict(self):
        return {
            "physics": self.physics.to_dict(),
            "planning_valve_openings": self.planning_valve_openings.copy(),
            "pressure_node": self.pressure_node,
            "pressure_controller": asdict(self.pressure_controller),
            "speed_controllers": {n: asdict(c) for n, c in self.speed_controllers.items()},
            "dispatch_policy": None
            if self.dispatch_policy is None
            else asdict(self.dispatch_policy),
            **{
                k: getattr(self, k)
                for k in (
                    "control_interval_s",
                    "minimum_stage_dwell_s",
                    "dispatch_power_deadband_w",
                )
            },
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["physics"] = LiquidTransientNetwork.from_dict(data["physics"])
        data["pressure_controller"] = PIDController(**data["pressure_controller"])
        data["speed_controllers"] = {
            n: PIDController(**v) for n, v in data["speed_controllers"].items()
        }
        if data.get("dispatch_policy") is not None:
            data["dispatch_policy"] = PumpStationControl(**data["dispatch_policy"])
        return cls(**data)

    @property
    def model_id(self):
        return _digest({"model": "sampled_transient_liquid_station_v1", **self.to_dict()})

    def initialize(
        self,
        physical_initial=None,
        physical=None,
        running_pumps=(),
        applied_torques_nm=None,
        pressure_setpoint_pa=None,
    ):
        if physical is not None and physical_initial is not None:
            raise ValueError("Specify physical state or initial data, not both")
        physical = (
            self.physics.initialize(**(physical_initial or {})) if physical is None else physical
        )
        self.physics._check(physical)
        selected = tuple(running_pumps)
        applied = dict(applied_torques_nm or {})
        for n, value in applied.items():
            _number(value, "initial_applied_torque_nm")
            if n not in selected and value != 0:
                raise ValueError("Initial torque requires a selected running unit")
        if (
            set(selected) - set(self.names)
            or set(applied) - set(self.names)
            or set(selected) & set(physical.unavailable)
        ):
            raise ValueError("Invalid initial running units/torques")
        speeds = dict(physical.speeds_rad_s)
        measured = dict(physical.pressures_pa)[self.pressure_node]
        ratio = (
            math.fsum(speeds[n] / self.physics.nominal_speeds_rad_s[n] for n in selected)
            / len(selected)
            if selected
            else 0.0
        )
        target = (
            measured
            if pressure_setpoint_pa is None
            else _number(pressure_setpoint_pa, "pressure_setpoint_pa", positive=True)
        )
        pids = {
            n: c.initialize(
                speeds[n],
                applied.get(n, 0.0) if n in selected else 0.0,
                speeds[n],
                mode="auto" if n in selected else "off",
            )
            for n, c in self.speed_controllers.items()
        }
        outer = self.pressure_controller.initialize(
            measured, ratio, target, mode="auto" if selected else "off"
        )
        state = ControlledLiquidState(
            physical,
            outer,
            pids,
            {n: applied.get(n, 0.0) if n in selected else 0.0 for n in self.names},
            {n: 0.0 for n in self.names},
            0.0,
            selected,
            physical.time_s,
            0,
            physical.time_s,
            "",
            "not_dispatched",
            False,
            self.model_id,
        )
        self._check(state)
        return state

    def _check(self, state):
        if not isinstance(state, ControlledLiquidState) or state.model_id != self.model_id:
            raise ValueError("Controlled liquid state/model mismatch")
        self.physics._check(state.physical)
        if (
            set(dict(state.speed_pids)) != set(self.names)
            or set(dict(state.held_torques_nm)) != set(self.names)
            or set(dict(state.torque_integrals_nm_s)) != set(self.names)
            or set(state.selected) - set(self.names)
        ):
            raise ValueError("Controlled station unit memory mismatch")
        if state.pressure_pid.model_id != self.pressure_controller.model_id:
            raise ValueError("Pressure PID identity mismatch")
        for n, pid in state.speed_pids:
            c = self.speed_controllers[n]
            if (
                pid.model_id != c.model_id
                or not c.min_output <= dict(state.held_torques_nm)[n] <= c.max_output
            ):
                raise ValueError("Speed PID identity/held torque mismatch")
        now = state.physical.time_s
        next_time = state.clock_origin_s + state.next_tick * self.control_interval_s
        tol = 16 * max(math.ulp(now), math.ulp(next_time), math.ulp(self.control_interval_s))
        expected = 0.0 if state.next_tick == 0 else now - (next_time - self.control_interval_s)
        if (
            not math.isfinite(next_time)
            or next_time < now - tol
            or state.clock_origin_s > now
            or state.last_switch_time_s > now
            or not -tol <= expected <= self.control_interval_s + tol
            or abs(state.feedback_duration_s - expected) > max(tol, 1e-10)
        ):
            raise ValueError("Controlled station clock/feedback mismatch")

    def _plan(self, target, unavailable, previous, options):
        # A sealed finite inventory is a valid dynamic problem, but a steady
        # open-throughput pressure candidate need not exist or be unique.
        # Detect this structurally, never by swallowing a numerical failure.
        boundary, valves = self.physics._controls(
            options.get("boundary_pressures_pa"), options.get("valve_openings")
        )
        adjacency = {n.name: set() for n in self.physics.network.nodes}
        for branch in list(self.physics.network.pipes) + valves:
            if branch.open and getattr(branch, "opening", 1) > 0:
                adjacency[branch.start].add(branch.end)
                adjacency[branch.end].add(branch.start)
        visited, queue = set(), [self.pressure_node]
        while queue:
            n = queue.pop()
            if n not in visited:
                visited.add(n)
                queue.extend(adjacency[n] - visited)
        if self.dispatch_policy is not None and not visited.intersection(boundary):
            available = tuple(n for n in self.names if n not in unavailable)
            selected = tuple(n for n in previous if n in available)[
                : self.dispatch_policy.max_running
            ]
            for n in available:
                if len(selected) >= self.dispatch_policy.min_running:
                    break
                if n not in selected:
                    selected += (n,)
            return (
                selected,
                0.0,
                {
                    "status": "dynamic_feedback_only" if available else "no_available_units",
                    "reason": "No open passive path to a pressure reservoir",
                    "candidates": [],
                },
            )
        return self._planner._dispatch(target, unavailable, previous, options)

    def advance(
        self,
        state,
        duration_s,
        mode="auto",
        pressure_setpoint_pa=None,
        speed_ratio=1.0,
        running_pumps=None,
        motor_torques_nm=None,
        unavailable=None,
        boundary_pressures_pa=None,
        valve_openings=None,
        actuator_modes=None,
        reset_trip=False,
        redispatch=False,
        max_substeps=100000,
        max_control_samples=100000,
    ):
        self._check(state)
        duration = _number(duration_s, "duration_s", positive=True)
        end_time = state.physical.time_s + duration
        if not math.isfinite(end_time) or end_time <= state.physical.time_s:
            raise ValueError("Controlled report cannot advance time")
        if (
            mode not in ("auto", "manual", "torque", "off", "trip")
            or not isinstance(reset_trip, bool)
            or not isinstance(redispatch, bool)
            or (mode == "trip" and reset_trip)
        ):
            raise ValueError("Invalid controlled station mode/reset")
        for value in (max_substeps, max_control_samples):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Positive integer work budgets required")
        target = (
            self.dispatch_policy.target_pressure_pa
            if pressure_setpoint_pa is None and self.dispatch_policy is not None
            else pressure_setpoint_pa
        )
        if target is not None:
            target = _number(target, "pressure_setpoint_pa", positive=True)
        if mode == "auto" and target is None:
            raise ValueError("Automatic pressure control requires a setpoint")
        ratio = _number(speed_ratio, "speed_ratio", nonnegative=True)
        if ratio > self.pressure_controller.max_output:
            raise ValueError("Manual speed ratio exceeds controller range")
        unknown = set(self.names)
        unavailable = state.physical.unavailable if unavailable is None else tuple(unavailable)
        requested = state.selected if running_pumps is None else tuple(running_pumps)
        inputs = dict(motor_torques_nm or {})
        if mode == "torque" and running_pumps is None:
            requested = tuple(inputs)
        if (
            (set(unavailable) | set(requested) | set(inputs)) - unknown
            or len(unavailable) != len(set(unavailable))
            or len(requested) != len(set(requested))
        ):
            raise ValueError("Unknown/duplicate controlled unit command")
        for v in inputs.values():
            _number(v, "motor_torque_nm")
        self.physics._controls(boundary_pressures_pa, valve_openings)
        options = {
            "boundary_pressures_pa": dict(boundary_pressures_pa or {}),
            "valve_openings": {**dict(valve_openings or {}), **self.planning_valve_openings},
        }
        physical = state.physical
        tripped = (physical.tripped and not reset_trip) or mode == "trip"
        effective = "off" if tripped else mode
        selected = state.selected
        held = dict(state.held_torques_nm)
        pids = dict(state.speed_pids)
        outer = state.pressure_pid
        integrals = dict(state.torque_integrals_nm_s)
        feedback = state.feedback_duration_s
        tick = state.next_tick
        last_switch = state.last_switch_time_s
        signature, status = state.dispatch_signature, state.dispatch_status
        pending = state.pending_redispatch or redispatch
        if effective in ("off", "trip"):
            held = {n: 0.0 for n in self.names}
            if selected:
                last_switch = physical.time_s
            selected = ()
            signature = ""
            status = "off"
        # Hard availability cuts act now; reserve selection remains sampled.
        held.update({n: 0.0 for n in unavailable})
        samples, dispatches, steps = [], [], []
        retries = {}
        first_segment = True
        while physical.time_s < end_time:
            now = physical.time_s
            next_time = state.clock_origin_s + tick * self.control_interval_s
            tol = 4 * max(math.ulp(now), math.ulp(next_time))
            # A few ULPs of report-time accumulation must not consume the next
            # command sample with the old period's inputs. Integrate the tiny
            # terminal tail with held commands, leaving that tick pending.
            terminal_roundoff = abs(now - next_time) <= tol and end_time - now <= 4 * max(
                math.ulp(now), math.ulp(next_time), math.ulp(end_time)
            )
            if abs(now - next_time) <= tol and not terminal_roundoff:
                if len(samples) >= max_control_samples:
                    raise HydraulicConvergenceError("Controlled station sample budget exhausted")
                measured = dict(physical.pressures_pa)[self.pressure_node]
                speeds = dict(physical.speeds_rad_s)
                before = selected
                seed = 0.0
                if effective == "auto":
                    wanted = _digest(
                        {"target": target, "unavailable": unavailable, "options": options}
                    )
                    count_violation = (
                        self.dispatch_policy is not None
                        and not self.dispatch_policy.min_running
                        <= len(selected)
                        <= self.dispatch_policy.max_running
                    )
                    forced = (
                        bool(set(selected) & set(unavailable)) or not selected or count_violation
                    )
                    due = (signature != wanted or pending) and (
                        forced or now - last_switch >= self.minimum_stage_dwell_s
                    )
                    if due:
                        selected, seed, plan = self._plan(target, unavailable, selected, options)
                        signature = wanted
                        status = plan["status"]
                        pending = False
                        dispatches.append(
                            {
                                "time_s": now,
                                "selected": list(selected),
                                "feedforward_speed_ratio": seed,
                                **plan,
                            }
                        )
                else:
                    selected = (
                        ()
                        if effective in ("off", "trip")
                        else tuple(n for n in requested if n not in unavailable)
                    )
                    status = effective
                    signature = ""
                if set(selected) != set(before):
                    last_switch = now
                actual_ratio = (
                    math.fsum(speeds[n] / self.physics.nominal_speeds_rad_s[n] for n in selected)
                    / len(selected)
                    if selected
                    else 0.0
                )
                seeded = effective == "auto" and selected and not before and seed > 0
                if seeded:
                    outer = self.pressure_controller.initialize(
                        measured,
                        min(seed, self.pressure_controller.max_output),
                        target,
                        mode="auto",
                    )
                outer_mode = (
                    "auto"
                    if effective == "auto" and selected
                    else ("manual" if selected and effective in ("manual", "torque") else "off")
                )
                outer_response = self.pressure_controller.step(
                    outer,
                    target if target is not None else measured,
                    measured,
                    self.control_interval_s,
                    outer_mode,
                    ratio,
                    min(self.pressure_controller.max_output, max(0.0, actual_ratio)),
                )
                outer = PIDState(**outer_response["state"])
                desired_ratio = outer_response["output"] if effective == "auto" else ratio
                applied = {
                    n: (integrals[n] / feedback if feedback > 0 else held[n]) for n in self.names
                }
                responses = {}
                for n, c in self.speed_controllers.items():
                    enabled = (
                        n in selected and n not in unavailable and effective not in ("off", "trip")
                    )
                    local_mode = (
                        ("auto" if effective in ("auto", "manual") else "manual")
                        if enabled
                        else "off"
                    )
                    actual = min(c.max_output, max(c.min_output, applied[n]))
                    response = c.step(
                        pids[n],
                        desired_ratio * self.physics.nominal_speeds_rad_s[n],
                        speeds[n],
                        self.control_interval_s,
                        local_mode,
                        inputs.get(n, 0.0),
                        actual,
                    )
                    pids[n] = PIDState(**response["state"])
                    held[n] = response["output"]
                    responses[n] = response
                samples.append(
                    {
                        "time_s": now,
                        "tick": tick,
                        "mode": effective,
                        "measured_pressure_pa": measured,
                        "measured_speeds_rad_s": speeds,
                        "selected": list(selected),
                        "unavailable": list(unavailable),
                        "actual_speed_ratio": actual_ratio,
                        "desired_speed_ratio": desired_ratio,
                        "pressure_pid_seeded": bool(seeded),
                        "pressure_pid": outer_response,
                        "speed_pids": responses,
                        "applied_torques_nm": applied,
                        "feedback_duration_s": feedback,
                        "motor_commands_nm": dict(held),
                        "dispatch_status": status,
                    }
                )
                integrals = {n: 0.0 for n in self.names}
                feedback = 0.0
                tick += 1
                next_time = state.clock_origin_s + tick * self.control_interval_s
                if next_time <= now:
                    raise ValueError("Controller tick cannot advance time")
            until = end_time if terminal_roundoff else min(end_time, next_time)
            if until <= physical.time_s:
                raise ValueError("Controller clock skipped or regressed")
            budget = max_substeps - len(steps) - sum(retries.values())
            if budget < 1:
                raise HydraulicConvergenceError("Controlled station physical budget exhausted")
            result = self.physics.advance(
                physical,
                until - physical.time_s,
                motor_torques_nm=held,
                mode="trip"
                if mode == "trip" and first_segment
                else ("off" if effective == "off" else "torque"),
                boundary_pressures_pa=boundary_pressures_pa,
                valve_openings=valve_openings,
                actuator_modes=actuator_modes,
                unavailable=unavailable,
                reset_trip=reset_trip if first_segment else False,
                max_substeps=budget,
            )
            for n in self.names:
                integrals[n] = math.fsum(
                    [integrals[n]] + [s["motor_torques_nm"][n] * s["dt_s"] for s in result["steps"]]
                )
            feedback = math.fsum((feedback, result["duration_s"]))
            for reason, count in result["retry_reasons"].items():
                retries[reason] = retries.get(reason, 0) + count
            steps.extend(result["steps"])
            physical = LiquidTransientState(**result["state"])
            first_segment = False
        final = ControlledLiquidState(
            physical,
            outer,
            pids,
            held,
            integrals,
            feedback,
            selected,
            state.clock_origin_s,
            tick,
            last_switch,
            signature,
            status,
            pending,
            self.model_id,
        )
        self._check(final)
        return {
            "state": final.to_dict(),
            "properties": self.physics.inspect(physical),
            "steps": steps,
            "control_samples": samples,
            "dispatches": dispatches,
            "retry_reasons": retries,
            "failed_steps": sum(retries.values()),
            "duration_s": duration,
            "requested_mode": mode,
            "effective_mode": effective,
            "pressure_error_pa": None
            if effective != "auto"
            else target - dict(physical.pressures_pa)[self.pressure_node],
        }
