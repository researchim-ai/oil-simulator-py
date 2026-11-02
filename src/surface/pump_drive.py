"""Sampled PID and midpoint shaft coupled to a quasi-steady liquid network.

One controlled centrifugal pump with non-return/isolation at rest. Inertia
belongs to the shaft; pipe inventory, water hammer and fluid temperature are
not modeled. All dissipated energy is recorded, not silently deleted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

from scipy.optimize import brentq

from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    LiquidNetwork,
    _number,
)
from .operation import _digest
from .pid import PIDController, PIDState
from .rotating_drive import RotatingDrive


@dataclass(frozen=True)
class PumpDriveState:
    angular_velocity_rad_s: float
    time_s: float
    controller_state: PIDState | None
    applied_torque_nm: float
    model_id: str
    tripped: bool = False

    def __post_init__(self):
        _number(self.angular_velocity_rad_s, "angular_velocity_rad_s", nonnegative=True)
        _number(self.time_s, "time_s", nonnegative=True)
        _number(self.applied_torque_nm, "applied_torque_nm")
        if not isinstance(self.tripped, bool):
            raise TypeError("Trip latch must be boolean")
        if self.controller_state is not None and not isinstance(self.controller_state, PIDState):
            raise TypeError("Expected PIDState")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.get("controller_state") is not None:
            value["controller_state"] = PIDState(**value["controller_state"])
        return cls(**value)


class DynamicPumpNetwork:
    def __init__(
        self,
        network,
        drive,
        nominal_speed_rad_s,
        controller=None,
        measurement="pressure",
        measurement_node=None,
        control_interval_s=0.1,
        maximum_dt_s=0.02,
        minimum_dt_s=1e-7,
        max_speed_change_rad_s=2.0,
        no_flow_power_w_at_nominal=0.0,
    ):
        if not isinstance(network, LiquidNetwork) or not isinstance(drive, RotatingDrive):
            raise TypeError("Expected LiquidNetwork and RotatingDrive")
        if len(network.pumps) != 1 or not network.pumps[0].check_valve or not network.pumps[0].open:
            raise ValueError("Dynamic pump network requires one open pump with a check valve")
        if any(n.source_mass_rate_kg_s != 0 for n in network.nodes):
            raise ValueError("Startup requires pressure-driven boundaries, not imposed node demand")
        if controller is not None and not isinstance(controller, PIDController):
            raise TypeError("Expected PIDController")
        if measurement not in ("pressure", "flow", "speed"):
            raise ValueError("Unknown feedback measurement")
        if measurement == "pressure" and measurement_node not in network._index:
            raise ValueError("Pressure measurement requires a known node")
        for name, value in (
            ("nominal_speed_rad_s", nominal_speed_rad_s),
            ("control_interval_s", control_interval_s),
            ("maximum_dt_s", maximum_dt_s),
            ("minimum_dt_s", minimum_dt_s),
            ("max_speed_change_rad_s", max_speed_change_rad_s),
        ):
            _number(value, name, positive=True)
        if minimum_dt_s > maximum_dt_s:
            raise ValueError("Minimum step must not exceed maximum step")
        if controller is not None and (
            controller.min_output < -drive.max_braking_torque_nm
            or controller.max_output > drive.max_motor_torque_nm
        ):
            raise ValueError("PID output bounds must fit drive torque bounds")
        _number(no_flow_power_w_at_nominal, "no_flow_power_w_at_nominal", nonnegative=True)
        self.no_flow_power_w_at_nominal = float(no_flow_power_w_at_nominal)
        self.network, self.drive, self.controller = network, drive, controller
        self.pump = network.pumps[0]
        self.nominal_speed_rad_s = float(nominal_speed_rad_s)
        self.measurement, self.measurement_node = measurement, measurement_node
        self.control_interval_s, self.maximum_dt_s = float(control_interval_s), float(maximum_dt_s)
        self.minimum_dt_s, self.max_speed_change_rad_s = (
            float(minimum_dt_s),
            float(max_speed_change_rad_s),
        )
        # Trial states use the same physical curve; only safety constraints are
        # deferred to accepted midpoint/endpoint validation, never extrapolated.
        trial_pump = replace(
            self.pump, max_shaft_power_w=None, npsh_required_m=None, npsh_curve=None
        )
        self._trial_network = LiquidNetwork(
            network.fluid, network.nodes, network.pipes, [trial_pump], network.valves
        )

    def to_dict(self):
        return {
            "network": self.network.to_dict(),
            "drive": asdict(self.drive),
            "nominal_speed_rad_s": self.nominal_speed_rad_s,
            "no_flow_power_w_at_nominal": self.no_flow_power_w_at_nominal,
            "controller": None if self.controller is None else asdict(self.controller),
            **{
                key: getattr(self, key)
                for key in (
                    "measurement",
                    "measurement_node",
                    "control_interval_s",
                    "maximum_dt_s",
                    "minimum_dt_s",
                    "max_speed_change_rad_s",
                )
            },
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["network"] = LiquidNetwork.from_dict(value["network"])
        value["drive"] = RotatingDrive(**value["drive"])
        if value.get("controller") is not None:
            value["controller"] = PIDController(**value["controller"])
        return cls(**value)

    @property
    def model_id(self):
        return _digest({"model": "midpoint_shaft_quasi_steady_network_v1", **self.to_dict()})

    def _hydraulics(self, speed, options, trial=False):
        network = self._trial_network if trial else self.network
        controls = {"pump_status": {self.pump.name: speed > 0}}
        if speed > 0:
            controls["pump_speed_ratios"] = {self.pump.name: speed / self.nominal_speed_rad_s}
        result = network.solve(**controls, **options)
        extra = self.no_flow_power_w_at_nominal * (speed / self.nominal_speed_rad_s) ** 3
        base_power = result.pump_shaft_power_w.get(self.pump.name, 0.0)
        if extra > 0:
            power = base_power + extra
            if (
                not trial
                and self.pump.max_shaft_power_w is not None
                and power > self.pump.max_shaft_power_w
            ):
                raise EquipmentOperatingError(
                    "Pump total shaft power including no-flow loss exceeds limit"
                )
            hydraulic = result.pump_hydraulic_power_w.get(self.pump.name, 0.0)
            result = replace(
                result,
                pump_shaft_power_w={**result.pump_shaft_power_w, self.pump.name: power},
                pump_efficiencies={**result.pump_efficiencies, self.pump.name: hydraulic / power},
            )
        power = result.pump_shaft_power_w.get(self.pump.name, 0.0)
        if power < -1e-10:
            raise EquipmentOperatingError("Pump turbine operation is unsupported")
        scale = max(
            1,
            sum(abs(x) for x in result.pump_hydraulic_power_w.values()),
            sum(abs(x) for x in result.pipe_dissipation_w.values()),
            sum(abs(x) for x in result.valve_dissipation_w.values()),
        )
        if abs(result.power_balance_error_w) > 1e-5 + 1e-7 * scale:
            raise ConservationError("Coupled hydraulic power balance failed")
        return result, max(0.0, power) / speed if speed else 0.0

    def _measurement(self, speed, hydraulics):
        if self.measurement == "speed":
            return speed
        if self.measurement == "flow":
            return hydraulics.pump_volume_rates_m3_s[self.pump.name]
        return hydraulics.pressures_pa[self.measurement_node]

    def initialize(self, angular_velocity_rad_s=0.0, time_s=0.0, network_options=None):
        self.drive.initialize(angular_velocity_rad_s, time_s)
        options = self._options(network_options)
        hydraulic, _ = self._hydraulics(angular_velocity_rad_s, options)
        pid = (
            None
            if self.controller is None
            else self.controller.initialize(
                self._measurement(angular_velocity_rad_s, hydraulic), mode="off"
            )
        )
        return PumpDriveState(angular_velocity_rad_s, time_s, pid, 0.0, self.model_id)

    @staticmethod
    def _options(options):
        options = dict(options or {})
        if set(options) - {"boundary_pressures_pa", "valve_openings"}:
            raise ValueError("Only boundary pressures and valve openings may be overridden")
        return options

    def _motor(self, command, peak_speed):
        drive = self.drive
        torque = min(drive.max_motor_torque_nm, max(-drive.max_braking_torque_nm, command))
        if torque > 0 and drive.max_electrical_draw_w == 0:
            torque = 0.0
        if torque < 0 and drive.max_electrical_export_w == 0 and drive.regeneration_efficiency > 0:
            torque = 0.0
        if peak_speed > 0:
            if torque > 0 and drive.max_electrical_draw_w is not None:
                torque = min(
                    torque, drive.max_electrical_draw_w * drive.motor_efficiency / peak_speed
                )
            if (
                torque < 0
                and drive.max_electrical_export_w is not None
                and drive.regeneration_efficiency > 0
            ):
                torque = max(
                    torque,
                    -drive.max_electrical_export_w / (drive.regeneration_efficiency * peak_speed),
                )
        return torque

    def _step(self, speed, duration, command, options):
        drive = self.drive
        evaluations = 0
        cache = {}

        def evaluate(end):
            nonlocal evaluations
            mid = 0.5 * (speed + end)
            if mid not in cache:
                cache[mid] = self._hydraulics(mid, options, trial=True)
                evaluations += 1
            network, load = cache[mid]
            torque = self._motor(command, max(speed, end))
            residual = (
                drive.inertia_kg_m2 * (end - speed) / duration
                - torque
                + load
                + drive.viscous_drag_nm_s * mid
            )
            return residual, torque, load, network

        lower = evaluate(0.0)
        moving = duration
        limited_speed = False
        if lower[0] > 0:
            if command > 0:
                raise HydraulicConvergenceError(
                    "Midpoint step overshoots a positive-speed equilibrium"
                )
            end = 0.0
            net_brake = lower[2] + drive.viscous_drag_nm_s * speed / 2 - lower[1]
            moving = drive.inertia_kg_m2 * speed / net_brake if speed else 0.0
            torque = lower[1]
        else:
            upper_speed = min(
                drive.max_speed_rad_s,
                speed
                + duration
                * max(0.0, min(command, drive.max_motor_torque_nm))
                / drive.inertia_kg_m2,
            )
            if command > 0 and upper_speed < drive.max_speed_rad_s:
                upper_speed = min(drive.max_speed_rad_s, math.nextafter(upper_speed, math.inf))
            upper = evaluate(upper_speed)
            if upper[0] < 0:
                if upper_speed != drive.max_speed_rad_s:
                    raise HydraulicConvergenceError("Shaft root bracket failed")
                end = upper_speed
                torque = (
                    drive.inertia_kg_m2 * (end - speed) / duration
                    + upper[2]
                    + drive.viscous_drag_nm_s * (end + speed) / 2
                )
                limited_speed = True
            elif lower[0] == 0:
                end = 0.0
                torque = lower[1]
            elif upper[0] == 0:
                end = upper_speed
                torque = upper[1]
            else:
                end = brentq(
                    lambda value: evaluate(value)[0], 0.0, upper_speed, xtol=1e-10, rtol=1e-12
                )
                torque = evaluate(end)[1]
        if abs(end - speed) > self.max_speed_change_rad_s * (1 + 1e-12):
            raise HydraulicConvergenceError("Shaft state-change accuracy guard")
        midpoint = 0.5 * (speed + end)
        mid_network, load = self._hydraulics(midpoint, options)
        end_network, _ = self._hydraulics(end, options)
        evaluations += 2
        cap = self._motor(command, max(speed, end))
        if not min(0.0, cap) - 1e-9 <= torque <= max(0.0, cap) + 1e-9:
            raise ConservationError("Accepted motor torque violates actuator envelope")
        hydraulic_work = moving * mid_network.pump_hydraulic_power_w.get(self.pump.name, 0.0)
        pump_work = moving * mid_network.pump_shaft_power_w.get(self.pump.name, 0.0)
        motor_work = torque * midpoint * moving
        drag = drive.viscous_drag_nm_s * midpoint**2 * moving
        kinetic = 0.5 * drive.inertia_kg_m2 * (end - speed) * (end + speed)
        electrical = (
            motor_work / drive.motor_efficiency
            if motor_work >= 0
            else motor_work * drive.regeneration_efficiency
        )
        motor_heat = electrical - motor_work
        pump_heat = pump_work - hydraulic_work
        shaft_residual = motor_work - pump_work - drag - kinetic

        # Heat/export accounting for the entire isothermal hydraulic network.
        def hydraulic_budget(network):
            passive = sum(network.pipe_dissipation_w.values()) + sum(
                network.valve_dissipation_w.values()
            )
            boundary = -sum(
                (
                    network.pressures_pa[n.name]
                    + self.network.fluid.density_kg_m3 * 9.80665 * n.elevation_m
                )
                * network.boundary_mass_rates_kg_s.get(n.name, 0.0)
                / self.network.fluid.density_kg_m3
                for n in self.network.nodes
            )
            return passive, boundary

        passive, boundary = hydraulic_budget(mid_network)
        passive *= moving
        boundary *= moving
        if moving < duration:
            rest_passive, rest_boundary = hydraulic_budget(end_network)
            passive += (duration - moving) * rest_passive
            boundary += (duration - moving) * rest_boundary
        total_residual = electrical - kinetic - motor_heat - drag - pump_heat - passive - boundary
        scale = max(
            1.0, abs(electrical) + abs(kinetic) + abs(pump_work) + abs(passive) + abs(boundary)
        )
        if (
            not all(math.isfinite(x) for x in (shaft_residual, total_residual, scale))
            or min(pump_heat, motor_heat, drag, passive) < -1e-8
            or abs(shaft_residual) > 1e-7 + 1e-8 * scale
            or abs(total_residual) > 1e-6 + 2e-7 * scale
        ):
            raise ConservationError("Coupled shaft/network energy balance failed")
        record = {
            "dt_s": duration,
            "moving_duration_s": moving,
            "initial_speed_rad_s": speed,
            "final_speed_rad_s": end,
            "command_torque_nm": command,
            "motor_torque_nm": torque,
            "pump_load_torque_nm": load,
            "speed_limited": limited_speed,
            "torque_limited": abs(torque - command) > 1e-10 * max(1.0, abs(command)),
            "peak_electrical_draw_w": max(0.0, torque) * max(speed, end) / drive.motor_efficiency,
            "peak_electrical_export_w": max(0.0, -torque)
            * max(speed, end)
            * drive.regeneration_efficiency,
            "rest_constraint_impulse_nm_s": max(0.0, -torque) * (duration - moving),
            "electrical_energy_j": electrical,
            "motor_work_j": motor_work,
            "pump_work_j": pump_work,
            "hydraulic_work_j": hydraulic_work,
            "pump_heat_j": pump_heat,
            "motor_heat_j": motor_heat,
            "drag_heat_j": drag,
            "passive_heat_j": passive,
            "boundary_work_out_j": boundary,
            "kinetic_energy_change_j": kinetic,
            "shaft_energy_residual_j": shaft_residual,
            "energy_residual_j": total_residual,
            "network_evaluations": evaluations,
            "midpoint_flow_m3_s": mid_network.pump_volume_rates_m3_s[self.pump.name],
            "midpoint_pressures_pa": mid_network.pressures_pa,
            "endpoint": asdict(end_network),
        }
        return end, record

    def advance(
        self,
        state,
        duration_s,
        mode="auto",
        setpoint=0.0,
        motor_torque_nm=0.0,
        network_options=None,
        max_substeps=100000,
        reset_trip=False,
    ):
        if not isinstance(state, PumpDriveState) or state.model_id != self.model_id:
            raise ValueError("Pump-drive state/model mismatch")
        if (self.controller is None) != (state.controller_state is None):
            raise ValueError("Pump-drive controller state mismatch")
        self.drive.initialize(state.angular_velocity_rad_s, state.time_s)
        duration = _number(duration_s, "duration_s", positive=True)
        target = _number(setpoint, "setpoint")
        manual = _number(motor_torque_nm, "motor_torque_nm")
        if mode not in ("auto", "manual", "off", "trip") or (
            mode == "auto" and self.controller is None
        ):
            raise ValueError("Automatic mode requires a PIDController")
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("max_substeps must be positive integer")
        if not math.isfinite(state.time_s + duration) or state.time_s + duration <= state.time_s:
            raise ValueError("Pump-drive interval cannot advance representable time")
        if not isinstance(reset_trip, bool) or (reset_trip and mode == "trip"):
            raise ValueError("Invalid trip reset")
        tripped = (state.tripped and not reset_trip) or mode == "trip"
        requested_mode = mode
        mode = "off" if tripped else mode
        options = self._options(network_options)
        if mode == "auto" and self.measurement == "pressure":
            node = self.network.nodes[self.network._index[self.measurement_node]]
            fixed = (options.get("boundary_pressures_pa") or {}).get(
                self.measurement_node, node.pressure_pa
            )
            if fixed is not None:
                raise ValueError("Cannot regulate an externally fixed pressure boundary")
        speed, pid, applied = (
            state.angular_velocity_rad_s,
            state.controller_state,
            state.applied_torque_nm,
        )
        elapsed, failures = 0.0, 0
        steps, samples = [], []
        while elapsed < duration:
            sample_end = min(duration, (len(samples) + 1) * self.control_interval_s)
            sample_dt = sample_end - elapsed
            before, _ = self._hydraulics(speed, options)
            measurement = self._measurement(speed, before)
            if self.controller is not None:
                response = self.controller.step(
                    pid, target, measurement, sample_dt, mode, manual, applied
                )
                pid = PIDState(**response["state"])
                command = response["output"]
            else:
                command = 0.0 if mode == "off" else manual
                response = {"output": command}
            samples.append(
                {
                    "time_s": state.time_s + elapsed,
                    "dt_s": sample_dt,
                    "measurement": measurement,
                    **response,
                }
            )
            within, torque_integral = 0.0, 0.0
            step_dt = min(self.maximum_dt_s, sample_dt)
            while within < sample_dt:
                if len(steps) + failures >= max_substeps:
                    raise HydraulicConvergenceError("Pump-drive substep budget exhausted")
                remaining = sample_dt - within
                # Merge an ULP-sized remainder into this step; no time/energy
                # interval is discarded and the PID is not sampled twice.
                roundoff = 8 * max(math.ulp(sample_end), math.ulp(elapsed), math.ulp(step_dt))
                dt = remaining if remaining <= step_dt + roundoff else step_dt
                try:
                    end, record = self._step(speed, dt, command, options)
                except HydraulicConvergenceError:
                    failures += 1
                    if dt / 2 < self.minimum_dt_s:
                        raise
                    step_dt = dt / 2
                    continue
                following = math.fsum([within, dt])
                if (
                    following <= within
                    or state.time_s + elapsed + following <= state.time_s + elapsed + within
                ):
                    raise ValueError("Pump-drive substep cannot advance representable time")
                steps.append({"time_s": state.time_s + elapsed + within, **record})
                speed = end
                torque_integral += record["motor_torque_nm"] * dt
                within = following
                step_dt = min(self.maximum_dt_s, dt * 1.5)
            applied = torque_integral / sample_dt
            # A convex time average is within actuator bounds; normalize only
            # its floating-point roundoff before feeding it back to the PID.
            low = (
                -self.drive.max_braking_torque_nm
                if self.controller is None
                else self.controller.min_output
            )
            high = (
                self.drive.max_motor_torque_nm
                if self.controller is None
                else self.controller.max_output
            )
            applied = min(high, max(low, applied))
            samples[-1]["applied_average_torque_nm"] = applied
            elapsed = sample_end
        final = PumpDriveState(speed, state.time_s + duration, pid, applied, self.model_id, tripped)
        endpoint = self._hydraulics(speed, options)[0]
        measured = self._measurement(speed, endpoint)
        return {
            "state": final.to_dict(),
            "measurement": measured,
            "control_error": target - measured if mode == "auto" else None,
            "steps": steps,
            "control_samples": samples,
            "failed_steps": failures,
            "requested_mode": requested_mode,
            "effective_mode": mode,
            "endpoint": asdict(endpoint),
        }
