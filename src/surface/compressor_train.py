"""Dry ideal-gas mapped compressor, dynamic plenum/shaft and cooled recycle.

Stable-map model, not a post-surge oscillation model. Below mapped speed an
explicit unloaded standby/off mode is required. Suction p/T are reservoirs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from .actuator import ConnectionActuator
from .compressor_map import CompressorEnvelopeError, CompressorMap, MappedGasCompressor
from .gas import IdealGas
from .gas_valve import ideal_gas_nozzle_flow
from .liquid import ConservationError, HydraulicConvergenceError, _number
from .operation import _digest
from .pid import PIDController, PIDState
from .rotating_drive import RotatingDrive


@dataclass(frozen=True)
class CompressorTrainState:
    mass_kg: float
    internal_energy_j: float
    angular_velocity_rad_s: float
    time_s: float
    recycle_position: float
    speed_pid: PIDState | None
    anti_surge_pid: PIDState
    applied_motor_torque_nm: float
    model_id: str
    tripped: bool = False

    def __post_init__(self):
        for key in ("mass_kg", "internal_energy_j"):
            _number(getattr(self, key), key, positive=True)
        for key in ("angular_velocity_rad_s", "time_s"):
            _number(getattr(self, key), key, nonnegative=True)
        ConnectionActuator.position(self.recycle_position)
        _number(self.applied_motor_torque_nm, "applied_motor_torque_nm")
        if not isinstance(self.tripped, bool) or not isinstance(self.anti_surge_pid, PIDState):
            raise TypeError("Invalid compressor trip/controller state")
        if self.speed_pid is not None and not isinstance(self.speed_pid, PIDState):
            raise TypeError("Invalid speed PID state")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("speed_pid", "anti_surge_pid"):
            if value.get(key) is not None:
                value[key] = PIDState(**value[key])
        return cls(**value)


class DynamicCompressorTrain:
    def __init__(
        self,
        compressor,
        drive,
        recycle_actuator,
        anti_surge_controller,
        plenum_volume_m3,
        suction_pressure_pa,
        suction_temperature_k,
        export_back_pressure_pa,
        export_area_m2,
        recycle_area_m2,
        speed_controller=None,
        target_surge_margin=0.25,
        fast_open_margin=0.1,
        control_interval_s=0.05,
        maximum_dt_s=0.01,
        minimum_dt_s=1e-7,
        heat_transfer_w_k=0.0,
        ambient_temperature_k=300.0,
        min_temperature_k=150.0,
        max_temperature_k=700.0,
        max_pressure_pa=1e7,
        max_recycle_cooling_w=None,
        max_recycle_heating_w=None,
        max_fractional_mass_change=0.05,
        max_temperature_change_k=5.0,
        max_speed_change_rad_s=2.0,
    ):
        if not isinstance(compressor, MappedGasCompressor) or not isinstance(drive, RotatingDrive):
            raise TypeError("Expected mapped compressor and RotatingDrive")
        if (
            not isinstance(recycle_actuator, ConnectionActuator)
            or recycle_actuator.fail_position != 1
        ):
            raise ValueError("Recycle actuator must have fail-open position 1")
        if (
            not isinstance(anti_surge_controller, PIDController)
            or anti_surge_controller.min_output != 0
            or anti_surge_controller.max_output != 1
            or anti_surge_controller.direction != 1
        ):
            raise ValueError("Anti-surge PID must open on low margin, with output [0,1]")
        if speed_controller is not None and (
            not isinstance(speed_controller, PIDController)
            or speed_controller.min_output < -drive.max_braking_torque_nm
            or speed_controller.max_output > drive.max_motor_torque_nm
        ):
            raise ValueError("Speed PID bounds must fit motor torque bounds")
        self.compressor, self.drive, self.recycle_actuator = compressor, drive, recycle_actuator
        self.anti_surge_controller, self.speed_controller = anti_surge_controller, speed_controller
        values = locals().copy()
        for key in (
            "plenum_volume_m3",
            "suction_pressure_pa",
            "suction_temperature_k",
            "export_back_pressure_pa",
            "export_area_m2",
            "recycle_area_m2",
            "target_surge_margin",
            "control_interval_s",
            "maximum_dt_s",
            "minimum_dt_s",
            "ambient_temperature_k",
            "min_temperature_k",
            "max_temperature_k",
            "max_pressure_pa",
            "max_fractional_mass_change",
            "max_temperature_change_k",
            "max_speed_change_rad_s",
        ):
            setattr(self, key, float(_number(values[key], key, positive=True)))
        for key in ("heat_transfer_w_k", "fast_open_margin"):
            setattr(self, key, float(_number(values[key], key, nonnegative=True)))
        for key in ("max_recycle_cooling_w", "max_recycle_heating_w"):
            value = values[key]
            if value is not None:
                _number(value, key, nonnegative=True)
            setattr(self, key, value)
        if (
            not self.min_temperature_k < self.max_temperature_k
            or not self.min_temperature_k <= self.suction_temperature_k <= self.max_temperature_k
        ):
            raise ValueError("Invalid gas temperature bounds")
        if (
            self.minimum_dt_s > self.maximum_dt_s
            or self.max_fractional_mass_change > 1
            or self.fast_open_margin > self.target_surge_margin
        ):
            raise ValueError("Invalid compressor time/margin guards")

    @property
    def gas(self):
        return self.compressor.gas

    @property
    def cv(self):
        return self.gas.cp_j_kg_k - self.gas.specific_gas_constant_j_kg_k

    def to_dict(self):
        return {
            "compressor": {
                "map": self.compressor.map.to_dict(),
                "gas": asdict(self.gas),
                "mechanical_efficiency": self.compressor.mechanical_efficiency,
                "max_discharge_temperature_k": self.compressor.max_discharge_temperature_k,
                "max_shaft_power_w": self.compressor.max_shaft_power_w,
            },
            "drive": asdict(self.drive),
            "recycle_actuator": asdict(self.recycle_actuator),
            "anti_surge_controller": asdict(self.anti_surge_controller),
            "speed_controller": None
            if self.speed_controller is None
            else asdict(self.speed_controller),
            **{
                key: value
                for key, value in self.__dict__.items()
                if key
                not in {
                    "compressor",
                    "drive",
                    "recycle_actuator",
                    "anti_surge_controller",
                    "speed_controller",
                }
            },
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        comp = dict(value["compressor"])
        comp["map"] = CompressorMap(**comp["map"])
        comp["gas"] = IdealGas(**comp["gas"])
        value["compressor"] = MappedGasCompressor(**comp)
        for key, kind in (
            ("drive", RotatingDrive),
            ("recycle_actuator", ConnectionActuator),
            ("anti_surge_controller", PIDController),
            ("speed_controller", PIDController),
        ):
            if value.get(key) is not None:
                value[key] = kind(**value[key])
        return cls(**value)

    @property
    def model_id(self):
        return _digest({"model": "mapped_plenum_shaft_recycle_v1", **self.to_dict()})

    def properties(self, mass, energy):
        temperature = energy / (mass * self.cv)
        pressure = (self.gas.heat_capacity_ratio - 1) * energy / self.plenum_volume_m3
        return pressure, temperature

    def _entropy(self, mass, energy):
        pressure, temperature = self.properties(mass, energy)
        return mass * (
            self.gas.cp_j_kg_k * math.log(temperature / 300.0)
            - self.gas.specific_gas_constant_j_kg_k * math.log(pressure / 1e5)
        )

    def _check(self, state):
        if not isinstance(state, CompressorTrainState) or state.model_id != self.model_id:
            raise ValueError("Compressor train state/model mismatch")
        if state.speed_pid is not None and (
            self.speed_controller is None
            or state.speed_pid.model_id != self.speed_controller.model_id
        ):
            raise ValueError("Speed PID identity mismatch")
        if (state.speed_pid is None) != (
            self.speed_controller is None
        ) or state.anti_surge_pid.model_id != self.anti_surge_controller.model_id:
            raise ValueError("Compressor controller state mismatch")
        p, t = self.properties(state.mass_kg, state.internal_energy_j)
        if p > self.max_pressure_pa or not self.min_temperature_k <= t <= self.max_temperature_k:
            raise CompressorEnvelopeError("Plenum pressure/temperature outside bounds", "plenum")
        self.drive.initialize(state.angular_velocity_rad_s, state.time_s)

    def initialize(
        self,
        pressure_pa=None,
        temperature_k=None,
        angular_velocity_rad_s=0.0,
        recycle_position=1.0,
        time_s=0.0,
    ):
        p = (
            self.suction_pressure_pa
            if pressure_pa is None
            else _number(pressure_pa, "pressure_pa", positive=True)
        )
        t = (
            self.suction_temperature_k
            if temperature_k is None
            else _number(temperature_k, "temperature_k", positive=True)
        )
        mass = p * self.plenum_volume_m3 / (self.gas.specific_gas_constant_j_kg_k * t)
        anti = self.anti_surge_controller.initialize(
            0.0, recycle_position, setpoint=self.target_surge_margin
        )
        speed = (
            None
            if self.speed_controller is None
            else self.speed_controller.initialize(angular_velocity_rad_s, mode="off")
        )
        state = CompressorTrainState(
            mass,
            mass * self.cv * t,
            angular_velocity_rad_s,
            time_s,
            recycle_position,
            speed,
            anti,
            0.0,
            self.model_id,
        )
        self._check(state)
        return state

    def _motor(self, command, peak):
        d = self.drive
        torque = min(d.max_motor_torque_nm, max(-d.max_braking_torque_nm, command))
        if torque > 0 and d.max_electrical_draw_w == 0:
            torque = 0.0
        if torque < 0 and d.max_electrical_export_w == 0 and d.regeneration_efficiency > 0:
            torque = 0.0
        if peak > 0:
            if torque > 0 and d.max_electrical_draw_w is not None:
                torque = min(torque, d.max_electrical_draw_w * d.motor_efficiency / peak)
            if (
                torque < 0
                and d.max_electrical_export_w is not None
                and d.regeneration_efficiency > 0
            ):
                torque = max(
                    torque, -d.max_electrical_export_w / (d.regeneration_efficiency * peak)
                )
        return torque

    def _flows(self, mass, energy, speed, position, export_opening, running, strict):
        pressure, temperature = self.properties(mass, energy)
        comp = (
            self.compressor.run(
                speed, self.suction_pressure_pa, self.suction_temperature_k, pressure, strict
            )
            if running
            else None
        )
        inlet = 0.0 if comp is None else comp["mass_rate_kg_s"]
        exported = (
            0.0
            if pressure <= self.export_back_pressure_pa
            else ideal_gas_nozzle_flow(
                pressure,
                temperature,
                self.export_back_pressure_pa,
                self.gas,
                self.export_area_m2,
                opening=export_opening,
            )["mass_rate_kg_s"]
        )
        recycle = (
            0.0
            if pressure <= self.suction_pressure_pa
            else ideal_gas_nozzle_flow(
                pressure,
                temperature,
                self.suction_pressure_pa,
                self.gas,
                self.recycle_area_m2,
                opening=position,
            )["mass_rate_kg_s"]
        )
        conditioning = recycle * self.gas.cp_j_kg_k * (temperature - self.suction_temperature_k)
        if strict and (
            (self.max_recycle_cooling_w is not None and conditioning > self.max_recycle_cooling_w)
            or (
                self.max_recycle_heating_w is not None
                and -conditioning > self.max_recycle_heating_w
            )
        ):
            raise CompressorEnvelopeError("Recycle conditioning duty limit", "recycle_duty")
        return {
            "pressure_pa": pressure,
            "temperature_k": temperature,
            "compressor": comp,
            "compressor_mass_rate_kg_s": inlet,
            "export_mass_rate_kg_s": exported,
            "recycle_mass_rate_kg_s": recycle,
            "net_suction_mass_rate_kg_s": inlet - recycle,
            "conditioning_heat_w": conditioning,
            "wall_heat_w": self.heat_transfer_w_k * (temperature - self.ambient_temperature_k),
        }

    def _step(self, state, dt, command, recycle_target, actuator_mode, export_opening, running):
        valve = self.recycle_actuator.step(
            state.recycle_position, recycle_target, dt, actuator_mode
        )["final_position"]
        d = self.drive
        old_speed = state.angular_velocity_rad_s
        exact = None
        if not running:
            peak = min(
                d.max_speed_rad_s,
                old_speed + dt * max(0.0, min(command, d.max_motor_torque_nm)) / d.inertia_kg_m2,
            )
            torque = self._motor(command, peak)
            if d.viscous_drag_nm_s == 0:
                speed_torque = d.inertia_kg_m2 * (d.max_speed_rad_s - old_speed) / dt
            else:
                decay = math.exp(-d.viscous_drag_nm_s * dt / d.inertia_kg_m2)
                speed_torque = (
                    d.viscous_drag_nm_s
                    * (d.max_speed_rad_s - old_speed * decay)
                    / (-math.expm1(-d.viscous_drag_nm_s * dt / d.inertia_kg_m2))
                )
            if torque > 0:
                torque = min(torque, speed_torque * (1 - 8 * np.finfo(float).eps))
            exact = d.advance(d.initialize(old_speed), dt, torque)
        omega_scale = max(1.0, old_speed)
        energy_scale = max(1.0, self.suction_pressure_pa * self.plenum_volume_m3 * 0.001, dt * 1e4)
        mass_scale = max(state.mass_kg, 1e-6)
        max_speed = d.max_speed_rad_s
        if running:
            max_speed = min(
                max_speed,
                self.compressor.map.corrected_speed_rad_s[-1]
                * math.sqrt(
                    self.suction_temperature_k / self.compressor.map.reference_temperature_k
                ),
            )
        n = 3 if running else 2
        guess = [1.0, 1.0] + ([old_speed / omega_scale] if running else [])

        def evaluate(x, strict=False):
            mass = x[0] * mass_scale
            energy = x[1] * state.internal_energy_j
            end_speed = x[2] * omega_scale if running else exact["state"]["angular_velocity_rad_s"]
            midpoint = 0.5 * (old_speed + end_speed)
            flows = self._flows(mass, energy, midpoint, valve, export_opening, running, strict)
            comp = flows["compressor"]
            shaft = 0.0 if comp is None else comp["shaft_power_w"]
            h_in = self.gas.cp_j_kg_k * (
                self.suction_temperature_k if comp is None else comp["discharge_temperature_k"]
            )
            out = flows["export_mass_rate_kg_s"] + flows["recycle_mass_rate_kg_s"]
            res = [
                (mass - state.mass_kg - dt * (flows["compressor_mass_rate_kg_s"] - out))
                / mass_scale,
                (
                    energy
                    - state.internal_energy_j
                    - dt
                    * (
                        flows["compressor_mass_rate_kg_s"] * h_in
                        - out * self.gas.cp_j_kg_k * flows["temperature_k"]
                        - flows["wall_heat_w"]
                    )
                )
                / energy_scale,
            ]
            torque = (
                exact["motor_work_j"]
                if not running
                else self._motor(command, max(old_speed, end_speed))
            )
            if running:
                load = shaft / midpoint if midpoint > 0 else 0.0
                limit = (
                    d.inertia_kg_m2 * (max_speed - old_speed) / dt
                    + load
                    + d.viscous_drag_nm_s * midpoint
                )
                torque = min(torque, limit)
                res.append(
                    (
                        d.inertia_kg_m2 * (end_speed - old_speed)
                        - dt * (torque - load - d.viscous_drag_nm_s * midpoint)
                    )
                    / (d.inertia_kg_m2 * omega_scale)
                )
            return np.array(res), mass, energy, end_speed, torque, flows

        upper = [
            1e6,
            max(
                2.0,
                self.max_pressure_pa
                * self.plenum_volume_m3
                / ((self.gas.heat_capacity_ratio - 1) * state.internal_energy_j),
            ),
        ] + ([max_speed / omega_scale] if running else [])
        fit = least_squares(
            lambda x: evaluate(x)[0],
            guess,
            bounds=([1e-12] * n, upper),
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=100,
        )
        residual, mass, energy, speed, torque, flows = evaluate(fit.x)
        if not fit.success or np.max(abs(residual)) > 1e-8:
            raise HydraulicConvergenceError("Compressor/plenum/shaft solve failed")
        residual, mass, energy, speed, torque, flows = evaluate(fit.x, True)
        if running:
            self._flows(mass, energy, speed, valve, export_opening, True, True)
        p, t = self.properties(mass, energy)
        _, old_t = self.properties(state.mass_kg, state.internal_energy_j)
        if (
            abs(mass - state.mass_kg) / state.mass_kg > self.max_fractional_mass_change
            or abs(t - old_t) > self.max_temperature_change_k
            or abs(speed - old_speed) > self.max_speed_change_rad_s
        ):
            raise HydraulicConvergenceError("Compressor state-change accuracy guard")
        if not self.min_temperature_k <= t <= self.max_temperature_k or p > self.max_pressure_pa:
            raise CompressorEnvelopeError("Plenum state limit", "plenum")
        comp = flows["compressor"]
        shaft = 0.0 if comp is None else comp["shaft_power_w"]
        gas_power = 0.0 if comp is None else comp["gas_power_w"]
        if running:
            midpoint = 0.5 * (speed + old_speed)
            motor_work = dt * torque * midpoint
            drag = dt * d.viscous_drag_nm_s * midpoint**2
            electrical = (
                motor_work / d.motor_efficiency
                if motor_work >= 0
                else motor_work * d.regeneration_efficiency
            )
            motor_heat = electrical - motor_work
        else:
            motor_work = exact["motor_work_j"]
            drag = exact["drag_heat_j"]
            electrical = exact["electrical_energy_j"]
            motor_heat = exact["motor_heat_j"]
            torque = self._motor(command, peak)
            if torque > 0:
                torque = min(torque, speed_torque * (1 - 8 * np.finfo(float).eps))
        kinetic = 0.5 * d.inertia_kg_m2 * (speed - old_speed) * (speed + old_speed)
        fresh = dt * flows["net_suction_mass_rate_kg_s"]
        exported = dt * flows["export_mass_rate_kg_s"]
        inlet_h = self.gas.cp_j_kg_k * self.suction_temperature_k
        outlet_h = self.gas.cp_j_kg_k * t
        mechanical = dt * (shaft - gas_power)
        wall = dt * flows["wall_heat_w"]
        conditioning = dt * flows["conditioning_heat_w"]
        mass_error = mass - state.mass_kg - fresh + exported
        shaft_error = motor_work - dt * shaft - drag - kinetic
        error = (
            electrical
            + fresh * inlet_h
            - exported * outlet_h
            - (energy - state.internal_energy_j)
            - kinetic
            - motor_heat
            - drag
            - mechanical
            - wall
            - conditioning
        )
        scale = max(
            1.0,
            abs(electrical)
            + abs(fresh * inlet_h)
            + abs(exported * outlet_h)
            + abs(wall)
            + abs(conditioning),
        )
        if not all(
            math.isfinite(v)
            for v in (mass_error, shaft_error, error, scale, electrical, kinetic, conditioning)
        ):
            raise ConservationError("Nonfinite compressor transaction")
        if (
            abs(mass_error) > 1e-10 + 1e-8 * (abs(fresh) + abs(exported))
            or abs(shaft_error) > 1e-7 + 1e-8 * scale
            or abs(error) > 1e-5 + 2e-8 * scale
        ):
            raise ConservationError("Compressor mass/energy transaction balance failed")
        sin = self.gas.cp_j_kg_k * math.log(
            self.suction_temperature_k / 300.0
        ) - self.gas.specific_gas_constant_j_kg_k * math.log(self.suction_pressure_pa / 1e5)
        sout = self.gas.cp_j_kg_k * math.log(
            t / 300.0
        ) - self.gas.specific_gas_constant_j_kg_k * math.log(self.export_back_pressure_pa / 1e5)
        entropy = (
            self._entropy(mass, energy)
            - self._entropy(state.mass_kg, state.internal_energy_j)
            - fresh * sin
            + exported * sout
            + (wall + motor_heat + drag + mechanical) / self.ambient_temperature_k
            + conditioning / self.suction_temperature_k
        )
        if not math.isfinite(entropy):
            raise ConservationError("Nonfinite compressor entropy")
        if entropy < -1e-7 - 1e-8 * max(
            1.0,
            abs(fresh * sin)
            + abs(exported * sout)
            + abs(conditioning / self.suction_temperature_k),
        ):
            raise ConservationError("Compressor train entropy balance failed")
        return (
            mass,
            energy,
            speed,
            valve,
            torque,
            {
                "dt_s": dt,
                "initial_speed_rad_s": old_speed,
                "final_speed_rad_s": speed,
                "net_suction_mass_kg": fresh,
                "export_mass_kg": exported,
                "recycle_mass_kg": dt * flows["recycle_mass_rate_kg_s"],
                "mass_residual_kg": mass_error,
                "shaft_energy_residual_j": shaft_error,
                "energy_residual_j": error,
                "entropy_production_j_k": entropy,
                "electrical_energy_j": electrical,
                "motor_work_j": motor_work,
                "shaft_work_j": dt * shaft,
                "gas_work_j": dt * gas_power,
                "motor_heat_j": motor_heat,
                "drag_heat_j": drag,
                "mechanical_heat_j": mechanical,
                "wall_heat_j": wall,
                "conditioning_heat_j": conditioning,
                "suction_enthalpy_j": fresh * inlet_h,
                "export_enthalpy_j": exported * outlet_h,
                "gas_energy_change_j": energy - state.internal_energy_j,
                "kinetic_energy_change_j": kinetic,
                "pressure_pa": p,
                "temperature_k": t,
                "mass_kg": mass,
                "internal_energy_j": energy,
                "motor_torque_nm": torque,
                "peak_electrical_draw_w": max(0.0, torque)
                * max(speed, old_speed)
                / d.motor_efficiency,
                "peak_electrical_export_w": max(0.0, -torque)
                * max(speed, old_speed)
                * d.regeneration_efficiency,
                "motor_limited": bool(abs(torque - command) > 1e-10 * max(1.0, abs(command))),
                "recycle_position": valve,
                "nonlinear_evaluations": fit.nfev,
                "operating_point": comp,
            },
        )

    def advance(
        self,
        state,
        duration_s,
        mode="run",
        speed_control_mode="auto",
        speed_setpoint_rad_s=0.0,
        motor_torque_nm=0.0,
        export_opening=1.0,
        anti_surge_mode="auto",
        manual_recycle_opening=1.0,
        recycle_actuator_mode="normal",
        reset_trip=False,
        max_substeps=100000,
    ):
        self._check(state)
        duration = _number(duration_s, "duration_s", positive=True)
        target = _number(speed_setpoint_rad_s, "speed_setpoint_rad_s", nonnegative=True)
        manual = _number(motor_torque_nm, "motor_torque_nm")
        export = ConnectionActuator.position(export_opening)
        recycle_manual = ConnectionActuator.position(manual_recycle_opening)
        if (
            mode not in ("run", "standby", "off", "trip")
            or speed_control_mode not in ("auto", "manual")
            or anti_surge_mode not in ("auto", "manual")
        ):
            raise ValueError("Unknown compressor control mode")
        if (
            recycle_actuator_mode not in ("normal", "stuck", "fail_safe")
            or not isinstance(reset_trip, bool)
            or (reset_trip and mode == "trip")
        ):
            raise ValueError("Invalid compressor actuator/reset mode")
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("max_substeps must be a positive integer")
        if not math.isfinite(state.time_s + duration) or state.time_s + duration <= state.time_s:
            raise ValueError("Compressor interval cannot advance representable time")
        if mode == "run" and speed_control_mode == "auto" and self.speed_controller is None:
            raise ValueError("Automatic speed mode requires a PID")
        tripped = (state.tripped and not reset_trip) or mode == "trip"
        effective = "off" if tripped else mode
        running = effective == "run"
        minimum_speed = self.compressor.map.corrected_speed_rad_s[0] * math.sqrt(
            self.suction_temperature_k / self.compressor.map.reference_temperature_k
        )
        if running and speed_control_mode == "auto" and target < minimum_speed:
            raise ValueError("Run speed setpoint below mapped range; use unloaded standby/off")
        if running:
            self._flows(
                state.mass_kg,
                state.internal_energy_j,
                state.angular_velocity_rad_s,
                state.recycle_position,
                export,
                True,
                True,
            )
        sp, anti = state.speed_pid, state.anti_surge_pid
        current = state
        elapsed = 0.0
        steps = []
        samples = []
        failures = 0
        while elapsed < duration:
            sample_end = min(duration, (len(samples) + 1) * self.control_interval_s)
            sample_dt = sample_end - elapsed
            old_flows = self._flows(
                current.mass_kg,
                current.internal_energy_j,
                current.angular_velocity_rad_s,
                current.recycle_position,
                export,
                running,
                True,
            )
            margin = 0.0 if not running else old_flows["compressor"]["surge_margin"]
            if self.speed_controller is None:
                command = manual if effective in ("run", "standby") else 0.0
                speed_response = None
            else:
                speed_mode = (
                    speed_control_mode
                    if running
                    else ("manual" if effective == "standby" else "off")
                )
                speed_response = self.speed_controller.step(
                    sp,
                    target,
                    current.angular_velocity_rad_s,
                    sample_dt,
                    speed_mode,
                    manual,
                    current.applied_motor_torque_nm,
                )
                sp = PIDState(**speed_response["state"])
                command = speed_response["output"]
            anti_mode = anti_surge_mode if running else "manual"
            anti_response = self.anti_surge_controller.step(
                anti,
                self.target_surge_margin,
                margin,
                sample_dt,
                anti_mode,
                recycle_manual if running else 1.0,
                current.recycle_position,
            )
            anti = PIDState(**anti_response["state"])
            override = running and anti_mode == "auto" and margin < self.fast_open_margin
            valve_target = 1.0 if override or not running else anti_response["output"]
            actuator_mode = recycle_actuator_mode
            if not running and actuator_mode != "stuck":
                actuator_mode = "fail_safe"
            samples.append(
                {
                    "time_s": state.time_s + elapsed,
                    "dt_s": sample_dt,
                    "measured_speed_rad_s": current.angular_velocity_rad_s,
                    "measured_recycle_position": current.recycle_position,
                    "speed": speed_response,
                    "anti_surge": anti_response,
                    "measured_surge_margin": margin if running else None,
                    "recycle_target": valve_target,
                    "fast_open_override": override,
                    "actuator_mode": actuator_mode,
                    "motor_command_nm": command,
                }
            )
            within = 0.0
            weighted = []
            step_dt = min(sample_dt, self.maximum_dt_s)
            while within < sample_dt:
                if len(steps) + failures >= max_substeps:
                    raise HydraulicConvergenceError("Compressor substep budget exhausted")
                remaining = sample_dt - within
                roundoff = 8 * max(math.ulp(sample_end), math.ulp(elapsed), math.ulp(step_dt))
                dt = remaining if remaining <= step_dt + roundoff else step_dt
                try:
                    mass, energy, speed, valve, torque, record = self._step(
                        current, dt, command, valve_target, actuator_mode, export, running
                    )
                except (HydraulicConvergenceError, CompressorEnvelopeError) as error:
                    failures += 1
                    if dt / 2 < self.minimum_dt_s:
                        if isinstance(error, CompressorEnvelopeError):
                            error.last_accepted_state = current.to_dict()
                        raise
                    step_dt = dt / 2
                    continue
                following = math.fsum([within, dt])
                if following <= within or state.time_s + elapsed + following <= current.time_s:
                    raise ValueError("Compressor substep cannot advance representable time")
                steps.append({"time_s": current.time_s, **record})
                weighted.append(torque * dt)
                current = CompressorTrainState(
                    mass,
                    energy,
                    speed,
                    state.time_s + elapsed + following,
                    valve,
                    sp,
                    anti,
                    torque,
                    self.model_id,
                    tripped,
                )
                within = following
                step_dt = min(self.maximum_dt_s, dt * 1.5)
            applied = math.fsum(weighted) / sample_dt
            low = (
                -self.drive.max_braking_torque_nm
                if self.speed_controller is None
                else self.speed_controller.min_output
            )
            high = (
                self.drive.max_motor_torque_nm
                if self.speed_controller is None
                else self.speed_controller.max_output
            )
            applied = min(high, max(low, applied))
            samples[-1]["applied_motor_torque_nm"] = applied
            current = CompressorTrainState(
                current.mass_kg,
                current.internal_energy_j,
                current.angular_velocity_rad_s,
                state.time_s + sample_end,
                current.recycle_position,
                sp,
                anti,
                applied,
                self.model_id,
                tripped,
            )
            elapsed = sample_end
        p, t = self.properties(current.mass_kg, current.internal_energy_j)
        endpoint = self._flows(
            current.mass_kg,
            current.internal_energy_j,
            current.angular_velocity_rad_s,
            current.recycle_position,
            export,
            running,
            True,
        )
        margin = None if not running else endpoint["compressor"]["surge_margin"]
        return {
            "state": current.to_dict(),
            "endpoint": endpoint,
            "surge_margin": margin,
            "surge_margin_error": None if margin is None else self.target_surge_margin - margin,
            "speed_error_rad_s": target - current.angular_velocity_rad_s
            if running and speed_control_mode == "auto"
            else None,
            "steps": steps,
            "control_samples": samples,
            "failed_steps": failures,
            "requested_mode": mode,
            "effective_mode": effective,
            "pressure_pa": p,
            "temperature_k": t,
            "duration_s": duration,
        }
