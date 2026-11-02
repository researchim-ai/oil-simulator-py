"""Parallel pump shafts coupled through one quasi-steady liquid network.

Pressure-driven single-liquid plumbing; no pipe inventory or water hammer.
Each shaft has its own inertia, PID, limits and stopping event. Dispatch
reuses steady station candidates but prices full steady electrical demand.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.optimize import brentq, least_squares

from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    LiquidNetwork,
    _name,
    _number,
)
from .operation import _digest
from .pid import PIDController, PIDState
from .rotating_drive import RotatingDrive
from .station import PumpStationControl, PumpStationOperatingError


def _pairs(value):
    return tuple(sorted(dict(value).items()))


@dataclass(frozen=True)
class PumpStationState:
    speeds_rad_s: tuple
    speed_pid_states: tuple
    pressure_pid_state: PIDState | None
    applied_torques_nm: tuple
    target_running: tuple
    unavailable: tuple
    time_s: float
    last_switch_time_s: float
    dispatch_signature: str
    dispatch_status: str
    model_id: str
    tripped: bool = False

    def __post_init__(self):
        for key in ("speeds_rad_s", "speed_pid_states", "applied_torques_nm"):
            object.__setattr__(self, key, _pairs(getattr(self, key)))
        for name, speed in self.speeds_rad_s:
            _name(name)
            _number(speed, "speed_rad_s", nonnegative=True)
        for name, torque in self.applied_torques_nm:
            _name(name)
            _number(torque, "torque_nm")
        for key in ("target_running", "unavailable"):
            value = tuple(getattr(self, key))
            if len(value) != len(set(value)):
                raise ValueError("Duplicate station unit names")
            object.__setattr__(self, key, value)
        _number(self.time_s, "time_s", nonnegative=True)
        _number(self.last_switch_time_s, "last_switch_time_s", nonnegative=True)
        if not isinstance(self.tripped, bool):
            raise TypeError("Trip state must be boolean")

    def to_dict(self):
        return {
            "target_running": list(self.target_running),
            "unavailable": list(self.unavailable),
            "speeds_rad_s": dict(self.speeds_rad_s),
            "speed_pid_states": {
                n: None if s is None else s.to_dict() for n, s in self.speed_pid_states
            },
            "pressure_pid_state": None
            if self.pressure_pid_state is None
            else self.pressure_pid_state.to_dict(),
            "applied_torques_nm": dict(self.applied_torques_nm),
            **{
                key: getattr(self, key)
                for key in (
                    "time_s",
                    "last_switch_time_s",
                    "dispatch_signature",
                    "dispatch_status",
                    "model_id",
                    "tripped",
                )
            },
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["speed_pid_states"] = {
            n: None if s is None else PIDState(**s) for n, s in value["speed_pid_states"].items()
        }
        if value.get("pressure_pid_state") is not None:
            value["pressure_pid_state"] = PIDState(**value["pressure_pid_state"])
        return cls(**value)


class DynamicPumpStation:
    def __init__(
        self,
        network,
        drives,
        nominal_speeds_rad_s,
        speed_controllers=None,
        pressure_controller=None,
        pressure_node=None,
        dispatch_policy=None,
        control_interval_s=0.2,
        maximum_dt_s=0.05,
        minimum_dt_s=1e-7,
        max_speed_change_rad_s=2.0,
        minimum_stage_dwell_s=5.0,
        dispatch_power_deadband_w=10.0,
        max_station_electrical_draw_w=None,
        max_station_electrical_export_w=None,
        no_flow_power_w_at_nominal=None,
    ):
        if not isinstance(network, LiquidNetwork) or not 1 <= len(network.pumps) <= 8:
            raise ValueError("Station requires a LiquidNetwork with 1..8 pumps")
        self.names = tuple(p.name for p in network.pumps)
        if len({(p.start, p.end) for p in network.pumps}) != 1 or any(
            not p.check_valve or not p.open for p in network.pumps
        ):
            raise ValueError("Dynamic station pumps must be open parallel units with check valves")
        if any(n.source_mass_rate_kg_s != 0 for n in network.nodes):
            raise ValueError("Startup requires pressure-driven boundaries, not fixed node demands")
        if set(drives) != set(self.names) or set(nominal_speeds_rad_s) != set(self.names):
            raise ValueError("Station drives/nominal speeds must match all pumps")
        if not all(isinstance(d, RotatingDrive) for d in drives.values()):
            raise TypeError("Expected RotatingDrive")
        for n in self.names:
            _number(nominal_speeds_rad_s[n], "nominal_speed_rad_s", positive=True)
        speed_controllers = dict(speed_controllers or {n: None for n in self.names})
        if set(speed_controllers) != set(self.names):
            raise ValueError("Speed controllers must match all pumps")
        for n, c in speed_controllers.items():
            if c is not None and (
                not isinstance(c, PIDController)
                or c.min_output < -drives[n].max_braking_torque_nm
                or c.max_output > drives[n].max_motor_torque_nm
            ):
                raise ValueError("Local PID limits must fit its drive")
        if pressure_controller is not None and (
            not isinstance(pressure_controller, PIDController)
            or pressure_controller.min_output != 0
        ):
            raise ValueError("Pressure PID output is nonnegative common speed ratio")
        if pressure_controller is not None and any(c is None for c in speed_controllers.values()):
            raise ValueError("Pressure control requires a speed PID for every station unit")
        if pressure_node not in network._index:
            raise ValueError("Station requires a known pressure node")
        if dispatch_policy is not None and (
            not isinstance(dispatch_policy, PumpStationControl)
            or set(dispatch_policy.pumps) != set(self.names)
            or dispatch_policy.node != pressure_node
        ):
            raise ValueError("Dispatch policy must match station pumps and pressure node")
        if (
            pressure_controller is not None
            and dispatch_policy is not None
            and pressure_controller.max_output < dispatch_policy.max_speed_ratio
        ):
            raise ValueError("Pressure PID range must cover dispatch speed range")
        self.network, self.drives, self.nominal_speeds_rad_s = (
            network,
            dict(drives),
            dict(nominal_speeds_rad_s),
        )
        self.speed_controllers, self.pressure_controller = speed_controllers, pressure_controller
        self.pressure_node, self.dispatch_policy = pressure_node, dispatch_policy
        for key, value in (
            ("control_interval_s", control_interval_s),
            ("maximum_dt_s", maximum_dt_s),
            ("minimum_dt_s", minimum_dt_s),
            ("max_speed_change_rad_s", max_speed_change_rad_s),
        ):
            setattr(self, key, float(_number(value, key, positive=True)))
        for key, value in (
            ("minimum_stage_dwell_s", minimum_stage_dwell_s),
            ("dispatch_power_deadband_w", dispatch_power_deadband_w),
        ):
            setattr(self, key, float(_number(value, key, nonnegative=True)))
        if minimum_dt_s > maximum_dt_s:
            raise ValueError("Invalid station step bounds")
        for key, value in (
            ("max_station_electrical_draw_w", max_station_electrical_draw_w),
            ("max_station_electrical_export_w", max_station_electrical_export_w),
        ):
            if value is not None:
                _number(value, key, nonnegative=True)
            setattr(self, key, value)
        idle = dict(no_flow_power_w_at_nominal or {})
        if set(idle) - set(self.names):
            raise ValueError("Unknown no-flow power pump")
        self.no_flow_power_w_at_nominal = {
            n: float(_number(idle.get(n, 0.0), "no_flow_power_w", nonnegative=True))
            for n in self.names
        }
        self._trial = LiquidNetwork(
            network.fluid,
            network.nodes,
            network.pipes,
            [
                replace(p, max_shaft_power_w=None, npsh_required_m=None, npsh_curve=None)
                for p in network.pumps
            ],
            network.valves,
        )

    def to_dict(self):
        return {
            "network": self.network.to_dict(),
            "drives": {n: asdict(self.drives[n]) for n in self.names},
            "nominal_speeds_rad_s": self.nominal_speeds_rad_s,
            "speed_controllers": {
                n: None if self.speed_controllers[n] is None else asdict(self.speed_controllers[n])
                for n in self.names
            },
            "pressure_controller": None
            if self.pressure_controller is None
            else asdict(self.pressure_controller),
            "dispatch_policy": None
            if self.dispatch_policy is None
            else asdict(self.dispatch_policy),
            **{
                k: getattr(self, k)
                for k in (
                    "pressure_node",
                    "control_interval_s",
                    "maximum_dt_s",
                    "minimum_dt_s",
                    "max_speed_change_rad_s",
                    "minimum_stage_dwell_s",
                    "dispatch_power_deadband_w",
                    "max_station_electrical_draw_w",
                    "max_station_electrical_export_w",
                    "no_flow_power_w_at_nominal",
                )
            },
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["network"] = LiquidNetwork.from_dict(value["network"])
        value["drives"] = {n: RotatingDrive(**d) for n, d in value["drives"].items()}
        if value.get("speed_controllers") is not None:
            value["speed_controllers"] = {
                n: None if c is None else PIDController(**c)
                for n, c in value["speed_controllers"].items()
            }
        if value.get("pressure_controller") is not None:
            value["pressure_controller"] = PIDController(**value["pressure_controller"])
        if value.get("dispatch_policy") is not None:
            value["dispatch_policy"] = PumpStationControl(**value["dispatch_policy"])
        return cls(**value)

    @property
    def model_id(self):
        return _digest({"model": "coupled_parallel_station_v1", **self.to_dict()})

    @staticmethod
    def _options(options):
        options = dict(options or {})
        if set(options) - {"boundary_pressures_pa", "valve_openings"}:
            raise ValueError("Unknown station network overrides")
        return options

    def _hydraulics(self, speeds, options, trial=False):
        status = {n: float(speeds[i]) > 0 for i, n in enumerate(self.names)}
        ratios = {
            n: float(speeds[i]) / self.nominal_speeds_rad_s[n]
            for i, n in enumerate(self.names)
            if status[n]
        }
        r = (self._trial if trial else self.network).solve(
            pump_status=status, pump_speed_ratios=ratios, **options
        )
        powers = dict(r.pump_shaft_power_w)
        eff = dict(r.pump_efficiencies)
        loads = []
        for i, n in enumerate(self.names):
            extra = (
                self.no_flow_power_w_at_nominal[n] * (speeds[i] / self.nominal_speeds_rad_s[n]) ** 3
            )
            power = powers.get(n, 0.0) + extra
            if power < -1e-10:
                raise EquipmentOperatingError("Station turbine operation unsupported")
            pump = self.network.pumps[i]
            if not trial and pump.max_shaft_power_w is not None and power > pump.max_shaft_power_w:
                raise EquipmentOperatingError(f"{n}: total shaft power limit")
            if extra > 0:
                powers[n] = float(power)
                eff[n] = r.pump_hydraulic_power_w.get(n, 0.0) / power
            loads.append(float(max(0.0, power) / speeds[i]) if speeds[i] > 0 else 0.0)
        scale = max(
            1.0,
            sum(abs(x) for x in r.pump_hydraulic_power_w.values()),
            sum(abs(x) for x in r.pipe_dissipation_w.values()),
            sum(abs(x) for x in r.valve_dissipation_w.values()),
        )
        if abs(r.power_balance_error_w) > 1e-5 + 1e-7 * scale:
            raise ConservationError("Station hydraulic power audit failed")
        return replace(r, pump_shaft_power_w=powers, pump_efficiencies=eff), np.asarray(loads)

    @staticmethod
    def _unit_torque(d, command, peak):
        torque = min(d.max_motor_torque_nm, max(-d.max_braking_torque_nm, command))
        if torque > 0 and d.max_electrical_draw_w == 0:
            return 0.0
        if torque < 0 and d.max_electrical_export_w == 0 and d.regeneration_efficiency > 0:
            return 0.0
        if peak > 0:
            if torque > 0 and d.max_electrical_draw_w is not None:
                torque = min(torque, d.max_electrical_draw_w * d.motor_efficiency / peak)
            if (
                torque < 0
                and d.max_electrical_export_w is not None
                and d.regeneration_efficiency > 0
            ):
                torque = max(
                    torque, -d.max_electrical_export_w / (peak * d.regeneration_efficiency)
                )
        return torque

    def _allocate(self, old, end, commands, loads, dt):
        peaks = np.maximum(old, np.maximum(end, 0.0))
        mid = np.maximum(0.0, 0.5 * (old + end))
        torques = []
        for i, n in enumerate(self.names):
            d = self.drives[n]
            torque = self._unit_torque(d, commands[i], peaks[i])
            if torque > 0:
                torque = min(
                    torque,
                    d.inertia_kg_m2 * (d.max_speed_rad_s - old[i]) / dt
                    + loads[i]
                    + d.viscous_drag_nm_s * mid[i],
                )
            torques.append(torque)
        draw = math.fsum(
            max(0.0, torques[i]) * peaks[i] / self.drives[n].motor_efficiency
            for i, n in enumerate(self.names)
        )
        export = math.fsum(
            max(0.0, -torques[i]) * peaks[i] * self.drives[n].regeneration_efficiency
            for i, n in enumerate(self.names)
        )
        fd = (
            1.0
            if self.max_station_electrical_draw_w is None or draw == 0
            else min(1.0, self.max_station_electrical_draw_w / draw)
        )
        fe = (
            1.0
            if self.max_station_electrical_export_w is None or export == 0
            else min(1.0, self.max_station_electrical_export_w / export)
        )
        if self.max_station_electrical_draw_w == 0:
            fd = 0.0
        if self.max_station_electrical_export_w == 0:
            fe = 0.0
        return (
            np.asarray(
                [
                    t
                    * (
                        fd
                        if t > 0
                        else (fe if self.drives[self.names[i]].regeneration_efficiency > 0 else 1.0)
                    )
                    for i, t in enumerate(torques)
                ]
            ),
            fd,
            fe,
        )

    def _solve(self, old, dt, commands, options):
        _, initial_load = self._hydraulics(old, options, trial=True)
        initial_torque, _, _ = self._allocate(old, old, commands, initial_load, dt)
        moving = [i for i in range(len(old)) if old[i] > 0 or initial_torque[i] > 0]
        j = np.asarray([self.drives[n].inertia_kg_m2 for n in self.names])
        b = np.asarray([self.drives[n].viscous_drag_nm_s for n in self.names])
        max_speed = np.asarray([self.drives[n].max_speed_rad_s for n in self.names])
        scale = np.maximum(1.0, old)
        evaluations = 1
        cache = {}

        def evaluate(x):
            nonlocal evaluations
            end = np.zeros_like(old)
            end[moving] = x * scale[moving]
            mid = np.maximum(0.0, 0.5 * (old + end))
            key = tuple(mid)
            if key not in cache:
                cache[key] = self._hydraulics(mid, options, trial=True)
                evaluations += 1
            hydraulic, load = cache[key]
            torque, fd, fe = self._allocate(old, end, commands, load, dt)
            residual = (j * (end - old) - dt * (torque - load - b * mid)) / (j * scale)
            return residual[moving], end, torque, fd, fe, hydraulic

        if moving:
            lower = -old[moving] / scale[moving]
            upper = max_speed[moving] / scale[moving]
            predictor = old + dt * (initial_torque - initial_load - b * old) / j
            guess = np.clip(predictor[moving] / scale[moving], lower + 1e-12, upper - 1e-12)
            fit = least_squares(
                lambda x: evaluate(x)[0],
                guess,
                bounds=(lower, upper),
                x_scale="jac",
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=120,
            )
            data = evaluate(fit.x)
            if not fit.success or np.max(abs(data[0])) > 1e-9:
                raise HydraulicConvergenceError("Coupled station shaft solve failed")
        else:
            data = evaluate(np.asarray([]))
        return data[1], data[2], data[3], data[4], evaluations

    def _step(self, old, requested_dt, commands, options):
        dt = requested_dt
        end, torque, fd, fe, count = self._solve(old, dt, commands, options)
        candidates = [i for i in range(len(old)) if end[i] < -1e-8]
        if candidates:
            if any(torque[i] >= 0 for i in candidates):
                raise HydraulicConvergenceError("Station step overshoots nonnegative equilibrium")
            calls = [count]

            def event(h):
                if h == 0:
                    return min(old[i] for i in candidates)
                value = self._solve(old, h, commands, options)
                calls[0] += value[4]
                return min(value[0][i] for i in candidates)

            dt = brentq(event, 0.0, dt, xtol=1e-11, rtol=1e-11)
            end, torque, fd, fe, extra = self._solve(old, dt, commands, options)
            count = calls[0] + extra
        # Only a located stopping event may truncate a small positive speed.
        # Otherwise weak acceleration would acquire an artificial dead band.
        if candidates:
            for i in candidates:
                if abs(end[i]) < 1e-8:
                    end[i] = 0.0
        end = np.where((end < 0) & (end > -1e-10), 0.0, end)
        if np.any(end < 0):
            raise HydraulicConvergenceError("Station event produced negative speed")
        if np.max(abs(end - old)) > self.max_speed_change_rad_s * (1 + 1e-12):
            raise HydraulicConvergenceError("Station shaft accuracy guard")
        mid = 0.5 * (old + end)
        hydraulic, load = self._hydraulics(mid, options)
        endpoint, _ = self._hydraulics(end, options)
        count += 2
        actual, fd, fe = self._allocate(old, end, commands, load, dt)
        if np.max(abs(actual - torque)) > 1e-6:
            raise ConservationError("Station torque allocation changed at accepted state")
        unit = {}
        grid = []
        heat = []
        kinetic = []
        pump_total = 0.0
        hydraulic_total = 0.0
        for i, n in enumerate(self.names):
            d = self.drives[n]
            work = dt * torque[i] * mid[i]
            pump = dt * hydraulic.pump_shaft_power_w.get(n, 0.0)
            hw = dt * hydraulic.pump_hydraulic_power_w.get(n, 0.0)
            drag = dt * d.viscous_drag_nm_s * mid[i] ** 2
            dk = 0.5 * d.inertia_kg_m2 * (end[i] - old[i]) * (end[i] + old[i])
            electric = work / d.motor_efficiency if work >= 0 else work * d.regeneration_efficiency
            motor_heat = electric - work
            pump_heat = pump - hw
            error = work - pump - drag - dk
            scale = max(1.0, abs(work) + pump + drag + abs(dk))
            if (
                not all(math.isfinite(x) for x in (error, scale, electric))
                or abs(error) > 1e-7 + 1e-8 * scale
                or min(motor_heat, pump_heat, drag) < -1e-8
            ):
                raise ConservationError("Individual station shaft energy balance failed")
            unit[n] = {
                "initial_speed_rad_s": float(old[i]),
                "final_speed_rad_s": float(end[i]),
                "motor_torque_nm": float(torque[i]),
                "command_torque_nm": float(commands[i]),
                "pump_load_torque_nm": float(load[i]),
                "motor_work_j": float(work),
                "pump_work_j": float(pump),
                "hydraulic_work_j": float(hw),
                "drag_heat_j": float(drag),
                "kinetic_energy_change_j": float(dk),
                "electrical_energy_j": float(electric),
                "motor_heat_j": float(motor_heat),
                "pump_heat_j": float(pump_heat),
                "shaft_energy_residual_j": float(error),
                "rest_constraint_impulse_nm_s": float(
                    -min(0.0, torque[i]) * dt if old[i] == 0 and end[i] == 0 else 0.0
                ),
            }
            grid.append(electric)
            heat.append(motor_heat + pump_heat + drag)
            kinetic.append(dk)
            pump_total += pump
            hydraulic_total += hw
        passive = dt * (
            sum(hydraulic.pipe_dissipation_w.values()) + sum(hydraulic.valve_dissipation_w.values())
        )
        rho = self.network.fluid.density_kg_m3
        boundary = -dt * sum(
            (hydraulic.pressures_pa[n.name] + rho * 9.80665 * n.elevation_m)
            * hydraulic.boundary_mass_rates_kg_s.get(n.name, 0.0)
            / rho
            for n in self.network.nodes
        )
        error = math.fsum(grid) - math.fsum(heat) - math.fsum(kinetic) - passive - boundary
        scale = max(1.0, sum(abs(x) for x in grid + heat + kinetic) + abs(passive) + abs(boundary))
        if not math.isfinite(error) or abs(error) > 1e-6 + 2e-7 * scale:
            raise ConservationError("Whole station energy balance failed")
        peakdraw = math.fsum(
            max(0.0, torque[i]) * max(old[i], end[i]) / self.drives[n].motor_efficiency
            for i, n in enumerate(self.names)
        )
        peakexport = math.fsum(
            max(0.0, -torque[i]) * max(old[i], end[i]) * self.drives[n].regeneration_efficiency
            for i, n in enumerate(self.names)
        )
        if (
            self.max_station_electrical_draw_w is not None
            and peakdraw > self.max_station_electrical_draw_w * (1 + 1e-10)
        ) or (
            self.max_station_electrical_export_w is not None
            and peakexport > self.max_station_electrical_export_w * (1 + 1e-10)
        ):
            raise ConservationError("Station electrical capacity violated")
        return end, {
            "dt_s": float(dt),
            "units": unit,
            "peak_electrical_draw_w": float(peakdraw),
            "peak_electrical_export_w": float(peakexport),
            "draw_allocation_factor": float(fd),
            "export_allocation_factor": float(fe),
            "electrical_energy_j": float(math.fsum(grid)),
            "kinetic_energy_change_j": float(math.fsum(kinetic)),
            "unit_heat_j": float(math.fsum(heat)),
            "passive_heat_j": float(passive),
            "boundary_work_out_j": float(boundary),
            "energy_residual_j": float(error),
            "network_evaluations": count,
            "midpoint": asdict(hydraulic),
            "boundary_mass_transfers_kg": {
                n: float(dt * m) for n, m in hydraulic.boundary_mass_rates_kg_s.items()
            },
            "midpoint_pump_flows_m3_s": hydraulic.pump_volume_rates_m3_s,
            "midpoint_pressures_pa": hydraulic.pressures_pa,
            "endpoint": asdict(endpoint),
        }

    def _check(self, state):
        if not isinstance(state, PumpStationState) or state.model_id != self.model_id:
            raise ValueError("Dynamic station state/model mismatch")
        speeds = dict(state.speeds_rad_s)
        pids = dict(state.speed_pid_states)
        torques = dict(state.applied_torques_nm)
        if (
            set(speeds) != set(self.names)
            or set(pids) != set(self.names)
            or set(torques) != set(self.names)
        ):
            raise ValueError("Station state unit mismatch")
        for n in self.names:
            self.drives[n].initialize(speeds[n], state.time_s)
            c = self.speed_controllers[n]
            s = pids[n]
            if (c is None) != (s is None) or (
                s is not None and (not isinstance(s, PIDState) or s.model_id != c.model_id)
            ):
                raise ValueError("Station speed PID identity mismatch")
        if (self.pressure_controller is None) != (state.pressure_pid_state is None) or (
            state.pressure_pid_state is not None
            and state.pressure_pid_state.model_id != self.pressure_controller.model_id
        ):
            raise ValueError("Station pressure PID identity mismatch")
        if (set(state.target_running) | set(state.unavailable)) - set(
            self.names
        ) or state.last_switch_time_s > state.time_s:
            raise ValueError("Invalid station availability/staging memory")

    def initialize(self, speeds_rad_s=None, time_s=0.0, unavailable=(), network_options=None):
        values = dict(speeds_rad_s or {})
        if set(values) - set(self.names) or set(unavailable) - set(self.names):
            raise ValueError("Unknown station initial unit")
        speeds = {
            n: float(_number(values.get(n, 0.0), "speed_rad_s", nonnegative=True))
            for n in self.names
        }
        for n in self.names:
            self.drives[n].initialize(speeds[n], time_s)
        r, _ = self._hydraulics(
            np.asarray([speeds[n] for n in self.names]), self._options(network_options)
        )
        pids = {
            n: None
            if self.speed_controllers[n] is None
            else self.speed_controllers[n].initialize(speeds[n], mode="off")
            for n in self.names
        }
        pressure = (
            None
            if self.pressure_controller is None
            else self.pressure_controller.initialize(r.pressures_pa[self.pressure_node], mode="off")
        )
        state = PumpStationState(
            speeds,
            pids,
            pressure,
            {n: 0.0 for n in self.names},
            (),
            tuple(unavailable),
            time_s,
            time_s,
            "",
            "not_dispatched",
            self.model_id,
        )
        self._check(state)
        return state

    def _dispatch(self, setpoint, unavailable, previous, options):
        available = tuple(n for n in self.names if n not in unavailable)
        if not available:
            return (), 0.0, {"status": "no_available_units", "candidates": []}
        if self.dispatch_policy is None:
            selected = tuple(n for n in previous if n in available) or available
            return selected, 0.0, {"status": "explicit_units", "candidates": []}
        policy = replace(self.dispatch_policy, target_pressure_pa=setpoint)
        try:
            plan = policy.solve(self.network, unavailable=unavailable, **options)
            candidates = [dict(row) for row in plan.candidates]
        except PumpStationOperatingError as error:
            candidates = [dict(row) for row in error.candidates]
        feasible = []
        for row in candidates:
            if row["status"] != "feasible":
                continue
            names = tuple(row["pumps"])
            ratio = row["speed_ratio"]
            speeds = np.asarray(
                [ratio * self.nominal_speeds_rad_s[n] if n in names else 0.0 for n in self.names]
            )
            try:
                _hydraulic, loads = self._hydraulics(speeds, options)
                electrical = 0.0
                for i, n in enumerate(self.names):
                    if n not in names:
                        continue
                    d = self.drives[n]
                    d.initialize(speeds[i])
                    torque = loads[i] + d.viscous_drag_nm_s * speeds[i]
                    power = torque * speeds[i] / d.motor_efficiency
                    if torque > d.max_motor_torque_nm or (
                        d.max_electrical_draw_w is not None and power > d.max_electrical_draw_w
                    ):
                        raise EquipmentOperatingError(f"{n}: steady drive torque/power limit")
                    electrical += power
                if (
                    self.max_station_electrical_draw_w is not None
                    and electrical > self.max_station_electrical_draw_w
                ):
                    raise EquipmentOperatingError("Station steady gross electrical limit")
            except EquipmentOperatingError as error:
                row["status"] = "drive_or_power_infeasible"
                row["reason"] = str(error)
                continue
            row["electrical_power_w"] = float(electrical)
            feasible.append((electrical, names, ratio))
        if feasible:
            feasible.sort(key=lambda v: (v[0], len(v[1]), v[1]))
            chosen = feasible[0]
            status = "minimum_sampled_electrical_power"
            retained = next((v for v in feasible if set(v[1]) == set(previous)), None)
            if retained is not None and retained[0] <= chosen[0] + self.dispatch_power_deadband_w:
                chosen = retained
                status = "retained_by_electrical_deadband"
            return (
                chosen[1],
                chosen[2],
                {
                    "status": status,
                    "candidates": candidates,
                    "steady_electrical_power_w": float(chosen[0]),
                },
            )
        # Infeasible requested pressure is a degraded operating condition, not
        # permission to enlarge maps or invent drive power. Physics still runs.
        high = policy.max_running
        fallback = tuple(n for n in previous if n in available)[:high]
        if len(fallback) < policy.min_running:
            fallback = available[:high]
        return fallback, 0.0, {"status": "target_infeasible", "candidates": candidates}

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
        network_options=None,
        reset_trip=False,
        redispatch=False,
        max_substeps=100000,
    ):
        self._check(state)
        duration = float(_number(duration_s, "duration_s", positive=True))
        manual_ratio = float(_number(speed_ratio, "speed_ratio", nonnegative=True))
        setpoint = (
            self.dispatch_policy.target_pressure_pa
            if pressure_setpoint_pa is None and self.dispatch_policy is not None
            else pressure_setpoint_pa
        )
        if (
            mode not in ("auto", "manual", "torque", "off", "trip")
            or not isinstance(reset_trip, bool)
            or not isinstance(redispatch, bool)
            or (mode == "trip" and reset_trip)
        ):
            raise ValueError("Invalid station mode/reset")
        if mode == "auto" and (self.pressure_controller is None or setpoint is None):
            raise ValueError("Automatic station requires a pressure PID and setpoint")
        if setpoint is not None:
            setpoint = float(_number(setpoint, "pressure_setpoint_pa", positive=True))
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("max_substeps must be positive integer")
        if not math.isfinite(state.time_s + duration) or state.time_s + duration <= state.time_s:
            raise ValueError("Station interval cannot advance time")
        unavailable = state.unavailable if unavailable is None else tuple(unavailable)
        requested = state.target_running if running_pumps is None else tuple(running_pumps)
        commands_input = dict(motor_torques_nm or {})
        if (
            (set(unavailable) | set(requested) | set(commands_input)) - set(self.names)
            or len(unavailable) != len(set(unavailable))
            or len(requested) != len(set(requested))
        ):
            raise ValueError("Unknown/duplicate station unit command")
        for value in commands_input.values():
            _number(value, "motor_torque_nm")
        if mode == "torque" and running_pumps is None:
            requested = tuple(commands_input)
        options = self._options(network_options)
        tripped = (state.tripped and not reset_trip) or mode == "trip"
        effective = "off" if tripped else mode
        if effective == "auto":
            node = self.network.nodes[self.network._index[self.pressure_node]]
            if (options.get("boundary_pressures_pa") or {}).get(
                node.name, node.pressure_pa
            ) is not None:
                raise ValueError("Cannot regulate a fixed pressure boundary")
        speeds = np.asarray([dict(state.speeds_rad_s)[n] for n in self.names])
        pids = dict(state.speed_pid_states)
        pressure_pid = state.pressure_pid_state
        applied = dict(state.applied_torques_nm)
        selected = state.target_running
        last_switch = state.last_switch_time_s
        signature = state.dispatch_signature
        status = state.dispatch_status
        elapsed = 0.0
        steps = []
        samples = []
        dispatches = []
        failures = 0
        failure_reasons = {}
        while elapsed < duration:
            end_sample = min(duration, (len(samples) + 1) * self.control_interval_s)
            sample_dt = end_sample - elapsed
            now = state.time_s + elapsed
            hydraulic, _ = self._hydraulics(speeds, options)
            measured = hydraulic.pressures_pa[self.pressure_node]
            selected_before = selected
            feedforward = 0.0
            if effective == "auto":
                wanted = _digest(
                    {"setpoint": setpoint, "unavailable": unavailable, "options": options}
                )
                forced = bool(set(selected) & set(unavailable)) or not selected
                due = (signature != wanted or (redispatch and not samples)) and (
                    forced or now - last_switch >= self.minimum_stage_dwell_s
                )
                if due:
                    selected, feedforward, dispatch = self._dispatch(
                        setpoint, unavailable, selected, options
                    )
                    signature = wanted
                    status = dispatch["status"]
                    dispatches.append(
                        {
                            "time_s": now,
                            **dispatch,
                            "selected": list(selected),
                            "feedforward_speed_ratio": feedforward,
                        }
                    )
            else:
                selected = (
                    ()
                    if effective == "off"
                    else tuple(n for n in requested if n not in unavailable)
                )
                status = effective
                # An automatic policy must re-evaluate after explicit manual/off.
                signature = ""
            if set(selected) != set(selected_before):
                last_switch = now
            ratio_actual = (
                math.fsum(
                    speeds[self.names.index(n)] / self.nominal_speeds_rad_s[n] for n in selected
                )
                / len(selected)
                if selected
                else 0.0
            )
            outer_initialized = False
            if (
                effective == "auto"
                and selected
                and self.pressure_controller is not None
                and not selected_before
                and feedforward > 0
            ):
                seed = min(self.pressure_controller.max_output, feedforward)
                pressure_pid = self.pressure_controller.initialize(
                    measured, seed, setpoint, mode="auto"
                )
                outer_initialized = True
            if self.pressure_controller is not None:
                outer = self.pressure_controller
                tracking = min(outer.max_output, max(outer.min_output, ratio_actual))
                outer_mode = (
                    "auto"
                    if effective == "auto" and selected
                    else ("manual" if effective in ("manual", "torque") and selected else "off")
                )
                response = outer.step(
                    pressure_pid,
                    setpoint if setpoint is not None else measured,
                    measured,
                    sample_dt,
                    outer_mode,
                    min(outer.max_output, manual_ratio),
                    tracking,
                )
                pressure_pid = PIDState(**response["state"])
                common_ratio = response["output"] if effective == "auto" else manual_ratio
            else:
                response = None
                common_ratio = manual_ratio
            unit_commands = {}
            unit_responses = {}
            for i, n in enumerate(self.names):
                enabled = n in selected and n not in unavailable and effective != "off"
                controller = self.speed_controllers[n]
                if effective in ("auto", "manual") and enabled and controller is None:
                    raise ValueError("Speed control requires local PID for every selected unit")
                local_mode = (
                    ("auto" if effective in ("auto", "manual") else "manual") if enabled else "off"
                )
                if controller is not None:
                    r = controller.step(
                        pids[n],
                        common_ratio * self.nominal_speeds_rad_s[n],
                        float(speeds[i]),
                        sample_dt,
                        local_mode,
                        commands_input.get(n, 0.0),
                        applied[n],
                    )
                    pids[n] = PIDState(**r["state"])
                    unit_commands[n] = r["output"]
                    unit_responses[n] = r
                else:
                    unit_commands[n] = commands_input.get(n, 0.0) if enabled else 0.0
                    unit_responses[n] = None
            sample = {
                "time_s": now,
                "dt_s": sample_dt,
                "measured_pressure_pa": measured,
                "measured_speeds_rad_s": dict(zip(self.names, map(float, speeds))),
                "selected": list(selected),
                "unavailable": list(unavailable),
                "dispatch_status": status,
                "feedforward_speed_ratio": feedforward,
                "outer_initialized": outer_initialized,
                "actual_speed_ratio": ratio_actual,
                "common_speed_ratio": common_ratio,
                "pressure_pid": response,
                "speed_pid": unit_responses,
                "motor_commands_nm": unit_commands,
            }
            samples.append(sample)
            within = 0.0
            weighted = {n: [] for n in self.names}
            step_dt = min(self.maximum_dt_s, sample_dt)
            while within < sample_dt:
                if len(steps) + failures >= max_substeps:
                    raise HydraulicConvergenceError("Station substep budget exhausted")
                remaining = sample_dt - within
                roundoff = 8 * max(math.ulp(end_sample), math.ulp(elapsed), math.ulp(step_dt))
                dt = remaining if remaining <= step_dt + roundoff else step_dt
                try:
                    end, record = self._step(
                        speeds, dt, np.asarray([unit_commands[n] for n in self.names]), options
                    )
                except HydraulicConvergenceError as error:
                    failures += 1
                    reason = str(error)
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                    if dt / 2 < self.minimum_dt_s:
                        raise
                    step_dt = dt / 2
                    continue
                accepted = record["dt_s"]
                following = math.fsum([within, accepted])
                if (
                    following <= within
                    or state.time_s + elapsed + following <= state.time_s + elapsed + within
                ):
                    raise ValueError("Station substep cannot advance time")
                if following > sample_dt + 1e-12:
                    raise ConservationError("Station event advanced beyond control sample")
                steps.append({"time_s": state.time_s + elapsed + within, **record})
                speeds = end
                for n in self.names:
                    weighted[n].append(record["units"][n]["motor_torque_nm"] * accepted)
                within = min(sample_dt, following)
                step_dt = min(self.maximum_dt_s, max(self.minimum_dt_s, accepted * 1.5))
            for n in self.names:
                c = self.speed_controllers[n]
                lo = -self.drives[n].max_braking_torque_nm if c is None else c.min_output
                hi = self.drives[n].max_motor_torque_nm if c is None else c.max_output
                applied[n] = min(hi, max(lo, math.fsum(weighted[n]) / sample_dt))
            sample["applied_torques_nm"] = dict(applied)
            elapsed = end_sample
        final = PumpStationState(
            dict(zip(self.names, map(float, speeds))),
            pids,
            pressure_pid,
            applied,
            selected,
            unavailable,
            state.time_s + duration,
            last_switch,
            signature,
            status,
            self.model_id,
            tripped,
        )
        self._check(final)
        endpoint, _ = self._hydraulics(speeds, options)
        return {
            "state": final.to_dict(),
            "steps": steps,
            "control_samples": samples,
            "dispatches": dispatches,
            "failed_steps": failures,
            "retry_reasons": failure_reasons,
            "requested_mode": mode,
            "effective_mode": effective,
            "endpoint": asdict(endpoint),
            "pressure_error_pa": None
            if effective != "auto"
            else setpoint - endpoint.pressures_pa[self.pressure_node],
            "target_pressure_pa": setpoint,
            "duration_s": duration,
        }
