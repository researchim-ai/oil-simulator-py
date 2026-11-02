"""Period-boundary restart and physical/controller audits for compressor trains."""

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import scipy

from .compressor_train import CompressorTrainState
from .liquid import ConservationError, _number
from .operation import _digest
from .pid import PIDState


def run_compressor_schedule(system, initial, periods, checkpoint=None, stop_after=None):
    system._check(initial)
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Compressor schedule requires periods")
    periods = copy.deepcopy(periods)
    allowed = {
        "duration_s",
        "mode",
        "speed_control_mode",
        "speed_setpoint_rad_s",
        "motor_torque_nm",
        "export_opening",
        "anti_surge_mode",
        "manual_recycle_opening",
        "recycle_actuator_mode",
        "reset_trip",
        "max_substeps",
    }
    for period in periods:
        if not isinstance(period, dict) or "duration_s" not in period or set(period) - allowed:
            raise ValueError("Invalid compressor period fields")
        _number(period["duration_s"], "duration_s", positive=True)
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0
    ):
        raise ValueError("stop_after must be a nonnegative integer")
    root = Path(__file__).resolve().parent
    names = (
        "compressor_schedule.py",
        "compressor_train.py",
        "compressor_map.py",
        "gas.py",
        "gas_valve.py",
        "pid.py",
        "actuator.py",
        "rotating_drive.py",
        "liquid.py",
        "operation.py",
    )
    fingerprint = _digest(
        {
            "system": system.to_dict(),
            "initial": initial.to_dict(),
            "periods": periods,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "source": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
            },
        }
    )
    state = {
        "fingerprint": fingerprint,
        "next_period": 0,
        "state": initial.to_dict(),
        "records": [],
    }
    if checkpoint is not None:
        saved = copy.deepcopy(checkpoint)
        if set(saved) != set(state) | {"checksum"}:
            raise ValueError("Invalid compressor checkpoint fields")
        checksum = saved.pop("checksum")
        if checksum != _digest(saved) or saved["fingerprint"] != fingerprint:
            raise ValueError("Compressor checksum/fingerprint mismatch")
        index = saved["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(saved["records"]) != index
        ):
            raise ValueError("Compressor checkpoint index mismatch")
        state = saved

    def audit():
        current = initial
        for index, record in enumerate(state["records"]):
            period = periods[index]
            if record["period"] != index or record["initial_state"] != current.to_dict():
                raise ConservationError("Compressor checkpoint history mismatch")
            tripped = (current.tripped and not period.get("reset_trip", False)) or period.get(
                "mode", "run"
            ) == "trip"
            effective = "off" if tripped else period.get("mode", "run")
            running = effective == "run"
            speed_pid, anti_pid = current.speed_pid, current.anti_surge_pid
            applied = current.applied_motor_torque_nm
            samples = record["control_samples"]
            if not samples or not record["steps"]:
                raise ValueError("Empty compressor checkpoint period")
            sample_clock = 0.0
            for sample in samples:
                dt = _number(sample["dt_s"], "control_dt_s", positive=True)
                if not math.isclose(
                    sample["time_s"], current.time_s + sample_clock, abs_tol=1e-9, rel_tol=0
                ):
                    raise ConservationError("Compressor controller clock mismatch")
                if system.speed_controller is not None:
                    mode = (
                        period.get("speed_control_mode", "auto")
                        if running
                        else ("manual" if effective == "standby" else "off")
                    )
                    replay = system.speed_controller.step(
                        speed_pid,
                        period.get("speed_setpoint_rad_s", 0.0),
                        sample["measured_speed_rad_s"],
                        dt,
                        mode,
                        period.get("motor_torque_nm", 0.0),
                        applied,
                    )
                    if replay != sample["speed"]:
                        raise ConservationError("Compressor speed PID replay mismatch")
                    speed_pid = PIDState(**replay["state"])
                mode = period.get("anti_surge_mode", "auto") if running else "manual"
                replay = system.anti_surge_controller.step(
                    anti_pid,
                    system.target_surge_margin,
                    sample["measured_surge_margin"] if running else 0.0,
                    dt,
                    mode,
                    period.get("manual_recycle_opening", 1.0) if running else 1.0,
                    sample["measured_recycle_position"],
                )
                if replay != sample["anti_surge"]:
                    raise ConservationError("Compressor anti-surge PID replay mismatch")
                anti_pid = PIDState(**replay["state"])
                override = (
                    running
                    and mode == "auto"
                    and sample["measured_surge_margin"] < system.fast_open_margin
                )
                expected_target = 1.0 if override or not running else replay["output"]
                expected_command = (
                    sample["speed"]["output"]
                    if system.speed_controller is not None
                    else (
                        period.get("motor_torque_nm", 0.0)
                        if effective in ("run", "standby")
                        else 0.0
                    )
                )
                if (
                    sample["recycle_target"] != expected_target
                    or sample["fast_open_override"] != override
                    or sample["motor_command_nm"] != expected_command
                ):
                    raise ConservationError("Compressor control selector audit failed")
                applied = sample["applied_motor_torque_nm"]
                sample_clock = math.fsum([sample_clock, dt])
            mass, energy, speed, position = (
                current.mass_kg,
                current.internal_energy_j,
                current.angular_velocity_rad_s,
                current.recycle_position,
            )
            clock = 0.0
            sample_index = 0
            total_mass = []
            total_energy = []
            for row in record["steps"]:
                dt = _number(row["dt_s"], "dt_s", positive=True)
                if not math.isclose(row["time_s"], current.time_s + clock, abs_tol=1e-9, rel_tol=0):
                    raise ConservationError("Compressor checkpoint step clock mismatch")
                while (
                    sample_index + 1 < len(samples)
                    and row["time_s"] >= samples[sample_index + 1]["time_s"] - 1e-10
                ):
                    sample_index += 1
                sample = samples[sample_index]
                if row["time_s"] + dt > sample["time_s"] + sample["dt_s"] + 1e-9:
                    raise ConservationError("Compressor substep crosses a controller sample")
                expected_position = system.recycle_actuator.step(
                    position, sample["recycle_target"], dt, sample["actuator_mode"]
                )["final_position"]
                m = _number(row["mass_kg"], "mass_kg", positive=True)
                u = _number(row["internal_energy_j"], "internal_energy_j", positive=True)
                w = _number(row["final_speed_rad_s"], "speed", nonnegative=True)
                if (
                    row["initial_speed_rad_s"] != speed
                    or abs(row["recycle_position"] - expected_position) > 1e-12
                ):
                    raise ConservationError("Compressor actuator/speed path mismatch")
                flows = system._flows(
                    m,
                    u,
                    0.5 * (speed + w),
                    expected_position,
                    period.get("export_opening", 1.0),
                    running,
                    True,
                )
                comp = flows["compressor"]
                if running:
                    system._flows(
                        m, u, w, expected_position, period.get("export_opening", 1.0), True, True
                    )
                gas = system.gas
                fresh = dt * flows["net_suction_mass_rate_kg_s"]
                exported = dt * flows["export_mass_rate_kg_s"]
                recycle = dt * flows["recycle_mass_rate_kg_s"]
                p, t = system.properties(m, u)
                shaft = dt * (0.0 if comp is None else comp["shaft_power_w"])
                gas_work = dt * (0.0 if comp is None else comp["gas_power_w"])
                motor = row["motor_torque_nm"]
                cap = system._motor(sample["motor_command_nm"], max(speed, w))
                if not min(0.0, cap) - 1e-8 <= motor <= max(0.0, cap) + 1e-8:
                    raise ConservationError("Compressor motor limit audit failed")
                if running:
                    work = dt * motor * 0.5 * (speed + w)
                    drag = dt * system.drive.viscous_drag_nm_s * (0.5 * (speed + w)) ** 2
                else:
                    exact = system.drive.advance(system.drive.initialize(speed), dt, motor)
                    if abs(exact["state"]["angular_velocity_rad_s"] - w) > 1e-9:
                        raise ConservationError("Unloaded shaft trajectory audit failed")
                    work, drag = exact["motor_work_j"], exact["drag_heat_j"]
                electric = (
                    work / system.drive.motor_efficiency
                    if work >= 0
                    else work * system.drive.regeneration_efficiency
                )
                kinetic = 0.5 * system.drive.inertia_kg_m2 * (w - speed) * (w + speed)
                mechanical = shaft - gas_work
                motor_heat = electric - work
                wall = dt * flows["wall_heat_w"]
                conditioning = dt * flows["conditioning_heat_w"]
                inlet = fresh * gas.cp_j_kg_k * system.suction_temperature_k
                outlet = exported * gas.cp_j_kg_k * t
                net = (
                    electric + inlet - outlet - motor_heat - drag - mechanical - wall - conditioning
                )
                scale = max(
                    1.0, abs(electric) + abs(inlet) + abs(outlet) + abs(conditioning) + abs(wall)
                )
                mass_scale = 1e-10 + 1e-8 * (abs(fresh) + abs(exported) + abs(recycle))
                if any(
                    abs(a - b) > mass_scale
                    for a, b in (
                        (row["net_suction_mass_kg"], fresh),
                        (row["export_mass_kg"], exported),
                        (row["recycle_mass_kg"], recycle),
                    )
                ):
                    raise ConservationError("Compressor recorded mass transfer audit failed")
                pairs = [
                    (row["motor_work_j"], work),
                    (row["shaft_work_j"], shaft),
                    (row["gas_work_j"], gas_work),
                    (row["electrical_energy_j"], electric),
                    (row["mechanical_heat_j"], mechanical),
                    (row["motor_heat_j"], motor_heat),
                    (row["drag_heat_j"], drag),
                    (row["wall_heat_j"], wall),
                    (row["conditioning_heat_j"], conditioning),
                    (row["suction_enthalpy_j"], inlet),
                    (row["export_enthalpy_j"], outlet),
                    (row["gas_energy_change_j"], u - energy),
                    (row["kinetic_energy_change_j"], kinetic),
                    (net, u - energy + kinetic),
                    (work - shaft - drag, kinetic),
                ]
                if any(not math.isfinite(a) or abs(a - b) > 1e-5 + 2e-8 * scale for a, b in pairs):
                    raise ConservationError("Compressor work/energy audit failed")
                if abs(m - mass - fresh + exported) > 1e-10 + 1e-8 * (abs(fresh) + abs(exported)):
                    raise ConservationError("Compressor mass audit failed")
                if (
                    not system.min_temperature_k <= t <= system.max_temperature_k
                    or p > system.max_pressure_pa
                ):
                    raise ConservationError("Compressor gas state audit failed")
                sin = gas.cp_j_kg_k * math.log(
                    system.suction_temperature_k / 300.0
                ) - gas.specific_gas_constant_j_kg_k * math.log(system.suction_pressure_pa / 1e5)
                sout = gas.cp_j_kg_k * math.log(
                    t / 300.0
                ) - gas.specific_gas_constant_j_kg_k * math.log(
                    system.export_back_pressure_pa / 1e5
                )
                entropy = (
                    system._entropy(m, u)
                    - system._entropy(mass, energy)
                    - fresh * sin
                    + exported * sout
                    + (wall + motor_heat + drag + mechanical) / system.ambient_temperature_k
                    + conditioning / system.suction_temperature_k
                )
                entropy_scale = max(
                    1.0,
                    abs(fresh * sin)
                    + abs(exported * sout)
                    + abs(conditioning / system.suction_temperature_k),
                )
                if (
                    not math.isfinite(entropy)
                    or entropy < -1e-7 - 1e-8 * entropy_scale
                    or abs(row["entropy_production_j_k"] - entropy) > 1e-7 + 1e-8 * entropy_scale
                ):
                    raise ConservationError("Compressor entropy audit failed")
                total_mass.append(fresh - exported)
                total_energy.append(net)
                mass, energy, speed, position = m, u, w, expected_position
                clock = math.fsum([clock, dt])
            end = CompressorTrainState.from_dict(record["state"])
            system._check(end)
            expected = CompressorTrainState(
                mass,
                energy,
                speed,
                current.time_s + period["duration_s"],
                position,
                speed_pid,
                anti_pid,
                applied,
                system.model_id,
                tripped,
            )
            if (
                end.to_dict() != expected.to_dict()
                or not math.isclose(clock, period["duration_s"], abs_tol=1e-10, rel_tol=1e-12)
                or not math.isclose(sample_clock, clock, abs_tol=1e-10, rel_tol=1e-12)
            ):
                raise ConservationError("Compressor checkpoint state/controller memory mismatch")
            kinetic = (
                0.5
                * system.drive.inertia_kg_m2
                * (speed - current.angular_velocity_rad_s)
                * (speed + current.angular_velocity_rad_s)
            )
            if abs(math.fsum(total_mass) - (mass - current.mass_kg)) > 1e-8 or abs(
                math.fsum(total_energy) - (energy - current.internal_energy_j + kinetic)
            ) > 1e-4 + 2e-8 * sum(abs(v) for v in total_energy):
                raise ConservationError("Compressor accumulated inventory/energy audit failed")
            current = end
        if current.to_dict() != state["state"]:
            raise ConservationError("Compressor saved inventory mismatch")
        return current

    current = audit()
    start = state["next_period"]
    end = len(periods) if stop_after is None else min(len(periods), start + stop_after)
    for index in range(start, end):
        result = system.advance(current, **periods[index])
        state["records"].append({"period": index, "initial_state": current.to_dict(), **result})
        state["state"] = result["state"]
        state["next_period"] = index + 1
        current = audit()
    return {**state, "checksum": _digest(state)}
