"""Audited period-boundary restart for coupled pump/shaft/PID dynamics."""

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import scipy

from .liquid import ConservationError, _number
from .operation import _digest
from .pid import PIDState
from .pump_drive import PumpDriveState


def run_pump_drive_schedule(system, initial, periods, checkpoint=None, stop_after=None):
    if initial.model_id != system.model_id:
        raise ValueError("Pump-drive initial state/model mismatch")
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Pump-drive schedule requires periods")
    periods = copy.deepcopy(periods)
    for period in periods:
        if (
            not isinstance(period, dict)
            or "duration_s" not in period
            or set(period)
            - {
                "duration_s",
                "mode",
                "setpoint",
                "motor_torque_nm",
                "network_options",
                "reset_trip",
                "max_substeps",
            }
        ):
            raise ValueError("Invalid pump-drive period fields")
        _number(period["duration_s"], "duration_s", positive=True)
        system._options(period.get("network_options"))
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0
    ):
        raise ValueError("stop_after must be a nonnegative integer")
    root = Path(__file__).resolve().parent
    fingerprint = _digest(
        {
            "system": system.to_dict(),
            "initial": initial.to_dict(),
            "periods": periods,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "source": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    "pump_drive.py",
                    "pid.py",
                    "pump_drive_schedule.py",
                    "rotating_drive.py",
                    "liquid.py",
                    "curves.py",
                    "operation.py",
                )
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
            raise ValueError("Invalid pump-drive checkpoint fields")
        checksum = saved.pop("checksum")
        if checksum != _digest(saved) or saved["fingerprint"] != fingerprint:
            raise ValueError("Pump-drive checksum/fingerprint mismatch")
        index = saved["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(saved["records"]) != index
        ):
            raise ValueError("Pump-drive checkpoint index mismatch")
        state = saved

    def audit():
        current = initial
        for index, record in enumerate(state["records"]):
            if record["period"] != index or record["initial_state"] != current.to_dict():
                raise ConservationError("Pump-drive checkpoint history mismatch")
            period = periods[index]
            duration = period["duration_s"]
            if not record["steps"] or not record["control_samples"]:
                raise ValueError("Empty pump-drive period history")
            speed, elapsed = current.angular_velocity_rad_s, 0.0
            electrical, losses, boundary = [], [], []
            for step in record["steps"]:
                dt = _number(step["dt_s"], "dt_s", positive=True)
                moving = _number(step["moving_duration_s"], "moving_duration_s", nonnegative=True)
                end = _number(step["final_speed_rad_s"], "final_speed_rad_s", nonnegative=True)
                if (
                    moving > dt
                    or end > system.drive.max_speed_rad_s
                    or step["initial_speed_rad_s"] != speed
                ):
                    raise ConservationError("Pump-drive checkpoint speed path mismatch")
                if not math.isclose(
                    step["time_s"], current.time_s + elapsed, abs_tol=1e-9, rel_tol=0
                ):
                    raise ConservationError("Pump-drive checkpoint clock mismatch")
                motor = _number(step["motor_torque_nm"], "motor_torque_nm")
                if (
                    not -system.drive.max_braking_torque_nm
                    <= motor
                    <= system.drive.max_motor_torque_nm
                ):
                    raise ConservationError("Pump-drive checkpoint torque limit mismatch")
                cap = system._motor(step["command_torque_nm"], max(speed, end))
                if not min(0.0, cap) - 1e-9 <= motor <= max(0.0, cap) + 1e-9:
                    raise ConservationError("Pump-drive checkpoint actuator cap mismatch")
                midpoint = 0.5 * (speed + end)
                kinetic = 0.5 * system.drive.inertia_kg_m2 * (end - speed) * (end + speed)
                work = motor * midpoint * moving
                pump_work = _number(step["pump_work_j"], "pump_work_j", nonnegative=True)
                drag = system.drive.viscous_drag_nm_s * midpoint**2 * moving
                power_energy = (
                    work / system.drive.motor_efficiency
                    if work >= 0
                    else work * system.drive.regeneration_efficiency
                )
                heat = power_energy - work
                pump_heat = _number(step["pump_heat_j"], "pump_heat_j", nonnegative=True)
                passive = _number(step["passive_heat_j"], "passive_heat_j", nonnegative=True)
                exported = _number(step["boundary_work_out_j"], "boundary_work_out_j")
                scale = max(
                    1.0,
                    abs(power_energy)
                    + abs(work)
                    + pump_work
                    + abs(kinetic)
                    + passive
                    + abs(exported),
                )
                comparisons = [
                    (step["kinetic_energy_change_j"], kinetic),
                    (step["motor_work_j"], work),
                    (step["drag_heat_j"], drag),
                    (step["electrical_energy_j"], power_energy),
                    (step["motor_heat_j"], heat),
                    (work - pump_work - drag, kinetic),
                    (pump_work - step["hydraulic_work_j"], pump_heat),
                    (power_energy - heat - drag - pump_heat - passive - exported, kinetic),
                ]
                if any(
                    not math.isfinite(a) or abs(a - b) > 1e-6 + 2e-7 * scale for a, b in comparisons
                ):
                    raise ConservationError("Pump-drive checkpoint work/energy audit failed")
                electrical.append(power_energy)
                losses.append(heat + drag + pump_heat + passive)
                boundary.append(exported)
                speed = end
                elapsed = math.fsum([elapsed, dt])
            samples = record["control_samples"]
            sample_time = 0.0
            for sample in samples:
                dt = _number(sample["dt_s"], "control_dt_s", positive=True)
                if not math.isclose(
                    sample["time_s"], current.time_s + sample_time, abs_tol=1e-9, rel_tol=0
                ):
                    raise ConservationError("Pump-drive checkpoint controller clock mismatch")
                sample_time = math.fsum([sample_time, dt])
            expected_trip = (current.tripped and not period.get("reset_trip", False)) or period.get(
                "mode", "auto"
            ) == "trip"
            effective_mode = "off" if expected_trip else period.get("mode", "auto")
            if system.controller is not None:
                pid = current.controller_state
                applied = current.applied_torque_nm
                for sample in samples:
                    replay = system.controller.step(
                        pid,
                        period.get("setpoint", 0.0),
                        sample["measurement"],
                        sample["dt_s"],
                        effective_mode,
                        period.get("motor_torque_nm", 0.0),
                        applied,
                    )
                    if any(sample[key] != replay[key] for key in replay):
                        raise ConservationError("Pump-drive checkpoint controller replay mismatch")
                    pid = PIDState(**replay["state"])
                    applied = sample["applied_average_torque_nm"]
            end_state = PumpDriveState.from_dict(record["state"])
            if (
                not math.isclose(elapsed, duration, abs_tol=1e-10, rel_tol=1e-12)
                or not math.isclose(sample_time, duration, abs_tol=1e-10, rel_tol=1e-12)
                or end_state.time_s != current.time_s + duration
                or end_state.model_id != system.model_id
                or end_state.angular_velocity_rad_s != speed
            ):
                raise ConservationError("Pump-drive checkpoint final state mismatch")
            if (
                end_state.controller_state is not None
                and end_state.controller_state.to_dict() != samples[-1]["state"]
            ):
                raise ConservationError("Pump-drive checkpoint controller memory mismatch")
            last_sample = samples[-1]
            torque_sum = math.fsum(
                s["motor_torque_nm"] * s["dt_s"]
                for s in record["steps"]
                if s["time_s"] >= last_sample["time_s"] - 1e-10
            )
            expected_applied = torque_sum / last_sample["dt_s"]
            low = (
                -system.drive.max_braking_torque_nm
                if system.controller is None
                else system.controller.min_output
            )
            high = (
                system.drive.max_motor_torque_nm
                if system.controller is None
                else system.controller.max_output
            )
            expected_applied = min(high, max(low, expected_applied))
            if not math.isclose(
                end_state.applied_torque_nm, expected_applied, abs_tol=1e-9, rel_tol=1e-12
            ):
                raise ConservationError("Pump-drive checkpoint actuator memory mismatch")
            expected_trip = (current.tripped and not period.get("reset_trip", False)) or period.get(
                "mode", "auto"
            ) == "trip"
            if end_state.tripped != expected_trip:
                raise ConservationError("Pump-drive checkpoint trip latch mismatch")
            kinetic = (
                0.5
                * system.drive.inertia_kg_m2
                * (speed - current.angular_velocity_rad_s)
                * (speed + current.angular_velocity_rad_s)
            )
            residual = math.fsum(electrical) - math.fsum(losses) - math.fsum(boundary) - kinetic
            scale = max(1.0, sum(abs(x) for x in electrical + losses + boundary) + abs(kinetic))
            if abs(residual) > 1e-6 + 2e-7 * scale:
                raise ConservationError("Pump-drive accumulated energy audit failed")
            current = end_state
        if current.to_dict() != state["state"]:
            raise ConservationError("Pump-drive checkpoint inventory mismatch")
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
