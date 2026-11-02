"""Exact one-direction shaft mechanics with constant commanded/load torques.

Standalone CPU building block. No implicit hydraulic coupling, motor current,
thermal motor model, compressor surge map or reverse rotation is implied.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

from .liquid import ConservationError, EquipmentOperatingError, Pump, _number
from .operation import _digest


@dataclass(frozen=True)
class RotorState:
    angular_velocity_rad_s: float
    time_s: float
    model_id: str

    def __post_init__(self):
        _number(self.angular_velocity_rad_s, "angular_velocity_rad_s", nonnegative=True)
        _number(self.time_s, "time_s", nonnegative=True)
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("Rotor state requires a model identifier")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RotatingDrive:
    inertia_kg_m2: float
    max_speed_rad_s: float
    max_motor_torque_nm: float
    max_braking_torque_nm: float
    viscous_drag_nm_s: float = 0.0
    motor_efficiency: float = 1.0
    regeneration_efficiency: float = 1.0
    max_electrical_draw_w: float | None = None
    max_electrical_export_w: float | None = None

    def __post_init__(self):
        for key in ("inertia_kg_m2", "max_speed_rad_s", "max_motor_torque_nm"):
            _number(getattr(self, key), key, positive=True)
        for key in ("max_braking_torque_nm", "viscous_drag_nm_s"):
            _number(getattr(self, key), key, nonnegative=True)
        if not 0 < _number(self.motor_efficiency, "motor_efficiency") <= 1:
            raise ValueError("Motor efficiency must be in (0,1]")
        if not 0 <= _number(self.regeneration_efficiency, "regeneration_efficiency") <= 1:
            raise ValueError("Regeneration efficiency must be in [0,1]")
        for key in ("max_electrical_draw_w", "max_electrical_export_w"):
            if getattr(self, key) is not None:
                _number(getattr(self, key), key, nonnegative=True)

    @property
    def model_id(self):
        return _digest({"model": "constant_torque_rotor_v1", **asdict(self)})

    def initialize(self, angular_velocity_rad_s=0.0, time_s=0.0):
        state = RotorState(angular_velocity_rad_s, time_s, self.model_id)
        self._check(state)
        return state

    def _check(self, state):
        if not isinstance(state, RotorState) or state.model_id != self.model_id:
            raise ValueError("Rotor inventory/model mismatch")
        if state.angular_velocity_rad_s > self.max_speed_rad_s:
            raise EquipmentOperatingError("Rotor speed exceeds declared limit")

    def _trajectory(self, speed, torque, duration):
        """End speed, angle and integral of squared speed, without cancellation."""
        if duration == 0:
            return speed, 0.0, 0.0
        k = self.viscous_drag_nm_s / self.inertia_kg_m2
        x = k * duration
        if x < 0.5:
            # Entire functions; includes k=0 exactly. phi3 integrates
            # ((1-exp(-k*t))/k)^2 and tends to duration^3/3.
            phi1 = sum((-x) ** n / math.factorial(n + 1) for n in range(18))
            phi2 = sum((-x) ** n / math.factorial(n + 2) for n in range(18))
            phi3 = sum((-x) ** n * (2 ** (n + 2) - 2) / math.factorial(n + 3) for n in range(18))
            acceleration = (torque - self.viscous_drag_nm_s * speed) / self.inertia_kg_m2
            end = speed + acceleration * duration * phi1
            angle = speed * duration + acceleration * duration**2 * phi2
            square = (
                speed**2 * duration
                + 2 * speed * acceleration * duration**2 * phi2
                + acceleration**2 * duration**3 * phi3
            )
        else:
            equilibrium = torque / self.viscous_drag_nm_s
            delta = speed - equilibrium
            phi1 = -math.expm1(-x) / x
            phi_double = -math.expm1(-2 * x) / (2 * x)
            end = equilibrium + delta * math.exp(-x)
            angle = duration * (equilibrium + delta * phi1)
            square = duration * (
                equilibrium**2 + 2 * equilibrium * delta * phi1 + delta**2 * phi_double
            )
        return end, angle, square

    def advance(self, state, duration_s, motor_torque_nm, load_torque_nm=0.0):
        """Exact J*domega/dt = motor-load-b*omega until unilateral rest.

        Torques are constant over this interval. Limits reject the command;
        they do not silently clip it. Power is checked at the trajectory peak.
        """
        self._check(state)
        duration = _number(duration_s, "duration_s", positive=True)
        motor = _number(motor_torque_nm, "motor_torque_nm")
        load = _number(load_torque_nm, "load_torque_nm", nonnegative=True)
        if not -self.max_braking_torque_nm <= motor <= self.max_motor_torque_nm:
            raise EquipmentOperatingError("Motor torque outside declared limits")
        time = state.time_s + duration
        if not math.isfinite(time) or time <= state.time_s:
            raise ValueError("Rotor interval cannot advance representable time")
        net = motor - load
        speed = state.angular_velocity_rad_s
        moving = duration
        if net < 0:
            stopping = (
                -self.inertia_kg_m2 * speed / net
                if self.viscous_drag_nm_s == 0
                else self.inertia_kg_m2
                / self.viscous_drag_nm_s
                * math.log1p(self.viscous_drag_nm_s * speed / -net)
            )
            moving = min(duration, stopping)
        try:
            end, angle, square = self._trajectory(speed, net, moving)
        except (OverflowError, ZeroDivisionError) as error:
            raise ConservationError("Rotor trajectory is not numerically representable") from error
        if not all(math.isfinite(v) for v in (end, angle, square)):
            raise ConservationError("Rotor trajectory is not numerically representable")
        # At an analytically located stop, roundoff is not a reverse rotor.
        if moving < duration or (net < 0 and moving == stopping):
            end = 0.0
        next_state = self.initialize(end, time)
        peak_speed = max(speed, end)  # Autonomous affine trajectory is monotone.
        peak_draw = max(0.0, motor) * peak_speed / self.motor_efficiency
        peak_export = max(0.0, -motor) * peak_speed * self.regeneration_efficiency
        if self.max_electrical_draw_w is not None and peak_draw > self.max_electrical_draw_w:
            raise EquipmentOperatingError("Electrical draw limit exceeded along rotor trajectory")
        if self.max_electrical_export_w is not None and peak_export > self.max_electrical_export_w:
            raise EquipmentOperatingError("Regeneration limit exceeded along rotor trajectory")
        shaft_work = motor * angle
        load_work = load * angle
        drag_heat = self.viscous_drag_nm_s * square
        electrical = (
            shaft_work / self.motor_efficiency
            if shaft_work >= 0
            else shaft_work * self.regeneration_efficiency
        )
        motor_heat = electrical - shaft_work
        kinetic_change = 0.5 * self.inertia_kg_m2 * (end - speed) * (end + speed)
        shaft_residual = shaft_work - load_work - drag_heat - kinetic_change
        shaft_throughput = abs(shaft_work) + load_work + drag_heat + abs(kinetic_change)
        residual = electrical - motor_heat - load_work - drag_heat - kinetic_change
        throughput = abs(electrical) + motor_heat + load_work + drag_heat + abs(kinetic_change)
        if (
            not all(
                math.isfinite(v) for v in (residual, throughput, shaft_residual, shaft_throughput)
            )
            or min(angle, square, motor_heat, drag_heat) < 0
            or abs(shaft_residual) > 1e-9 + 1e-10 * shaft_throughput
            or abs(residual) > 1e-9 + 1e-10 * throughput
        ):
            raise ConservationError("Rotor energy balance failed")
        return {
            "state": next_state.to_dict(),
            "duration_s": duration,
            "moving_duration_s": moving,
            "angle_rad": angle,
            "motor_work_j": shaft_work,
            "electrical_energy_j": electrical,
            "load_work_j": load_work,
            "drag_heat_j": drag_heat,
            "motor_heat_j": motor_heat,
            "kinetic_energy_change_j": kinetic_change,
            "energy_residual_j": residual,
            "shaft_energy_residual_j": shaft_residual,
            "peak_electrical_draw_w": peak_draw,
            "peak_electrical_export_w": peak_export,
            "rest_constraint_impulse_nm_s": max(0.0, -net) * (duration - moving),
        }


def pump_shaft_load(pump, liquid, speed_rad_s, nominal_speed_rad_s, flow_m3_s, suction_pressure_pa):
    """Load sample from the existing pump map; NOT a dynamic network solve."""
    if not isinstance(pump, Pump):
        raise TypeError("Expected Pump")
    speed = _number(speed_rad_s, "speed_rad_s", nonnegative=True)
    nominal = _number(nominal_speed_rad_s, "nominal_speed_rad_s", positive=True)
    flow = _number(flow_m3_s, "flow_m3_s", nonnegative=True)
    _number(suction_pressure_pa, "suction_pressure_pa", positive=True)
    if not pump.open:
        raise EquipmentOperatingError("Cannot sample a closed pump as a running shaft load")
    if speed == 0:
        if flow != 0:
            raise EquipmentOperatingError("Stationary pump does not admit imposed forward flow")
        return {"load_torque_nm": 0.0, "shaft_power_w": 0.0, "hydraulic_power_w": 0.0}
    actual = replace(pump, speed_ratio=speed / nominal)
    hydraulic, shaft, _ = actual.operating_state(flow, suction_pressure_pa, liquid)
    if hydraulic < 0:
        raise EquipmentOperatingError("Pump turbine operation is not supported")
    return {"load_torque_nm": shaft / speed, "shaft_power_w": shaft, "hydraulic_power_w": hydraulic}


def run_drive_schedule(drive, initial, periods, checkpoint=None, stop_after=None):
    """Checked restart; replaying the cheap analytic history audits every work term."""
    from pathlib import Path

    drive._check(initial)
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Drive schedule requires periods")
    for period in periods:
        if not isinstance(period, dict) or not {"duration_s", "motor_torque_nm"} <= period.keys():
            raise ValueError("Drive period requires duration and commanded torque")
        if set(period) - {"duration_s", "motor_torque_nm", "load_torque_nm"}:
            raise ValueError("Unknown drive period fields")
        _number(period["duration_s"], "duration_s", positive=True)
        _number(period["motor_torque_nm"], "motor_torque_nm")
        _number(period.get("load_torque_nm", 0), "load_torque_nm", nonnegative=True)
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0
    ):
        raise ValueError("stop_after must be a nonnegative integer")
    root = Path(__file__).resolve().parent
    fingerprint = _digest(
        {
            "model": asdict(drive),
            "initial": initial.to_dict(),
            "periods": periods,
            "source": {
                name: (root / name).read_text()
                for name in ("rotating_drive.py", "liquid.py", "operation.py", "curves.py")
            },
        }
    )
    current = initial
    records = []
    if checkpoint is not None:
        import copy

        saved = copy.deepcopy(checkpoint)
        if set(saved) != {"fingerprint", "next_period", "state", "records", "checksum"}:
            raise ValueError("Invalid drive checkpoint fields")
        checksum = saved.pop("checksum")
        if checksum != _digest(saved) or saved["fingerprint"] != fingerprint:
            raise ValueError("Drive checkpoint checksum/fingerprint mismatch")
        index = saved["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(saved["records"]) != index
        ):
            raise ValueError("Drive checkpoint index mismatch")
        for i, record in enumerate(saved["records"]):
            expected = {"period": i, **drive.advance(current, **periods[i])}
            if record != expected:
                raise ConservationError("Drive checkpoint replay/work audit failed")
            current = RotorState(**expected["state"])
            records.append(expected)
        if current.to_dict() != saved["state"]:
            raise ConservationError("Drive checkpoint state audit failed")
    start = len(records)
    end = len(periods) if stop_after is None else min(len(periods), start + stop_after)
    for i in range(start, end):
        record = {"period": i, **drive.advance(current, **periods[i])}
        current = RotorState(**record["state"])
        records.append(record)
    result = {
        "fingerprint": fingerprint,
        "next_period": len(records),
        "state": current.to_dict(),
        "records": records,
    }
    return {**result, "checksum": _digest(result)}
