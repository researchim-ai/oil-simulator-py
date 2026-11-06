"""Torque actuators and sampled, latched protections for the common PR plant."""

import math
from dataclasses import dataclass

from .actuator import ConnectionActuator
from .liquid import _name, _number


@dataclass(frozen=True)
class PRMotorControl:
    name: str
    machine: str
    actuator: ConnectionActuator
    minimum_torque_nm: float
    maximum_torque_nm: float
    model: str = "motor_torque"

    def __post_init__(self):
        _name(self.name)
        _name(self.machine)
        if self.model != "motor_torque" or not isinstance(self.actuator, ConnectionActuator):
            raise ValueError("Motor control requires a physical actuator")
        _number(self.minimum_torque_nm, "minimum controlled torque")
        _number(self.maximum_torque_nm, "maximum controlled torque")
        if not self.minimum_torque_nm <= 0 < self.maximum_torque_nm:
            raise ValueError("Controlled torque interval must include zero and positive drive")

    @property
    def port(self):
        return self.machine

    def torque(self, position):
        x = ConnectionActuator.position(position)
        return self.minimum_torque_nm + x * (self.maximum_torque_nm - self.minimum_torque_nm)


@dataclass(frozen=True)
class PRSurfaceTrip:
    name: str
    target: str
    measurement: str
    sensor: str
    trip_threshold: float
    reset_threshold: float
    delay_s: float = 0.0
    direction: str = "high"
    isolate_switches: tuple = ()

    def __post_init__(self):
        for key in ("name", "target", "sensor"):
            _name(getattr(self, key))
        if self.measurement not in ("speed", "pressure", "temperature") or self.direction not in (
            "high",
            "low",
        ):
            raise ValueError("Explicit protection sensor and direction required")
        for key in ("trip_threshold", "reset_threshold", "delay_s"):
            _number(getattr(self, key), key, nonnegative=True)
        if not (
            self.reset_threshold < self.trip_threshold
            if self.direction == "high"
            else self.reset_threshold > self.trip_threshold
        ):
            raise ValueError("Protection requires a nonzero reset hysteresis")
        object.__setattr__(self, "isolate_switches", tuple(self.isolate_switches))
        if len(set(self.isolate_switches)) != len(self.isolate_switches):
            raise ValueError("Duplicate protection isolation switches")
        for name in self.isolate_switches:
            _name(name)

    def read(self, plant):
        if self.measurement == "speed":
            return dict(plant.transport.rotors)[self.sensor]
        cell = dict(plant.transport.nodes)[self.sensor]
        return cell.pressure_pa if self.measurement == "pressure" else cell.temperature_k

    def unsafe(self, value):
        return (
            value >= self.trip_threshold
            if self.direction == "high"
            else value <= self.trip_threshold
        )

    def can_reset(self, value):
        return (
            value <= self.reset_threshold
            if self.direction == "high"
            else value >= self.reset_threshold
        )

    def step(self, latched, elapsed, value, dt):
        active = latched or (self.delay_s == 0 and self.unsafe(value))
        if latched:
            return True, elapsed, active
        if not self.unsafe(value):
            return False, 0.0, False
        elapsed = min(self.delay_s, math.fsum((elapsed, dt)))
        reached = elapsed >= self.delay_s or math.isclose(
            elapsed, self.delay_s, rel_tol=1e-13, abs_tol=1e-15
        )
        return reached, self.delay_s if reached else elapsed, active


def compressor_margin(unit, transport_state):
    """Stable-map flow margin from the finite inlet/outlet and actual shaft.

    This is a sampled sensor reconstruction, not a second compressor solve.
    Pressure and speed outside the accepted map are never extrapolated.
    """
    import bisect

    from .liquid import PropertyModelError

    cells = dict(transport_state.nodes)
    source, target = cells[unit.source], cells[unit.target]
    speed = dict(transport_state.rotors).get(unit.name, unit.speed_rad_s)
    corrected = speed / math.sqrt(source.temperature_k / unit.card.reference_temperature_k)
    axis = unit.card.corrected_speed_rad_s
    tolerance = 64 * math.ulp(max(1.0, axis[-1]))
    if not axis[0] - tolerance <= corrected <= axis[-1] + tolerance:
        raise PropertyModelError("Anti-surge sensor speed outside the compressor map")
    corrected = min(axis[-1], max(axis[0], corrected))
    i = min(max(bisect.bisect_right(axis, corrected) - 1, 0), len(axis) - 2)
    fraction = (corrected - axis[i]) / (axis[i + 1] - axis[i])
    flow = [
        (1 - fraction) * a + fraction * b
        for a, b in zip(
            unit.card.corrected_mass_flow_kg_s[i], unit.card.corrected_mass_flow_kg_s[i + 1]
        )
    ]
    ratio = [
        (1 - fraction) * a + fraction * b
        for a, b in zip(unit.card.pressure_ratio[i], unit.card.pressure_ratio[i + 1])
    ]
    requested = target.pressure_pa / source.pressure_pa
    if requested > ratio[0] + 64 * math.ulp(max(1.0, ratio[0])):
        raise PropertyModelError("Anti-surge sensor pressure outside the stable compressor map")
    if requested <= ratio[-1]:
        actual = flow[-1]
    else:
        j = min(max(bisect.bisect_right([-r for r in ratio], -requested) - 1, 0), len(ratio) - 2)
        fraction = (ratio[j] - min(ratio[0], requested)) / (ratio[j] - ratio[j + 1])
        actual = flow[j] + fraction * (flow[j + 1] - flow[j])
    return actual / flow[0] - 1
