"""Exact rate-limited first-order connection actuators (dimensionless position)."""

import math
from dataclasses import dataclass

from .liquid import HydraulicConvergenceError, _number


@dataclass(frozen=True)
class ConnectionActuator:
    opening_rate_per_s: float
    closing_rate_per_s: float
    fail_position: float
    time_constant_s: float = 0.0
    max_position_change: float = 0.1

    def __post_init__(self):
        for key in ("opening_rate_per_s", "closing_rate_per_s", "max_position_change"):
            _number(getattr(self, key), key, positive=True)
        _number(self.time_constant_s, "time_constant_s", nonnegative=True)
        self.position(self.fail_position)
        if self.max_position_change > 1:
            raise ValueError("max_position_change must be <= 1")

    @staticmethod
    def position(value):
        if not 0 <= _number(value, "actuator_position") <= 1:
            raise ValueError("Actuator positions/targets must be in [0,1]")
        return float(value)

    def advance(self, position, target, dt, mode="normal"):
        """Exact autonomous trajectory; hydraulic integration remains backward Euler."""
        x, command = self.position(position), self.position(target)
        dt = _number(dt, "actuator_dt_s", nonnegative=True)
        if mode not in ("normal", "fail_safe", "stuck"):
            raise ValueError("Unknown actuator mode")
        target = self.fail_position if mode == "fail_safe" else command
        gap = abs(target - x)
        rate = self.opening_rate_per_s if target > x else self.closing_rate_per_s
        if mode == "stuck" or gap == 0 or dt == 0:
            end = x
        elif self.time_constant_s == 0:
            end = x + math.copysign(min(gap, rate * dt), target - x)
        else:
            transition = max(0.0, (gap - rate * self.time_constant_s) / rate)
            linear_time = min(dt, transition)
            movement = rate * linear_time
            if dt > transition:
                tail = gap - movement
                movement += tail * -math.expm1(-(dt - transition) / self.time_constant_s)
            end = x + math.copysign(min(gap, movement), target - x)
        return {
            "initial_position": x,
            "command": command,
            "effective_target": float(target),
            "mode": mode,
            "final_position": min(1.0, max(0.0, end)),
        }

    def step(self, position, target, dt, mode="normal"):
        record = self.advance(position, target, dt, mode)
        if abs(record["final_position"] - position) > self.max_position_change * (1 + 1e-12):
            raise HydraulicConvergenceError("Actuator motion exceeds substep accuracy guard")
        return record
