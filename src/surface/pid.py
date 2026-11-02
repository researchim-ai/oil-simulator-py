"""Sampled PID: derivative on measurement, filtered D and actuator tracking."""

from dataclasses import asdict, dataclass

from .liquid import _number
from .operation import _digest


@dataclass(frozen=True)
class PIDState:
    integral_output: float
    previous_measurement: float
    filtered_derivative: float
    previous_output: float
    mode: str
    model_id: str

    def __post_init__(self):
        for key in (
            "integral_output",
            "previous_measurement",
            "filtered_derivative",
            "previous_output",
        ):
            _number(getattr(self, key), key)
        if self.mode not in ("auto", "manual", "off"):
            raise ValueError("Unknown PID mode")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PIDController:
    proportional_gain: float
    integral_gain: float
    derivative_gain: float
    min_output: float
    max_output: float
    derivative_filter_s: float = 0.1
    tracking_time_s: float = 0.5
    bias: float = 0.0
    direction: int = 1

    def __post_init__(self):
        for key in ("proportional_gain", "integral_gain", "derivative_gain", "derivative_filter_s"):
            _number(getattr(self, key), key, nonnegative=True)
        _number(self.tracking_time_s, "tracking_time_s", positive=True)
        for key in ("min_output", "max_output", "bias"):
            _number(getattr(self, key), key)
        if not self.min_output <= 0 <= self.max_output or self.min_output == self.max_output:
            raise ValueError("PID bounds must straddle/include zero and have positive width")
        if isinstance(self.direction, bool) or self.direction not in (-1, 1):
            raise ValueError("PID direction must be +1 or -1")

    @property
    def model_id(self):
        return _digest({"model": "sampled_tracking_PID_v1", **asdict(self)})

    def initialize(self, measurement, output=0.0, setpoint=None, mode="manual"):
        measurement = _number(measurement, "measurement")
        output = _number(output, "output")
        if not self.min_output <= output <= self.max_output:
            raise ValueError("Initial PID output outside bounds")
        target = measurement if setpoint is None else _number(setpoint, "setpoint")
        proportional = self.direction * self.proportional_gain * (target - measurement)
        return PIDState(
            output - self.bias - proportional, measurement, 0, output, mode, self.model_id
        )

    def step(
        self,
        state,
        setpoint,
        measurement,
        duration_s,
        mode="auto",
        manual_output=0.0,
        applied_output=None,
    ):
        if not isinstance(state, PIDState) or state.model_id != self.model_id:
            raise ValueError("PID state/model mismatch")
        if mode not in ("auto", "manual", "off"):
            raise ValueError("Unknown PID mode")
        target = _number(setpoint, "setpoint")
        measured = _number(measurement, "measurement")
        duration = _number(duration_s, "duration_s", positive=True)
        applied = (
            state.previous_output
            if applied_output is None
            else _number(applied_output, "applied_output")
        )
        if not self.min_output <= applied <= self.max_output:
            raise ValueError("Applied PID output outside bounds")
        raw_derivative = (measured - state.previous_measurement) / duration
        weight = duration / (self.derivative_filter_s + duration)
        filtered = state.filtered_derivative + weight * (raw_derivative - state.filtered_derivative)
        error = self.direction * (target - measured)
        proportional = self.proportional_gain * error
        derivative = -self.direction * self.derivative_gain * filtered
        tracking_weight = duration / (self.tracking_time_s + duration)
        if mode != "auto":
            requested = 0.0 if mode == "off" else _number(manual_output, "manual_output")
            command = min(self.max_output, max(self.min_output, requested))
            integral = command - self.bias - proportional - derivative
        elif state.mode != "auto":
            # First automatic sample tracks the actually applied actuator,
            # including external power/speed limits. No integral kick on transfer.
            command = applied
            requested = applied
            integral = applied - self.bias - proportional - derivative
        else:
            integral = state.integral_output + tracking_weight * (applied - state.previous_output)
            integral += self.integral_gain * error * duration
            requested = self.bias + proportional + integral + derivative
            command = min(self.max_output, max(self.min_output, requested))
            integral += tracking_weight * (command - requested)
        next_state = PIDState(integral, measured, filtered, command, mode, self.model_id)
        return {
            "state": next_state.to_dict(),
            "output": command,
            "unbounded_output": requested,
            "saturated": command != requested,
            "error": error,
            "proportional": proportional,
            "derivative": derivative,
        }
