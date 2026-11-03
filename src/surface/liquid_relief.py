"""Memoryless differential-pressure relief characteristic and liquid sensors.

Finite-area, one-way orifice with linear spring lift between cracking and
full-lift differential pressures. No chatter, stem inertia, flashing or sizing
certification. Flow and work are solved inside the liquid network equations.
"""

import math
from dataclasses import dataclass, replace

from scipy.optimize import brentq

from .liquid import EquipmentOperatingError, _number


@dataclass(frozen=True)
class LiquidRelief:
    cracking_pressure_pa: float
    full_open_pressure_pa: float

    def __post_init__(self):
        _number(self.cracking_pressure_pa, "cracking_pressure_pa", nonnegative=True)
        _number(self.full_open_pressure_pa, "full_open_pressure_pa", positive=True)
        if self.full_open_pressure_pa <= self.cracking_pressure_pa:
            raise ValueError("Full lift pressure must exceed cracking pressure")

    def opening(self, differential_pressure_pa):
        dp = _number(differential_pressure_pa, "differential_pressure_pa")
        return min(
            1.0,
            max(
                0.0,
                (dp - self.cracking_pressure_pa)
                / (self.full_open_pressure_pa - self.cracking_pressure_pa),
            ),
        )

    def volume_rate(self, valve, differential_pressure_pa, liquid):
        dp = _number(differential_pressure_pa, "differential_pressure_pa")
        opening = self.opening(dp)
        area = math.pi * valve.diameter_m**2 / 4 * opening
        return (
            valve.discharge_coefficient * area * math.sqrt(2 * max(0.0, dp) / liquid.density_kg_m3)
        )


def inspect_liquid_pumps(physics, state, boundary_pressures_pa=None):
    """Algebraic pump flows at ACTUAL node pressures, not a network steady solve."""
    physics._check(state)
    if physics.backend == "torch":
        if physics._torch_bridge is None:
            from .torch_liquid_bridge import TorchLiquidBridge

            physics._torch_bridge = TorchLiquidBridge(physics, physics.device)
        return physics._torch_bridge.sensors(state, boundary_pressures_pa)
    boundary, _ = physics._controls(boundary_pressures_pa)
    pressures = {**boundary, **dict(state.pressures_pa)}
    speeds = dict(state.speeds_rad_s)
    result = {}
    for pump in physics.network.pumps:
        speed = speeds[pump.name]
        if speed == 0:
            result[pump.name] = {"flow_m3_s": 0.0, "speed_ratio": 0.0, "npsh_margin_m": None}
            continue
        effective = replace(pump, speed_ratio=speed / physics.nominal_speeds_rad_s[pump.name])
        dp = pressures[pump.start] - pressures[pump.end]
        if dp <= effective.loss_and_gradient(0.0, physics.network.fluid)[0]:
            flow = 0.0
        else:
            upper = effective.speed_ratio * effective.max_flow_m3_s
            if dp > effective.loss_and_gradient(upper, physics.network.fluid)[0]:
                raise EquipmentOperatingError(
                    f"{pump.name}: sensor flow outside forward pump curve"
                )
            flow = brentq(
                lambda q, effective=effective, dp=dp: (
                    effective.loss_and_gradient(q, physics.network.fluid)[0] - dp
                ),
                0.0,
                upper,
                xtol=1e-15,
                rtol=1e-13,
            )
        _, _, npsh = effective.operating_state(flow, pressures[pump.start], physics.network.fluid)
        required = effective.required_npsh_at(flow)
        result[pump.name] = {
            "flow_m3_s": flow,
            "speed_ratio": effective.speed_ratio,
            "npsh_margin_m": None if required is None else npsh - required,
        }
    return result
