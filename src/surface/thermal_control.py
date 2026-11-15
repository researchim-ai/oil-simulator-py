"""Shared sampled thermal protection and utility allocation semantics."""

import math

from .liquid_protection import LiquidTripMemory


def sample_temperature_trip(rule, memory, value, now, tolerance_s=0.0):
    """Latch only at protection samples; never imply continuous detection."""
    if memory.tripped_at_s is not None:
        return memory, False
    since = now if memory.armed_since_s is None else memory.armed_since_s
    bad = (
        (now if memory.violating_since_s is None else memory.violating_since_s)
        if rule.violating(value)
        else None
    )
    trigger = bad is not None and now - bad + tolerance_s >= rule.delay_s
    return LiquidTripMemory(since, bad, now if trigger else None), trigger


def allocate_thermal_duties(requested, max_heating_w, max_cooling_w):
    """Same proportional allocation for controlled heaters and coolers."""
    held = requested.copy()
    for cap, sign in ((max_heating_w, 1), (max_cooling_w, -1)):
        total = math.fsum(max(0.0, sign * v) for v in held.values())
        if cap is not None and total > cap:
            for name, value in held.items():
                if sign * value > 0:
                    held[name] = value * cap / total
    return held
