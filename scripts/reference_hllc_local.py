"""Independent scalar NumPy local-wave HLLC reference for serialized audits."""

import numpy as np
from check_pr_momentum_vessels import port_flux


def wave_flux(left, right, sound_left, sound_right, bound, estimate="davis"):
    u0, f0, p0, v0, r0, s0 = left
    u1, f1, p1, v1, r1, s1 = right
    if estimate == "pvrs":
        predictor = max(
            0.0, 0.5 * (p0 + p1) - 0.125 * (v1 - v0) * (r0 + r1) * (sound_left + sound_right)
        )
        g0 = r0 * sound_left**2 / p0
        g1 = r1 * sound_right**2 / p1
        sl = v0 - sound_left * np.sqrt(1 + (g0 + 1) / (2 * g0) * max(0.0, predictor / p0 - 1))
        sr = v1 + sound_right * np.sqrt(1 + (g1 + 1) / (2 * g1) * max(0.0, predictor / p1 - 1))
    else:
        sl = min(v0 - sound_left, v1 - sound_right)
        sr = max(v0 + sound_left, v1 + sound_right)
    contact = (p1 - p0 + r0 * (sl - v0) * v0 - r1 * (sr - v1) * v1) / (
        r0 * (sl - v0) - r1 * (sr - v1)
    )
    pressure = p0 + r0 * (sl - v0) * (contact - v0)
    ratio0 = (sl - v0) / (sl - contact)
    ratio1 = (sr - v1) / (sr - contact)
    if pressure <= 0 or ratio0 <= 0 or ratio1 <= 0 or not sl < contact < sr:
        return (*port_flux(left, right, bound), True)
    if sl >= 0:
        return f0, s0 * v0, v0, p0, False
    if sr <= 0:
        return f1, s1 * v1, v1, p1, False
    u, f, p, v, r, s, speed = (
        (u0, f0, p0, v0, r0, s0, sl) if contact >= 0 else (u1, f1, p1, v1, r1, s1, sr)
    )
    ratio = (speed - v) / (speed - contact)
    us = np.r_[
        u[:-2] * ratio,
        r * ratio * contact,
        ((speed - v) * u[-1] - p * v + pressure * contact) / (speed - contact),
    ]
    return f + speed * (us - u), s * ratio * contact, contact, pressure, False
