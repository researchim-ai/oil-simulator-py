"""Independent scalar Holmes equations with SciPy volume-flux inversion."""

import math

import numpy as np
from scipy.optimize import brentq


def parameters(law, alpha, gas_density, liquid_density, mass_velocity, diameter, axial, gravity):
    direction = math.copysign(1, axial)
    cosine = abs(axial) / gravity
    if not math.cos(math.radians(law["maximum_deviation_deg"])) <= cosine <= 1 + 1e-12:
        raise ValueError("Reference Holmes inclination outside domain")
    cosine = min(1, cosine)
    tilt = math.sqrt(cosine) * (1 + math.sqrt(max(0, 1 - cosine**2))) ** 2
    rg, rl = gas_density, liquid_density
    dd = diameter * math.sqrt(gravity * (rl - rg) / law["surface_tension_n_m"])
    if not law["minimum_dimensionless_diameter"] <= dd <= law["maximum_dimensionless_diameter"]:
        raise ValueError("Reference Holmes diameter outside domain")
    ku = float(np.interp(dd, [2, 4, 10, 14, 20, 28, 50], [0, 1, 2.1, 2.5, 2.8, 3, 3.2]))
    base = (law["surface_tension_n_m"] * gravity * (rl - rg) / rl**2) ** 0.25
    film = ku * math.sqrt(rl / rg) * base
    mean = alpha * rg + (1 - alpha) * rl

    def relation(j):
        beta = max(alpha, law["velocity_sensitivity"] * alpha * j / film)
        reduced = max(
            0,
            min(
                1, (beta - law["profile_flattening_start"]) / (1 - law["profile_flattening_start"])
            ),
        )
        c0 = law["profile_parameter"] / (1 + (law["profile_parameter"] - 1) * reduced**2)
        if alpha <= law["bubble_transition_start"]:
            k = 1.53 / c0
        elif alpha >= law["bubble_transition_end"]:
            k = ku
        else:
            w = (alpha - law["bubble_transition_start"]) / (
                law["bubble_transition_end"] - law["bubble_transition_start"]
            )
            k = (1 - w) * 1.53 / c0 + w * ku
        vd = (
            (1 - alpha * c0)
            * c0
            * k
            / (math.sqrt(rg / rl) * alpha * c0 + 1 - alpha * c0)
            * base
            * tilt
        )
        return c0, vd

    # Independently verify a sufficient global uniqueness bound. A positive
    # slope at one root alone does not rule out multiple admissible branches.
    cmax = relation(0)[0]
    weight = max(
        0,
        min(
            1,
            (alpha - law["bubble_transition_start"])
            / (law["bubble_transition_end"] - law["bubble_transition_start"]),
        ),
    )
    ratio = math.sqrt(rg / rl)
    bmin = 1 - (1 - ratio) * alpha * cmax
    ckmax = 1.53 * (1 - weight) + ku * weight * cmax
    dvd_min = -base * tilt * ckmax * alpha * ratio / bmin**2
    j_min = (
        max(alpha, law["profile_flattening_start"]) * film / (law["velocity_sensitivity"] * alpha)
    )
    a = law["profile_parameter"]
    peak = min(1, 1 / math.sqrt(3 * (a - 1))) if a > 1 else 1
    dc_max = (
        2
        * a
        * (a - 1)
        * peak
        / (1 + (a - 1) * peak**2) ** 2
        * law["velocity_sensitivity"]
        * alpha
        / (film * (1 - law["profile_flattening_start"]))
    )
    bound = rl - alpha * (rl - rg) * cmax - alpha * (rl - rg) * max(0, -(j_min + dvd_min)) * dc_max
    if bound <= 1e-8 * rl:
        raise ValueError("Reference Holmes global monotonicity domain rejected")

    def residual(j):
        c0, vd = relation(j)
        return rl * j + alpha * (rg - rl) * (c0 * j + vd) - mean * mass_velocity * direction

    bound = max(1.0, abs(mass_velocity))
    for _ in range(64):
        if residual(-bound) <= 0 <= residual(bound):
            j = brentq(residual, -bound, bound, xtol=1e-13, rtol=1e-14)
            c0, vd = relation(j)
            return c0, vd * direction
        bound *= 2
    raise ValueError("No bracket for reference Holmes volume flux")
