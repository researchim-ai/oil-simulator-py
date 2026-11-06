"""Independent NumPy/SciPy disk-face/profile/HLLC balance reconstruction.

Uses the existing independent NumPy PR reference and SciPy chemical traces;
never calls the runtime layered module, column solve, or Torch face operator.
"""

import math

import numpy as np
from check_pr_momentum_vessels import port_flux
from check_stratified_pr_flow import profile
from check_stratified_pr_network import reference_darcy


def patch(bottom, a, b, diameter, cosine, order=16):
    if b <= a:
        return np.empty(0), np.empty(0)
    low, high = np.arcsin(np.clip(2 * (np.array([a, b]) - bottom) / (diameter * cosine) - 1, -1, 1))
    points, weights = np.polynomial.legendre.leggauss(order)
    phi = (low + high) / 2 + (high - low) / 2 * points
    heights = bottom + diameter * cosine / 2 * (1 + np.sin(phi))
    weights = weights * (high - low) * diameter**2 / 4 * np.cos(phi) ** 2
    return heights, weights


def reconstruct(model, state, order=16):
    columns = model["columns"]
    geometry = columns[0]["geometry"]
    angle = geometry["inclination_rad"]
    cosine = math.cos(angle)
    diameter = geometry["diameter_m"]
    gravity = columns[0]["gravity_m_s2"]
    datum = columns[0]["elevation_m"]
    decoded = [profile(c["profile"], cfg) for c, cfg in zip(state["cells"], columns)]
    mw = np.array([c.molar_mass_kg_mol for c in decoded[0][0].components])
    n = len(columns)
    rates = [np.zeros(len(mw) + 2) for _ in columns]
    pressure = np.zeros((n, 2))
    wall_pressure = np.zeros((n, 2))
    masses = []
    bounds = []
    for cfg, cell in zip(columns, state["cells"]):
        p = cell["profile"]
        masses.append(np.asarray(p["component_moles"]) @ mw)
        bounds.append(
            np.r_[
                cfg["elevation_m"],
                np.asarray(p["interface_heights_m"]) + cfg["elevation_m"],
                cfg["elevation_m"] + decoded[len(bounds)][1].height_m,
            ]
        )

    def bottom(i, side):
        cfg = columns[i]
        return (
            cfg["elevation_m"]
            + decoded[i][1].height_m / 2
            + side * cfg["geometry"]["length_m"] / 2 * math.sin(angle)
            - diameter * cosine / 2
        )

    def packet(i, phase, height):
        cfg = columns[i]
        p, x, q = decoded[i][2](height - cfg["elevation_m"], phase)
        v = state["cells"][i]["momentum_kg_m_s"] / masses[i]
        energy = q["u"] / q["vm"] + q["rho"] * (0.5 * v * v + gravity * (height - datum))
        u = np.r_[x / q["vm"], q["rho"] * v, energy]
        flux = np.r_[x / q["vm"] * v, q["rho"] * v * v + p, (energy + p) * v]
        return u, flux, p, v, q["rho"], q["s"] / q["vm"]

    for i in range(n):
        for k, side in enumerate((-1, 1)):
            base = bottom(i, side)
            for phase, (a, b) in enumerate(zip(bounds[i][:-1], bounds[i][1:])):
                h, w = patch(
                    base, max(a, base), min(b, base + diameter * cosine), diameter, cosine, order
                )
                for height, weight in zip(h, w):
                    inside = packet(i, phase, height)
                    pressure[i, k] += weight * inside[2]
                    if (i == 0 and side == -1) or (i == n - 1 and side == 1):
                        u, flux, p, velocity, rho, entropy = inside
                        ghost = (
                            np.r_[u[:-2], -u[-2], u[-1]],
                            np.r_[-flux[:-2], flux[-2], -flux[-1]],
                            p,
                            -velocity,
                            rho,
                            entropy,
                        )
                        left, right = (ghost, inside) if side == -1 else (inside, ghost)
                        fan = port_flux(left, right, model["wave_speed_bound_m_s"])
                        wall_pressure[i, k] += weight * fan[0][-2]
    for i in range(n - 1):
        base = bottom(i, 1)
        for pa, (a, b) in enumerate(zip(bounds[i][:-1], bounds[i][1:])):
            for pb, (c, d) in enumerate(zip(bounds[i + 1][:-1], bounds[i + 1][1:])):
                h, w = patch(
                    base,
                    max(a, c, base),
                    min(b, d, base + diameter * cosine),
                    diameter,
                    cosine,
                    order,
                )
                for height, weight in zip(h, w):
                    flux = port_flux(
                        packet(i, pa, height),
                        packet(i + 1, pb, height),
                        model["wave_speed_bound_m_s"],
                    )[0]
                    rates[i] -= weight * flux
                    rates[i + 1] += weight * flux
    viscosity = model.get("dynamic_viscosity_pa_s", 0.0)
    roughness = model.get("roughness_m", 0.0)
    friction = np.zeros(n)
    if viscosity > 0:
        for i in range(n):
            velocity = state["cells"][i]["momentum_kg_m_s"] / masses[i]
            volume = decoded[i][1].volume_m3
            friction[i] = (
                -volume
                * reference_darcy(
                    velocity, masses[i] / volume, 1.0, diameter, viscosity, roughness
                )[0]
            )
    return {
        "friction_force_n": friction,
        "rates": np.array(rates),
        "pressure": pressure,
        "wall_pressure": wall_pressure,
        "source": pressure[:, 1] - pressure[:, 0],
        "mass": np.array(masses),
        "gravity": -np.array(masses) * gravity * math.sin(angle),
        "molar_masses": mw,
        "gravity_acceleration": gravity,
        "datum": datum,
    }
