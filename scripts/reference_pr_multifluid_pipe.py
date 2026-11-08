"""NumPy PR packets and Gaussian path integration, no runtime flux calls."""

import math

import numpy as np
from check_pr_equilibrium_transport import reference_sound
from reference_pr_interphase_relaxation import decode


def packets(model, state, fluid):
    roots = model["sources"][0]["phase_roots"]
    volumes = model.get("volumes_m3")
    if volumes is None:
        area = math.pi * model["diameter_m"] ** 2 / 4
        volumes = [area * length for length in model["lengths_m"]]
    result = []
    for cell, volume in zip(state["cells"], volumes, strict=True):
        data = decode(fluid, cell, roots)
        alpha = np.array([v["volume"] / volume for v in data])
        if abs(alpha.sum() - 1) > 2e-9 or np.min(alpha) < model["minimum_phase_fraction"]:
            raise ValueError("Independent pipe volume/fraction domain rejected")
        phase_state = []
        phase_flux = []
        sounds = []
        for d, root in zip(data, roots, strict=True):
            n, m, u, p = d["n"], d["mass"], d["velocity"], d["p"]
            a = d["volume"] / volume
            energy = d["internal"] + 0.5 * m * u * u
            phase_state.append(np.r_[n / volume, m * u / volume, energy / volume])
            phase_flux.append(
                np.r_[n * u / volume, m * u * u / volume + a * p, (energy / volume + a * p) * u]
            )
            sounds.append(reference_sound(fluid, p, d["t"], n / n.sum(), root))
        velocity = np.array([d["velocity"] for d in data])
        sounds = np.array(sounds)
        if np.any(np.abs(velocity) + sounds > model["wave_speed_bound_m_s"] * (1 + 1e-6)):
            raise ValueError("Independent pipe acoustic envelope rejected")
        if np.any(np.abs(np.abs(velocity[0] - velocity[1:]) - sounds[1:]) <= 1e-6):
            raise ValueError("Independent sonic resonance screen rejected")
        result.append(
            {
                "conserved": np.array(phase_state),
                "flux": np.array(phase_flux),
                "alpha": alpha,
                "pressure": np.array([d["p"] for d in data]),
                "velocity": velocity,
                "sound": sounds,
                "entropy": sum(d["entropy"] for d in data),
            }
        )
    return result


def path(left, right, speed):
    difference = right["alpha"] - left["alpha"]
    # Three-point Gaussian quadrature along the declared primitive-variable
    # path independently integrates p_k*u_0 rather than multiplying means.
    nodes, weights = np.polynomial.legendre.leggauss(3)
    pmean = 0.0
    umean = 0.0
    pumean = 0.0
    for node, weight in zip((nodes + 1) / 2, weights / 2):
        p = left["pressure"] + node * (right["pressure"] - left["pressure"])
        u = left["velocity"][0] + node * (right["velocity"][0] - left["velocity"][0])
        pmean = pmean + weight * p
        umean = umean + weight * u
        pumean = pumean + weight * p * u
    integral = np.zeros_like(left["conserved"])
    integral[1:, -2] = -pmean[1:] * difference[1:]
    integral[0, -2] = -sum(integral[1:, -2])
    integral[1:, -1] = -pumean[1:] * difference[1:]
    integral[0, -1] = -sum(integral[1:, -1])
    jump = right["flux"] - left["flux"] + integral
    dissipation = speed * (right["conserved"] - left["conserved"])
    alpha_jump = umean * difference[1:]
    minus = np.r_[0.5 * (jump - dissipation).ravel(), 0.5 * (alpha_jump - speed * difference[1:])]
    plus = np.r_[0.5 * (jump + dissipation).ravel(), 0.5 * (alpha_jump + speed * difference[1:])]
    flux = 0.5 * (left["flux"] + right["flux"]) - 0.5 * dissipation
    return minus, plus, {"flux": flux, "phase_path": integral, "alpha_path": alpha_jump}


def reconstruct(model, state, fluid):
    data = packets(model, state, fluid)
    n = len(data)
    nph = len(data[0]["alpha"])
    nc = len(fluid.names)
    width = nph * (nc + 2) + nph - 1
    incoming = np.zeros((n, width))
    faces = []
    wall = []
    for i in range(n if model["boundary_mode"] == "periodic" else n - 1):
        j = (i + 1) % n
        minus, plus, face = path(data[i], data[j], model["wave_speed_bound_m_s"])
        incoming[i] += minus
        incoming[j] += plus
        faces.append(face)
    impulse_rate = 0.0
    if model["boundary_mode"] == "walls":
        for i, side in ((0, -1), (n - 1, 1)):
            local = data[i]
            ghost = {
                **local,
                "conserved": local["conserved"].copy(),
                "flux": local["flux"].copy(),
                "velocity": -local["velocity"],
            }
            ghost["conserved"][:, -2] *= -1
            ghost["flux"][:, :-2] *= -1
            ghost["flux"][:, -1] *= -1
            left, right = (ghost, local) if side == -1 else (local, ghost)
            bound = np.abs(local["velocity"]) + local["sound"]
            jump = right["conserved"] - left["conserved"]
            ff = 0.5 * (left["flux"] + right["flux"]) - 0.5 * bound[:, None] * jump
            if np.any(ff[:, -2] <= 0):
                raise ValueError("Nonpositive rigid-wall traction")
            term = (right["flux"] - ff) if side == -1 else (ff - left["flux"])
            incoming[i, : nph * (nc + 2)] += term.ravel()
            impulse_rate += (-side) * sum(ff[:, -2])
            wall.append(ff)
    area = math.pi * model["diameter_m"] ** 2 / 4
    return {
        "packets": data,
        "incoming": incoming,
        "phase_rate": area * incoming[:, : nph * (nc + 2)].reshape(n, nph, nc + 2),
        "alpha_rate": incoming[:, nph * (nc + 2) :] / np.array(model["lengths_m"])[:, None],
        "faces": faces,
        "wall_fluxes": wall,
        "wall_force_n": area * impulse_rate,
    }
