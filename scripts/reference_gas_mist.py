"""Independent scalar constraints for dilute gas-carried liquid droplets."""

import math

import numpy as np


def velocities(
    c,
    alpha,
    rho,
    x,
    bulk,
    pipe_diameter,
    axial,
    gravity,
    pressure,
    temperature,
    molar_masses,
    phase_roles=None,
):
    chosen = 1 if c["dispersed_liquid"] == "oil" else 2
    markers = [
        c[k] for k in ("gas_component_index", "oil_component_index", "water_component_index")
    ]
    scores = np.array(x)[:, markers]
    roles = scores.argmax(1) if phase_roles is None else np.array(phase_roles)
    if 0 not in roles or any(r not in (0, chosen) for r in roles) or len(set(roles)) != len(roles):
        raise ValueError("Gas mist requires a gas carrier and at most one dispersed liquid")
    if phase_roles is None and np.any(scores.max(1) < c["minimum_marker_fraction"]):
        raise ValueError("Ambiguous gas mist markers")
    ig = int(np.flatnonzero(roles == 0)[0])
    if not c["minimum_gas_volume_fraction"] <= alpha[ig] <= c["maximum_gas_volume_fraction"]:
        raise ValueError("Gas mist outside dilute carrier domain")
    if len(alpha) == 1:
        return np.array([bulk])
    il = int(np.flatnonzero(roles == chosen)[0])
    d, mu = c["droplet_diameter_m"], c["gas_viscosity_pa_s"]
    mud = c["oil_viscosity_pa_s"] if chosen == 1 else c["water_viscosity_pa_s"]
    contrast = rho[il] - rho[ig]
    mobility = d * d / (18 * mu)
    if c["interface"] == "clean":
        mobility *= 3 * (mu + mud) / (2 * mu + 3 * mud)
    settling = -contrast * axial * mobility
    re = rho[ig] * abs(contrast) * gravity * mobility * d / mu
    eo = abs(contrast) * gravity * d * d / c["surface_tension_n_m"]
    mg = float(np.asarray(x)[ig] @ molar_masses)
    mean_free_path = mu / rho[ig] * math.sqrt(math.pi * mg / (2 * 8.31446261815324 * temperature))
    if (
        contrast <= 0
        or rho[ig] / rho[il] > c["maximum_gas_liquid_density_ratio"]
        or alpha[il] > c["maximum_dispersed_liquid_fraction"]
        or re > c["maximum_reynolds_number"]
        or eo > c["maximum_eotvos_number"]
        or d / pipe_diameter > c["maximum_confinement_ratio"]
        or 4 * c["surface_tension_n_m"] / d > c["maximum_capillary_pressure_fraction"] * pressure
        or mean_free_path / d > c["maximum_knudsen_number"]
    ):
        raise ValueError("Gas mist outside Stokes/capillary/continuum envelope")
    matrix = np.array([alpha * rho, np.eye(len(alpha))[il] - np.eye(len(alpha))[ig]])
    return np.linalg.solve(matrix, [float(alpha @ rho) * bulk, settling])
