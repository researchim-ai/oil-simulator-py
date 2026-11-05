"""Independent NumPy constraint solve for dilute liquid slip; no runtime calls."""

import numpy as np


def velocities(
    closure, alpha, rho, composition, bulk, diameter, axial_g, gravity, pressure, phase_roles=None
):
    if len(alpha) == 1:
        return np.array([bulk])
    markers = [
        closure[key]
        for key in ("gas_component_index", "oil_component_index", "water_component_index")
    ]
    scores = np.asarray(composition)[:, markers]
    roles = scores.argmax(1) if phase_roles is None else np.asarray(phase_roles)
    marker_valid = (
        (scores.max(1) >= closure["minimum_marker_fraction"])
        if phase_roles is None
        else ((np.max(scores[:, 1:], axis=1) >= closure["minimum_marker_fraction"]) | (roles == 0))
    )
    if len(set(roles)) != len(roles) or not np.all(marker_valid):
        raise ValueError("Ambiguous oil/water/gas phase markers")
    indices = {int(role): i for i, role in enumerate(roles)}
    if len(alpha) not in (2, 3):
        raise ValueError("Expected two or three distinct phase roles")
    slip = 0.0
    if 1 in indices and 2 in indices:
        io, iw = indices[1], indices[2]
        contrast = rho[iw] - rho[io]
        oil_drop = closure["dispersed_liquid"] == "oil"
        mu = closure["water_viscosity_pa_s"] if oil_drop else closure["oil_viscosity_pa_s"]
        drop_mu = closure["oil_viscosity_pa_s"] if oil_drop else closure["water_viscosity_pa_s"]
        carrier_density = rho[iw] if oil_drop else rho[io]
        d = closure["droplet_diameter_m"]
        if (
            4 * closure["surface_tension_n_m"] / d
            > closure["maximum_capillary_pressure_fraction"] * pressure
        ):
            raise ValueError("Common-pressure approximation exceeds capillary envelope")
        mobility = d * d / (18 * mu)
        if closure["interface"] == "clean":
            ratio = drop_mu / mu
            mobility *= 3 * (ratio + 1) / (3 * ratio + 2)
        slip = contrast * axial_g * mobility
        terminal = abs(contrast) * gravity * mobility
        reynolds = carrier_density * terminal * d / mu
        eotvos = abs(contrast) * gravity * d * d / closure["surface_tension_n_m"]
        if (
            min(rho[io], rho[iw]) <= 0
            or alpha[io if oil_drop else iw] / (alpha[io] + alpha[iw])
            > closure["maximum_dispersed_liquid_fraction"]
            or reynolds > closure["maximum_reynolds_number"]
            or eotvos > closure["maximum_eotvos_number"]
            or d / diameter > closure["maximum_confinement_ratio"]
        ):
            raise ValueError("Dilute creeping-droplet envelope exceeded")
    rows = [alpha * rho]
    rhs = [float((alpha * rho).sum()) * bulk]
    if 0 in indices:
        ig = indices[0]
        ag = alpha[ig]
        liquid = roles != 0
        rl = (alpha[liquid] * rho[liquid]).sum() / alpha[liquid].sum()
        if not (
            closure["minimum_gas_volume_fraction"] <= ag <= closure["maximum_gas_volume_fraction"]
            and rho[ig] / rl <= closure["maximum_gas_liquid_density_ratio"]
        ):
            raise ValueError("Gas drift envelope exceeded")
        row = -closure["distribution_parameter"] * alpha.copy()
        row[ig] += 1
        rows.append(row)
        rhs.append(closure["drift_velocity_m_s"])
    if 1 in indices and 2 in indices:
        row = np.zeros(len(alpha))
        row[indices[1]], row[indices[2]] = 1, -1
        rows.append(row)
        rhs.append(slip)
    result = np.linalg.solve(np.asarray(rows), np.asarray(rhs))
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite phase velocity")
    return result
