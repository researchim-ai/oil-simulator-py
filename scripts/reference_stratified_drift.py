"""Independent scalar stratified audit: math/NumPy/SciPy only, no runtime imports.

Public velocities uses +g*dz/L uphill, exactly like the core closure API.
parameters uses the same public convention, unlike runtime parameters (actual gx).
"""

import math

import numpy as np
from scipy.optimize import brentq


def parameters(closure, alpha, rho, bulk, diameter, axial_g, gravity, phase_viscosities_pa_s=None):
    """Return scalar audit diagnostics in input phase order; axial_g is uphill."""
    alpha, rho = np.asarray(alpha, dtype=float), np.asarray(rho, dtype=float)
    if alpha.shape != rho.shape or alpha.ndim != 1:
        raise ValueError("Matching phase vectors required")
    if len(alpha) == 1:
        return {"velocities": np.array([bulk]), "valid": True}
    if len(alpha) != 2:
        raise ValueError("Exactly two gas/liquid phases required")
    if not (
        np.isfinite(alpha).all()
        and np.isfinite(rho).all()
        and (alpha > 0).all()
        and (rho > 0).all()
    ):
        raise ValueError("Positive finite stocks and densities required")
    if not all(math.isfinite(x) for x in (bulk, diameter, axial_g, gravity)):
        raise ValueError("Finite geometry and velocity required")
    if diameter <= 0 or gravity <= 0:
        raise ValueError("Positive diameter/gravity required")
    order = np.argsort(rho)
    ig, il = order
    agf = alpha[ig] / alpha.sum()
    rg, rl = rho[ig], rho[il]
    if phase_viscosities_pa_s is None:
        mug = closure["gas_dynamic_viscosity_pa_s"]
        mul = closure["liquid_dynamic_viscosity_pa_s"]
    else:
        mu = np.asarray(phase_viscosities_pa_s, dtype=float)
        if mu.shape != (2,):
            raise ValueError("Two dynamic viscosities required")
        mug, mul = mu[ig], mu[il]
    if not all(math.isfinite(x) and x > 0 for x in (mug, mul)):
        raise ValueError("Positive finite dynamic viscosities required")
    if not 0 < agf < 1:
        raise ValueError("Interior holdup required")
    theta = brentq(
        lambda t: (t - math.sin(t) * math.cos(t)) / math.pi - (1 - agf), 0, math.pi, xtol=1e-14
    )
    area = math.pi * diameter**2 / 4
    al, ag = area * (1 - agf), area * agf
    sl, sg, si = diameter * theta, diameter * (math.pi - theta), diameter * math.sin(theta)
    dl, dg = 4 * al / sl, 4 * ag / (sg + si)
    rough = closure["wall_roughness_m"]
    fi = closure["interfacial_friction_factor"]
    mg = agf * rg / (agf * rg + (1 - agf) * rl)
    gx = -axial_g

    def wall(v, density, mu, dh):
        re = max(density * abs(v) * dh / mu, 2300)
        fd = (-1.8 * math.log10((rough / (3.7 * dh)) ** 1.11 + 6.9 / re)) ** -2
        return v * max(8 * mu / dh, density * fd * abs(v) / 8)

    def balance(s):
        vg, vl = bulk + (1 - mg) * s, bulk - mg * s
        tg, tl = wall(vg, rg, mug, dg), wall(vl, rl, mul, dl)
        ti = 8 * mug / dg * s + fi * rg * s * abs(s) / 2
        pg = rg * gx - (tg * sg + ti * si) / ag
        pl = rl * gx - (tl * sl - ti * si) / al
        return pl - pg, pg, pl, vg, vl

    limit = closure.get("maximum_slip_m_s", 100.0)
    if not balance(-limit)[0] <= 0 <= balance(limit)[0]:
        raise ValueError("Reference force root outside bounded slip interval")
    slip = brentq(lambda s: balance(s)[0], -limit, limit, xtol=1e-13, rtol=1e-14)
    residual, pg, pl, vg, vl = balance(slip)
    result = np.empty(2)
    result[ig], result[il] = vg, vl
    margin = (rl - rg) * math.sqrt(max(0, gravity**2 - gx**2)) * (ag / rg + al / rl) / si - slip**2
    valid = (
        closure.get("minimum_gas_volume_fraction", 0.02)
        <= agf
        <= closure.get("maximum_gas_volume_fraction", 0.98)
        and rg / rl <= closure.get("maximum_gas_liquid_density_ratio", 0.2)
        and abs(axial_g)
        <= gravity * math.sin(math.radians(closure.get("maximum_inclination_deg", 5.0)))
        and rough / min(dl, dg) <= 0.05
        and max(abs(vg), abs(vl)) <= closure.get("maximum_phase_speed_m_s", 100.0)
        and margin > 0
    )
    return {
        "velocities": result,
        "valid": bool(valid),
        "slip_m_s": slip,
        "pressure_gradient_pa_m": (pg + pl) / 2,
        "force_residual_pa_m": residual,
        "kh_margin_m2_s2": margin,
        "long_wave_stable": margin > 0,
    }


def velocities(closure, alpha, rho, bulk, diameter, axial_g, gravity, phase_viscosities_pa_s=None):
    """Audit phase velocities; reject states outside the declared stratified branch."""
    r = parameters(closure, alpha, rho, bulk, diameter, axial_g, gravity, phase_viscosities_pa_s)
    if not r["valid"]:
        raise ValueError("Reference stratified validity envelope rejected")
    return r["velocities"]


def wall_force(node, data, model):
    """Total stationary-wall reaction; interfacial shear cancels internally."""
    law = node.get("drift_flux") or {}
    if law.get("model") != "stratified_gas_liquid":
        return 0.0
    if len(data["pn"]) == 1:
        from check_stratified_pr_network import reference_darcy

        mu = (
            law["gas_dynamic_viscosity_pa_s"]
            if data["phase_z"][0] > 0.5
            else law["liquid_dynamic_viscosity_pa_s"]
        )
        return (
            -node["volume_m3"]
            * reference_darcy(
                data["u"], data["rho"], 1.0, node["diameter_m"], mu, law["wall_roughness_m"]
            )[0]
        )
    axial = model["gravity_m_s2"] * node["elevation_change_m"] / node["length_m"]
    diagnostics = parameters(
        law,
        data["pn"] * data["vm"],
        data["rho_phase"],
        data["u"],
        node["diameter_m"],
        axial,
        model["gravity_m_s2"],
    )
    if not diagnostics["valid"]:
        raise ValueError("Stratified wall force outside envelope")
    return node["volume_m3"] * (data["rho"] * axial + diagnostics["pressure_gradient_pa_m"])
