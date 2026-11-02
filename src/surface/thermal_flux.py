"""Entropy-consistent two-point fluxes for the frozen homogeneous pipe.

Entropy variables follow S = C ln T - Ug R ln(Ug/alpha). The central
flux satisfies Tadmor's jump identity; positive diffusion acts on conservative
variables and on temperature. No reference solution enters this module.
"""

import numpy as np


def logarithmic_mean(left, right):
    a, b = np.broadcast_arrays(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Logarithmic mean needs positive finite endpoints")
    mean = 0.5 * a + 0.5 * b
    d = (0.5 * b - 0.5 * a) / mean
    small = abs(d) < 1e-4
    strong = abs(d) > 0.9
    ratio = np.divide(
        d, np.arctanh(np.clip(d, -0.9, 0.9)), out=np.ones_like(d), where=~small & ~strong
    )
    d2 = d * d
    ratio = np.where(small, 1 - d2 / 3 - 4 * d2 * d2 / 45, ratio)
    result = mean * ratio
    direct = np.divide(b - a, np.log(b) - np.log(a), out=np.zeros_like(mean), where=strong)
    return np.where(strong, direct, result)


def entropy_density(pipe, u):
    _, alpha, _, temperature, capacity = pipe._decode(u, validate=False)
    return capacity * np.log(temperature) - u[..., 1] * pipe.gas_constant_j_kg_k * np.log(
        u[..., 1] / alpha
    )


def entropy_variables(pipe, u):
    _, alpha, velocity, temperature, _ = pipe._decode(u, validate=False)
    theta = 1 / temperature
    gas_density = u[..., 1] / alpha
    kinetic = 0.5 * velocity * velocity * theta
    liquid = (
        pipe.liquid_cv_j_kg_k * (1 + np.log(theta))
        - kinetic
        + pipe.gas_constant_j_kg_k * gas_density / pipe.liquid_density_kg_m3
    )
    gas = (
        pipe.gas_cv_j_kg_k * (1 + np.log(theta))
        - kinetic
        + pipe.gas_constant_j_kg_k * (1 + np.log(gas_density))
    )
    return np.stack([liquid, gas, velocity * theta, -theta], axis=-1)


def entropy_potential(pipe, u):
    _, alpha, velocity, _, _ = pipe._decode(u, validate=False)
    return pipe.gas_constant_j_kg_k * u[..., 1] / alpha * velocity


def entropy_conservative_flux(pipe, left, right):
    l, r = np.broadcast_arrays(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    _, al, vl, tl, _ = pipe._decode(l, validate=False)
    _, ar, vr, tr, _ = pipe._decode(r, validate=False)
    gl, gr = l[..., 1] / al, r[..., 1] / ar
    gas_mean = logarithmic_mean(gl, gr)
    theta_mean = logarithmic_mean(1 / tl, 1 / tr)
    # This symmetric mean preserves a uniform phase mass ratio exactly.
    ratio = 0.5 * (l[..., 0] / l[..., 1] + r[..., 0] / r[..., 1])
    alpha = pipe.liquid_density_kg_m3 / (pipe.liquid_density_kg_m3 + ratio * gas_mean)
    velocity = 0.5 * (vl + vr)
    pressure = pipe.gas_constant_j_kg_k * 0.5 * (gl + gr) / (0.5 / tl + 0.5 / tr)
    liquid_flux = pipe.liquid_density_kg_m3 * (1 - alpha) * velocity
    gas_flux = gas_mean * alpha * velocity
    mass_flux = liquid_flux + gas_flux
    momentum_flux = velocity * mass_flux + pressure
    energy_flux = (
        (pipe.liquid_cv_j_kg_k * liquid_flux + pipe.gas_cv_j_kg_k * gas_flux) / theta_mean
        - 0.25 * (vl * vl + vr * vr) * mass_flux
        + velocity * momentum_flux
    )
    return np.stack([liquid_flux, gas_flux, momentum_flux, energy_flux], axis=-1)


def entropy_stable_flux(pipe, left, right, dissipation=1.0):
    l, r = np.broadcast_arrays(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    flux = entropy_conservative_flux(pipe, l, r)
    factor = np.asarray(dissipation, dtype=float)
    if not np.isfinite(factor).all() or np.any(factor < 0) or np.any(factor > 1):
        raise ValueError("Entropy dissipation multiplier must be in [0,1]")
    speed = np.maximum(pipe._wave_speed(l), pipe._wave_speed(r)) * factor
    flux -= 0.5 * speed[..., None] * (r - l)
    _, _, _, tl, cl = pipe._decode(l, validate=False)
    _, _, _, tr, cr = pipe._decode(r, validate=False)
    cpl = cl + l[..., 1] * pipe.gas_constant_j_kg_k
    cpr = cr + r[..., 1] * pipe.gas_constant_j_kg_k
    capacity = 2 * cpl * cpr / (cpl + cpr)
    flux[..., 3] -= 0.5 * pipe.thermal_entropy_dissipation * speed * capacity * (tr - tl)
    return flux


def physical_entropy_flux(pipe, left, right, flux):
    return 0.5 * (entropy_potential(pipe, left) + entropy_potential(pipe, right)) - 0.5 * np.sum(
        (entropy_variables(pipe, left) + entropy_variables(pipe, right)) * flux, axis=-1
    )


def entropy_viscosity_sensor(pipe, u, left, right):
    """Dimensionless four-cell curvature in entropy variables and void fraction.

    Evaluate ONCE from the accepted state before each substep. Keeping these
    nonnegative coefficients fixed during Newton preserves the entropy sign
    and the three-cell nonlinear stencil; differentiating this nonsmooth
    sensor inside Newton requires a different Jacobian and globalization.
    """
    l = np.array(u[0] if left is None else left, copy=True)
    r = np.array(u[-1] if right is None else right, copy=True)
    if left is None:
        l[2] *= -1
    if right is None:
        r[2] *= -1
    extended = np.concatenate([l[None, :], u, r[None, :]])
    alpha = 1 - extended[:, 0] / pipe.liquid_density_kg_m3
    q = np.column_stack([entropy_variables(pipe, extended), alpha])
    jumps = np.diff(q, axis=0)
    before, middle, after = jumps[:-2], jumps[1:-1], jumps[2:]
    numerator = abs(middle - before) + abs(after - middle)
    denominator = abs(before) + 2 * abs(middle) + abs(after)
    local_scale = np.maximum.reduce([abs(q[:-3]), abs(q[1:-2]), abs(q[2:-1]), abs(q[3:])])
    floor = 64 * np.finfo(float).eps * np.maximum(1.0, local_scale)
    return np.clip(np.max(numerator / (denominator + floor), axis=-1), 0, 1)
