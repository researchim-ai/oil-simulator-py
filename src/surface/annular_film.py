"""Steady vertical, fully wetted laminar liquid annulus and turbulent gas core.

Upward is positive. Liquid momentum is integrated over the actual annulus,
with no slip at the wall and gas shear at the inner surface. Gas pressure
gradient and liquid film thickness are solved together. The interfacial
Fanning factor is Wallis' .005*(1+300*delta/D); this version uses relative
phase velocity, explicitly differing from the superficial-velocity variant.

This constitutive solver is NOT a transient PR pipe, flooding predictor or a
three-phase flow-regime map. No entrainment, partial wetting or fitted data.
"""

import math

import torch

from .liquid import HydraulicConvergenceError


def radial_pressure_mobility(y_minus_one, liquid_fraction):
    """Stokes profile retaining O(delta²) mobility as a real film vanishes."""
    e, alpha = y_minus_one, liquid_fraction
    tail = e.new_tensor(-1 / 10)
    for n in range(9, 2, -1):
        tail = ((-1.0) ** (n + 1)) / n + e * tail
    remainder = torch.where(e.abs() < 0.01, e**3 * tail, torch.log1p(e) - e + e.square() / 2)
    return -alpha * e / 2 - (2 - alpha) * e.square() / 4 + (1 - alpha) * remainder / 2


def film_mobility(liquid_fraction, diameter):
    """Exact Stokes profile averaged by eight-point radial Gauss quadrature.

    mean(u_l) = -pressure_mobility*(dp/dz+rho_l*g)/mu_l
                + shear_mobility*tau_i/mu_l.
    The integral is smooth on the declared thin-film domain (alpha_l<=.3).
    """
    a = liquid_fraction
    s = torch.sqrt(1 - a)
    # 1-s computed without cancellation for thin films.
    gap = a / (1 + s)
    t = a.new_tensor(
        [
            0.0198550717512319,
            0.101666761293187,
            0.237233795041836,
            0.408282678752175,
            0.591717321247825,
            0.762766204958164,
            0.898333238706813,
            0.980144928248768,
        ]
    )
    weights = a.new_tensor(
        [
            0.0506142681451881,
            0.111190517226687,
            0.156853322938944,
            0.181341891689181,
            0.181341891689181,
            0.156853322938944,
            0.111190517226687,
            0.0506142681451881,
        ]
    )
    y_minus_one = -gap[..., None] * (1 - t)
    y = 1 + y_minus_one
    log_y = torch.log1p(y_minus_one)
    radius = diameter / 2
    radial_weight = 2 * y / (1 + s[..., None])
    w = radial_pressure_mobility(y_minus_one, a[..., None])
    pressure = radius**2 * (weights * radial_weight * w).sum(-1)
    shear = -radius * s * (weights * radial_weight * log_y).sum(-1)
    return pressure, shear


def annular_response(alpha, diameter, viscosity):
    """ui=eta*ul+beta*tau; Fl=Si*eta*tau-c*ul from a no-slip Stokes annulus."""
    s = torch.sqrt(1 - alpha)
    gap = alpha / (1 + s)
    log_s = torch.log1p(-gap)
    radius = diameter / 2
    wi = radius**2 * radial_pressure_mobility(-gap, alpha)
    hi = -radius * s * log_s
    w, h = film_mobility(alpha, diameter)
    eta = wi / w
    beta = (hi - eta * h) / viscosity
    c = math.pi * radius**2 * alpha * viscosity / w
    return eta, beta, c, math.pi * diameter * s


def interfacial_conductance(alpha, diameter, mu_l, mu_g, rho_g, ug, ul):
    """Positive gas shear: exact laminar core, then Wallis turbulent factor.

    Coefficients freeze at the driving Reynolds rho_g*|ug-eta*ul|*2a/mu_g;
    the smooth 2000..4000 transition is a declared engineering assumption.
    """
    eta, beta, c, perimeter = annular_response(alpha, diameter, mu_l)
    radius = diameter / 2 * torch.sqrt(1 - alpha)
    drive = ug - eta * ul
    re = rho_g * drive.abs() * 2 * radius / mu_g
    blend = ((re - 2000) / 2000).clamp(0.0, 1.0)
    blend = blend.square() * (3 - 2 * blend)
    delta = diameter / 2 * alpha / (1 + torch.sqrt(1 - alpha))
    linear = (1 - blend) * 4 * mu_g / radius
    quadratic = blend * 0.5 * 0.005 * (1 + 300 * delta / diameter) * rho_g
    a = 1 + beta * linear
    slip = 2 * drive / (a + torch.sqrt(a.square() + 4 * beta * quadratic * drive.abs()))
    friction = linear + quadratic * slip.abs()
    effective = friction / (1 + beta * friction)
    return effective, eta, beta, c, perimeter, re


def countercurrent_film(
    gas_flux_m_s,
    liquid_mass_flow_kg_s,
    gas_density_kg_m3,
    *,
    diameter_m=0.085,
    liquid_density_kg_m3=998.2,
    liquid_viscosity_pa_s=0.001002,
    gravity_m_s2=9.80665,
    shear_velocity="mean",
    gas_viscosity_pa_s=1.81e-5,
):
    """Return the first (thin-film) steady root, rejecting absent/blocked roots.

    Inputs broadcast, remain float64 on their input tensor device. Flow rates
    are positive magnitudes (gas up, liquid down). A 128-point input-only scan
    brackets the first crossing; 56 bisections then converge the force balance.
    No observed pressures enter this calculation.
    """
    if shear_velocity not in ("mean", "interface"):
        raise ValueError("Declare mean or interface shear velocity")
    reference = next(
        (
            v
            for v in (gas_flux_m_s, liquid_mass_flow_kg_s, gas_density_kg_m3)
            if isinstance(v, torch.Tensor)
        ),
        None,
    )
    device = reference.device if reference is not None else "cpu"
    jg, ml, rg = torch.broadcast_tensors(
        *[
            torch.as_tensor(v, dtype=torch.float64, device=device)
            for v in (gas_flux_m_s, liquid_mass_flow_kg_s, gas_density_kg_m3)
        ]
    )
    constants = (
        diameter_m,
        liquid_density_kg_m3,
        liquid_viscosity_pa_s,
        gravity_m_s2,
        gas_viscosity_pa_s,
    )
    if any(not math.isfinite(v) or v <= 0 for v in constants):
        raise ValueError("Positive finite film geometry and properties required")
    if not bool(
        (
            torch.isfinite(jg).all()
            & torch.isfinite(ml).all()
            & torch.isfinite(rg).all()
            & (jg > 0).all()
            & (ml > 0).all()
            & (rg > 0).all()
            & (rg < liquid_density_kg_m3).all()
        ).cpu()
    ):
        raise ValueError("Finite countercurrent gas/liquid inputs required")
    area = math.pi * diameter_m**2 / 4
    jl = -ml / (liquid_density_kg_m3 * area)
    if not bool((ml / (math.pi * diameter_m * liquid_viscosity_pa_s) <= 400).all().cpu()):
        raise ValueError("Liquid film Reynolds number Gamma/mu exceeds laminar domain 400")

    def evaluate(alpha):
        gas_fraction = 1 - alpha
        core_radius = diameter_m / 2 * torch.sqrt(gas_fraction)
        delta = diameter_m / 2 * alpha / (1 + torch.sqrt(gas_fraction))
        ul = jl / alpha
        ug = jg / gas_fraction
        slip = ug - ul
        fanning = 0.005 * (1 + 300 * delta / diameter_m)
        tau = 0.5 * fanning * rg * slip * slip.abs()
        if shear_velocity == "interface":
            effective, eta, _, _, _, _ = interfacial_conductance(
                alpha, diameter_m, liquid_viscosity_pa_s, gas_viscosity_pa_s, rg, ug, ul
            )
            tau = effective * (ug - eta * ul)
        gradient = -rg * gravity_m_s2 - 2 * tau / core_radius
        w, h = film_mobility(alpha, diameter_m)
        profile_mean = (
            -w * (gradient + liquid_density_kg_m3 * gravity_m_s2) + h * tau
        ) / liquid_viscosity_pa_s
        return profile_mean - ul, gradient, tau, delta, ul, ug

    grid = torch.logspace(-5, math.log10(0.3), 128, dtype=jg.dtype, device=device)
    grid = grid.reshape((128,) + (1,) * jg.ndim)
    residual = evaluate(grid)[0]
    crossing = (residual[:-1] >= 0) & (residual[1:] <= 0)
    if not bool(crossing.any(0).all().cpu()):
        raise HydraulicConvergenceError(
            "No steady thin annular film root; no flooding extrapolation"
        )
    index = crossing.to(torch.int64).argmax(0)
    values = grid.reshape(128)
    lo, hi = values[index], values[index + 1]
    for _ in range(56):
        mid = (lo + hi) / 2
        positive = evaluate(mid)[0] >= 0
        lo, hi = torch.where(positive, mid, lo), torch.where(positive, hi, mid)
    alpha = (lo + hi) / 2
    error, gradient, tau, delta, ul, ug = evaluate(alpha)
    if not bool((torch.isfinite(error).all() & (error.abs() < 1e-8).all()).cpu()):
        raise HydraulicConvergenceError("Annular film momentum balance failed")
    perimeter_i = math.pi * diameter_m * torch.sqrt(1 - alpha)
    interface_force = perimeter_i * tau
    wall_force = interface_force - area * alpha * (gradient + liquid_density_kg_m3 * gravity_m_s2)
    return {
        "liquid_holdup": alpha,
        "film_thickness_m": delta,
        "pressure_gradient_pa_m": gradient,
        "interfacial_shear_pa": tau,
        "liquid_velocity_m_s": ul,
        "gas_velocity_m_s": ug,
        "gas_interfacial_force_n_m": -interface_force,
        "liquid_interfacial_force_n_m": interface_force,
        "liquid_wall_force_n_m": -wall_force,
        "liquid_velocity_residual_m_s": error,
    }
