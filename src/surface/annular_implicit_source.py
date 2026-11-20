"""Final-state viscous resistance solves for fixed annular inventories.

All operations are batched Torch on the input device. Geometry/temperature
remain fixed during this dissipative source substep. This changes temporal
integration, not the constitutive correlation or its experimental status.
"""

import torch

from .annular_film import annular_response, interfacial_conductance
from .liquid import HydraulicConvergenceError
from .pr_active_pipe_friction import darcy_factor


def implicit_annular_velocities(
    alpha, diameter, mu_l, mu_g, rho_g, ug0, ul0, mass_g, mass_l, lengths, dt, roughness
):
    """Eliminate both momenta, then solve a bounded scalar final drive.

    d = ug-eta*ul, tau=e(d)*d, d+A*tau=D0. The positive root lies between
    zero and |D0|; reversing both velocities reverses the solution. The
    wall contribution remains in the liquid Schur denominator.
    """
    eta, beta, c, perimeter = annular_response(alpha, diameter, mu_l)
    response = (eta, beta, c, perimeter)
    wall = lengths * c
    denominator = mass_l + dt * wall
    drive0 = ug0 - eta * mass_l / denominator * ul0
    coupling = dt * lengths * perimeter * (1 / mass_g + eta.square() / denominator)
    lo, hi = torch.zeros_like(drive0), drive0.abs()
    mid = hi
    for _ in range(16):
        effective, _, _, _, _, _, shear_slope = interfacial_conductance(
            alpha,
            diameter,
            mu_l,
            mu_g,
            rho_g,
            mid,
            torch.zeros_like(mid),
            dry_consistent=True,
            roughness_m=roughness,
            with_slope=True,
            _response=response,
        )
        residual = mid + coupling * effective * mid - drive0.abs()
        lo = torch.where(residual < 0, mid, lo)
        hi = torch.where(residual < 0, hi, mid)
        proposal = mid - residual / (1 + coupling * shear_slope)
        safe = torch.isfinite(proposal) & (proposal > lo) & (proposal < hi)
        proposal = torch.where(safe, proposal, (lo + hi) / 2)
        mid = torch.where(residual.abs() <= 1e-13 * (1 + drive0.abs()), mid, proposal)
    drive = torch.copysign(mid, drive0)
    effective, _, _, _, _, re = interfacial_conductance(
        alpha,
        diameter,
        mu_l,
        mu_g,
        rho_g,
        drive,
        torch.zeros_like(drive),
        dry_consistent=True,
        roughness_m=roughness,
        _response=response,
    )
    # Stable Schur reconstruction avoids subtracting a stiff impulse from ug0.
    k = lengths * perimeter * effective
    share = dt * k / (mass_g + dt * k)
    ul = (mass_l * ul0 + eta * mass_g * share * ug0) / (denominator + eta.square() * mass_g * share)
    ug = mass_g / (mass_g + dt * k) * ug0 + share * eta * ul
    error = (ug - eta * ul) - drive
    tolerance = 1e-11 * (1 + ug0.abs() + (eta * ul0).abs())
    if not bool((torch.isfinite(error) & (error.abs() <= tolerance)).all().cpu()):
        raise HydraulicConvergenceError("Implicit annular final-state resistance failed")
    return ug, ul, effective, eta, beta, c, perimeter, re


def implicit_darcy_velocity(u0, density, viscosity, diameter, roughness, dt):
    """Backward Euler with Darcy factor evaluated at final velocity."""
    lo, hi = torch.zeros_like(u0), u0.abs()
    for _ in range(56):
        mid = (lo + hi) / 2
        reynolds = density * mid * diameter / viscosity
        turbulent = darcy_factor(reynolds, roughness / diameter) * mid / (2 * diameter)
        drag = torch.where(reynolds <= 2000, 32 * viscosity / (density * diameter**2), turbulent)
        residual = mid * (1 + dt * drag) - u0.abs()
        lo = torch.where(residual < 0, mid, lo)
        hi = torch.where(residual < 0, hi, mid)
    un = torch.copysign((lo + hi) / 2, u0)
    re = density * un.abs() * diameter / viscosity
    drag = torch.where(
        re <= 2000,
        32 * viscosity / (density * diameter**2),
        darcy_factor(re, roughness / diameter) * un.abs() / (2 * diameter),
    )
    residual = un * (1 + dt * drag) - u0
    if not bool(
        (torch.isfinite(residual) & (residual.abs() <= 1e-11 * (1 + u0.abs()))).all().cpu()
    ):
        raise HydraulicConvergenceError("Implicit Darcy final-state resistance failed")
    return un
