"""Shared local Stokes mobilities of an isothermal planar wall film.

This is the constitutive limit used by ThinFilmDynamics, not a wave, entrance
length, turbulent film or gas-shear correlation. Coordinates follow the wall.
"""

import torch


def mobilities(thickness, viscosity, slip_length=0.0):
    """Pressure-to-flux, shear-to-flux and shear-to-interface-velocity response.

    Callers validate geometry and material inputs. Tensor operations stay on the
    input device and preserve autograd; no host synchronization in this kernel.
    """
    h = thickness
    return (
        h**3 / (3 * viscosity) + slip_length * h.square() / viscosity,
        h.square() / (2 * viscosity) + slip_length * h / viscosity,
        (h + slip_length) / viscosity,
    )


def resistance_response(thickness, viscosity, slip_length=0.0):
    """Eliminate the local Stokes profile in favor of its mean velocity.

    ui=eta*mean_u+beta*tau, net shear/wall force=eta*tau-c*mean_u.
    These stable forms avoid subtracting nearly equal slip-dominated
    mobilities. Dissipation per wall area is c*mean_u**2+beta*tau**2.
    """
    h, slip = thickness, slip_length
    denominator = h + 3 * slip
    eta = 1 + h / (2 * denominator)
    beta = h / (4 * viscosity) * (h + 4 * slip) / denominator
    resistance = viscosity / (h / 3 + slip)
    return eta, beta, resistance


def shear_free_gravity_thickness(mass_flow_per_width, density, viscosity, gravity=9.80665):
    """Uniform no-slip Nusselt reference from actual mass flux per wetted width.

    No upper Reynolds validity is inferred from this algebraic limit. A caller
    must distinguish using it as a reference from validating a finite-Re wave
    model. Positive gravity is along the descending film, no imposed gas shear.
    """
    gamma = torch.as_tensor(mass_flow_per_width, dtype=torch.float64)
    rho, mu, g = (
        torch.as_tensor(v, dtype=gamma.dtype, device=gamma.device)
        for v in (density, viscosity, gravity)
    )
    if not bool(
        (
            torch.isfinite(gamma)
            & (gamma >= 0)
            & torch.isfinite(rho)
            & (rho > 0)
            & torch.isfinite(mu)
            & (mu > 0)
            & torch.isfinite(g)
            & (g > 0)
        ).all()
    ):
        raise ValueError(
            "Finite nonnegative film mass flux and positive material/gravity inputs required"
        )
    return (3 * mu * gamma / (rho.square() * g)).pow(1 / 3)
