"""Differentiable quasi-steady Darcy loss, SI; prescribed effective viscosity.

Laminar branch is exactly Poiseuille. Above Re=4000 uses Haaland (1983),
DOI 10.1115/1.3240948. A C1 engineering blend spans 2000..4000; no claim
of a universal transitional/two-phase friction correlation is implied.
"""

import torch


def darcy_pressure_loss(velocity, density, length_m, diameter_m, viscosity_pa_s, roughness_m):
    reynolds = density * velocity.abs() * diameter_m / viscosity_pa_s
    laminar = 32 * viscosity_pa_s * length_m / diameter_m**2 * velocity
    re_turb = reynolds.clamp_min(2000.0)
    inv_sqrt = -1.8 * torch.log10((roughness_m / (3.7 * diameter_m)) ** 1.11 + 6.9 / re_turb)
    friction = inv_sqrt.reciprocal().square()
    turbulent = friction * length_m / diameter_m * density * velocity * velocity.abs() / 2
    blend = ((reynolds - 2000) / 2000).clamp(0, 1)
    blend = blend.square() * (3 - 2 * blend)
    return (1 - blend) * laminar + blend * turbulent, reynolds
