"""Adiabatic Darcy wall source on actual active PR phases, entirely in Torch.

Wetted perimeter is explicitly proportional to phase volume fraction. This is
an engineering constitutive assumption, not a flow-regime or film model.
Viscosities are prescribed per named slot. Colebrook coefficients are frozen
at the pre-source Reynolds number for this first-order source substep.
"""

import math
from dataclasses import replace

import torch

from .liquid import ConservationError, _number
from .pr_homogeneous_uv import recover_homogeneous_phases
from .pr_stock_balances import entropy_allowance


def darcy_factor(reynolds, relative_roughness):
    """64/Re; smooth 2000..4000 transition; implicit Colebrook turbulent root."""
    turbulent_re = reynolds.clamp_min(2000.0)
    z = torch.full_like(reynolds, 7.0)
    for _ in range(12):
        argument = relative_roughness / 3.7 + 2.51 * z / turbulent_re
        residual = z + 2 * torch.log10(argument)
        derivative = 1 + 2 / math.log(10) * (2.51 / turbulent_re) / argument
        z = (z - residual / derivative).clamp(1.0, 100.0)
    turbulent = z.reciprocal().square()
    laminar = 64 / reynolds.clamp_min(torch.finfo(reynolds.dtype).tiny)
    blend = ((reynolds - 2000) / 2000).clamp(0.0, 1.0)
    blend = blend.square() * (3 - 2 * blend)
    result = torch.where(reynolds <= 2000, laminar, (1 - blend) * laminar + blend * turbulent)
    return torch.where(reynolds > 0, result, torch.zeros_like(result))


class PRActivePipeFriction:
    def __init__(self, viscosity_pa_s, roughness_m=0.0):
        if not isinstance(viscosity_pa_s, dict) or not viscosity_pa_s:
            raise ValueError("Named phase viscosities required")
        self.viscosity = {
            n: _number(v, "phase viscosity", positive=True)
            for n, v in sorted(viscosity_pa_s.items())
        }
        self.roughness = _number(roughness_m, "wall roughness", nonnegative=True)

    def to_dict(self):
        return {
            "model": "darcy_volume_wetted_v1",
            "viscosity_pa_s": self.viscosity,
            "roughness_m": self.roughness,
        }

    @classmethod
    def from_dict(cls, data):
        raw = dict(data)
        if raw.get("model") == "laminar_annular_wallis_interface_v1":
            from .pr_active_annular_friction import PRActiveAnnularFriction

            return PRActiveAnnularFriction.from_dict(raw)
        if raw.pop("model", None) != "darcy_volume_wetted_v1":
            raise ValueError("Unknown active pipe friction closure")
        return cls(**raw)

    def apply(self, pipe, state, dt_s):
        dt = _number(dt_s, "friction step", positive=True)
        old = pipe.inspect(state)
        if set(self.viscosity) != set(pipe.names) or self.roughness / pipe.diameter > 0.05:
            raise ValueError("Friction requires every slot viscosity and relative roughness <=.05")
        flat = [(i, n, p) for i, cell in enumerate(state.cells) for n, p in cell]
        f = pipe.fluid
        n = f.tensor([p.component_moles for _, _, p in flat])
        mass = (n * f.molar_masses).sum(-1)
        volume = f.tensor([p.volume_m3 for _, _, p in flat])
        momentum = f.tensor([p.momentum_kg_m_s for _, _, p in flat])
        velocity = momentum / mass
        rho = mass / volume
        mu = f.tensor([self.viscosity[name] for _, name, _ in flat])
        reynolds = rho * velocity.abs() * pipe.diameter / mu
        factor = darcy_factor(reynolds, self.roughness / pipe.diameter)
        coefficient = factor / (2 * pipe.diameter)
        turbulent = 2 * velocity / (1 + torch.sqrt(1 + 4 * dt * coefficient * velocity.abs()))
        laminar = velocity / (1 + dt * 32 * mu / (rho * pipe.diameter**2))
        new_velocity = torch.where(reynolds <= 2000, laminar, turbulent)
        new_momentum = mass * new_velocity
        phases = recover_homogeneous_phases(
            f,
            n,
            volume,
            new_momentum,
            [p.total_energy_j for _, _, p in flat],
            [pipe.slots[pipe.names.index(name)]["root"] for _, name, _ in flat],
            [p.temperature_k for _, _, p in flat],
        )
        cells = [[] for _ in state.cells]
        for (i, name, _), p in zip(flat, phases):
            cells[i].append((name, p))
        new = replace(state, cells=tuple(tuple(c) for c in cells))
        current = pipe.inspect(new)
        entropy = current["entropy"] - old["entropy"]
        if entropy < -entropy_allowance(current["entropy"], old["entropy"]):
            raise ConservationError("Adiabatic wall friction decreased entropy")
        impulse = float((new_momentum - momentum).sum().cpu())
        heat = 0.5 * mass * (velocity - new_velocity) * (velocity + new_velocity)
        if not bool((torch.isfinite(heat).all() & (heat >= 0).all()).cpu()):
            raise ConservationError("Wall friction produced negative dissipation")
        return new, {
            "state": new.to_dict(),
            "dt_s": dt,
            "wall_impulse_kg_m_s": impulse,
            "entropy_production_j_k": entropy,
            "dissipation_j": heat.cpu().tolist(),
            "reynolds_numbers": reynolds.cpu().tolist(),
            "darcy_factors": factor.cpu().tolist(),
            "active_locations": [(i, name) for i, name, _ in flat],
            "conservation_residuals": pipe.ledger(state, new, impulse),
        }
