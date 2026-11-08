"""Canonical conservative UV target refinement on an already selected phase set.

Global stability/topology belongs to the preceding audited equilibrium flash.
This small joint Torch solve refines P/T/chemical allocation, then reconciles
roundoff in extensive primary stocks before homogeneous branch recovery.
"""

import math

import torch

from .liquid import HydraulicConvergenceError
from .pr_homogeneous_uv import recover_homogeneous_phases
from .pr_phase_equilibration import PRPhaseEquilibriumState
from .torch_linear import TorchNewton


def canonical_equilibrium_target(fluid, source, target):
    nc = len(fluid.names)
    count = len(target.phases)
    totals = [math.fsum(p.component_moles[j] for p in source.phases) for j in range(nc)]
    volume = math.fsum(p.volume_m3 for p in source.phases)
    energy = math.fsum(p.total_energy_j for p in source.phases)
    momentum = math.fsum(p.momentum_kg_m_s for p in source.phases)
    mass = math.fsum(v * c.molar_mass_kg_mol for v, c in zip(totals, fluid.components))
    velocity = momentum / mass
    internal = energy - 0.5 * momentum * velocity
    roots = target.phase_roots
    if count == 1:
        phases = recover_homogeneous_phases(
            fluid, [totals], [volume], [momentum], [energy], roots, [target.phases[0].temperature_k]
        )
        return PRPhaseEquilibriumState(phases, roots, target.model_id, target.time_s)
    n = fluid.tensor(totals)
    active = (n > 0).nonzero().flatten()
    initial_n = fluid.tensor([p.component_moles for p in target.phases])[:, active]
    logits = (initial_n[:-1].clamp_min(1e-300) / initial_n[-1:].clamp_min(1e-300)).log()
    initial = fluid.tensor(
        [math.log(target.phases[0].pressure_pa), target.phases[0].temperature_k / fluid.tref]
    )
    initial = torch.cat((initial, logits.flatten()))
    scale = max(
        1.0, abs(internal), float(n.sum().cpu()) * 8.31446261815324 * target.phases[0].temperature_k
    )

    def values(y):
        p = y[0].exp()
        t = y[1] * fluid.tref
        allocations = (
            torch.cat(
                (y[2:].reshape(count - 1, len(active)), y.new_zeros((1, len(active)))), 0
            ).softmax(0)
            * n[active]
        )
        amounts = y.new_zeros((count, nc)).index_copy(-1, active, allocations)
        amount = amounts.sum(-1)
        x = amounts / amount[:, None]
        q = [fluid.tensor_homogeneous(p, t, x[j], roots[j]) for j in range(count)]
        chemical = torch.stack(
            [
                x[j, active].clamp_min(1e-300).log() + v["log_fugacity_coefficients"][active]
                for j, v in enumerate(q)
            ]
        )
        vv = torch.stack([amount[j] * v["molar_volume_m3_mol"] for j, v in enumerate(q)])
        uu = torch.stack([amount[j] * v["molar_internal_energy_j_mol"] for j, v in enumerate(q)])
        residual = torch.cat(
            (
                (chemical[:-1] - chemical[-1]).flatten(),
                ((uu.sum() - internal) / scale).reshape(1),
                (vv.sum() / volume).log().reshape(1),
            )
        )
        return residual, amounts, vv, uu, t

    lower = fluid.tensor(
        [math.log(fluid.min_pressure_pa), fluid.min_temperature_k / fluid.tref]
        + [-690.0] * (len(initial) - 2)
    )
    upper = fluid.tensor(
        [math.log(fluid.max_pressure_pa), fluid.max_temperature_k / fluid.tref]
        + [690.0] * (len(initial) - 2)
    )
    solved, _residual, converged, _ = TorchNewton(fluid.device, maximum_iterations=10).solve(
        lambda y: values(y)[0], initial, lower, upper, tolerance=2e-13
    )
    if not bool(converged.cpu()):
        raise HydraulicConvergenceError("Canonical UV target refinement failed")
    _, amounts, vv, uu, t = values(solved)
    amounts = amounts.cpu().tolist()
    volumes = vv.cpu().tolist()
    energies = uu.cpu().tolist()
    for j in range(nc):
        owner = max(range(count), key=lambda i: amounts[i][j])
        amounts[owner][j] = totals[j] - math.fsum(amounts[i][j] for i in range(count) if i != owner)
    owner = max(range(count), key=volumes.__getitem__)
    volumes[owner] = volume - math.fsum(volumes[i] for i in range(count) if i != owner)
    masses = [
        math.fsum(v * c.molar_mass_kg_mol for v, c in zip(row, fluid.components)) for row in amounts
    ]
    momenta = [m * velocity for m in masses]
    owner = max(range(count), key=masses.__getitem__)
    momenta[owner] = momentum - math.fsum(momenta[i] for i in range(count) if i != owner)
    energies = [u + 0.5 * p * velocity for u, p in zip(energies, momenta)]
    energies[owner] = energy - math.fsum(energies[i] for i in range(count) if i != owner)
    phases = recover_homogeneous_phases(
        fluid, amounts, volumes, momenta, energies, roots, [float(t.cpu())] * count
    )
    return PRPhaseEquilibriumState(phases, roots, target.model_id, target.time_s)
