"""Single-phase PR column initialization from a specified local port state.

Prescribes thermodynamic equilibrium, integrates its N/U+PE, and checks the
existing conservative column residual. Never overwrites a running inventory.
"""

import math

import torch

from .liquid import HydraulicConvergenceError, PropertyModelError, _number
from .peng_robinson import R
from .pr_inclined_pipe import solve_chemical_nodes


def initialize_column_at_height(column, pressure_pa, temperature_k, mole_fractions, *, height_m):
    f = column.fluid
    height = _number(height_m, "anchor height", nonnegative=True)
    pressure = _number(pressure_pa, "anchor pressure", positive=True)
    temperature = _number(temperature_k, "anchor temperature", positive=True)
    if (
        height > column.geometry.height_m
        or not f.min_pressure_pa <= pressure <= f.max_pressure_pa
        or not f.min_temperature_k <= temperature <= f.max_temperature_k
    ):
        raise PropertyModelError("Hydrostatic anchor outside column domain")
    # Existing flash validates registry/normalization and stable one-phase domain.
    flash = f.flash_pt(pressure, temperature, mole_fractions)
    if len(flash["phases"]) != 1:
        raise PropertyModelError("Port-based initialization currently requires one stable phase")
    x = f.tensor(mole_fractions)
    active = torch.nonzero(x > 0).flatten()
    c = len(active)
    if (
        len(x) != len(f.names)
        or not bool((torch.isfinite(x).all() & (x >= 0).all()).cpu())
        or abs(float(x.sum().cpu()) - 1) > 1e-10
    ):
        raise ValueError("Normalized full registry composition required")
    q = f.tensor_homogeneous(f.tensor(pressure), f.tensor(temperature), x, "stable")
    mu = torch.log(x[active]) + q["log_fugacity_coefficients"][active] + math.log(pressure / f.pref)
    lam = mu + f.molar_masses[active] * column.gravity_m_s2 * height / (R * temperature)
    initial = torch.cat((f.tensor([math.log(pressure)]), (x[active][:-1] / x[active][-1]).log()))
    previous = None
    order = column.quadrature_order
    while order <= column.maximum_quadrature_order:
        hh, ww = column._grid(f.tensor([]), order)
        h = hh[0]
        w = ww[0]
        target = lam[None, :] - f.molar_masses[active][None, :] * column.gravity_m_s2 * h[
            :, None
        ] / (R * temperature)
        rows = solve_chemical_nodes(
            f, initial.expand(order, c), f.tensor([temperature] * order), target, active
        )
        pp, xx, qq = column._decode_nodes(rows, active, f.tensor(temperature), "stable")
        n = (w[:, None] * xx / qq["molar_volume_m3_mol"][:, None]).sum(0)
        u = (w * qq["molar_internal_energy_j_mol"] / qq["molar_volume_m3_mol"]).sum()
        pe = (w * qq["density_kg_m3"] * column.gravity_m_s2 * h).sum()
        mass = (n * f.molar_masses).sum()
        energy = u + pe + mass * column.gravity_m_s2 * column.elevation_m
        full = f.tensor([0.0] * len(f.names)).index_copy(0, active, lam)
        seed = ([xx[0]], ["stable"], f.tensor([]), float(pp[0].cpu()), temperature, full)
        y = torch.cat((rows.flatten(), lam, f.tensor([temperature / f.tref])))
        state, report = column._accept(n, float(energy.cpu()), seed, order, y)
        if previous is not None:
            oldn, olde = previous
            error = max(
                float(
                    (
                        (n - oldn).abs()
                        / (1e-10 + column.relative_tolerance * torch.maximum(n.abs(), oldn.abs()))
                    )
                    .max()
                    .cpu()
                ),
                float(
                    (
                        (u + pe - olde).abs()
                        / (
                            1e-5
                            + column.relative_tolerance * torch.maximum((u + pe).abs(), olde.abs())
                        )
                    ).cpu()
                ),
            )
            if error <= 1:
                return state, report
        previous = (n, u + pe)
        order *= 2
    raise HydraulicConvergenceError("Port-based column initialization quadrature exhausted")


def initialize_network_equilibrium(
    network,
    pressure_pa,
    temperature_k,
    mole_fractions,
    *,
    pipe_wall_temperature_k=None,
    vessel_wall_temperature_k=None,
):
    p = network.pipe.pipe
    if hasattr(p, "initialize_equilibrium"):
        ps = p.initialize_equilibrium(pressure_pa, temperature_k, mole_fractions)
    else:
        ps = p.initialize(pressure_pa, temperature_k, mole_fractions)
    tw = temperature_k if pipe_wall_temperature_k is None else pipe_wall_temperature_k
    pipe = network.pipe.initialize(ps, tw)
    walls = [temperature_k] * 2 if vessel_wall_temperature_k is None else vessel_wall_temperature_k
    if len(walls) != 2:
        raise ValueError("Two equilibrium vessel wall temperatures required")
    data = p._decode(p.fluid.tensor(ps.solution_variables), p._support(ps.component_moles))
    states = []
    for i, (col, h) in enumerate(zip(network.columns, network.heights)):
        endpoint, index = p._port_data(data, i)
        state, _ = initialize_column_at_height(
            col,
            float(endpoint["p"][index].cpu()),
            float(endpoint["t"][index].cpu()),
            endpoint["x"][index].cpu().tolist(),
            height_m=h,
        )
        states.append(network.initial_vessel_state(i, state, walls[i]))
    return network.initialize(pipe, states)
