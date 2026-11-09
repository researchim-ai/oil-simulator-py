"""Conservative periodic parcel remap for externally supplied phase-volume flux.

This is the material part of a split operator, not a new pressure/momentum
closure: fluxes must come from a physically coupled model. PR stocks are
transported once; donor volume, components, momentum and total energy share
one upwind fraction. No missing-phase floor or post-update clipping.
"""

import math
from dataclasses import replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .pr_homogeneous_uv import recover_homogeneous_phases
from .pr_stock_balances import entropy_allowance


def remap_phase_volumes(pipe, state, dt_s, face_volume_fluxes_m3_s):
    """Face i points from periodic cell i to i+1; all phase fluxes explicit.

    The caller owns the physical clock/splitting. This source does not change
    state.time_s. Sum of phase-volume fluxes must vanish at every face so
    fixed Eulerian cells remain full. Absent donors cannot supply parcels.
    """
    dt = _number(dt_s, "parcel interval", positive=True)
    old = pipe.inspect(state)
    if pipe.gravity:
        raise ValueError("Surveyed gravity work requires a separate coupled parcel operator")
    if pipe.boundary_mode != "periodic":
        raise ValueError("Periodic parcel operator requires periodic pipe boundaries")
    if set(face_volume_fluxes_m3_s) != set(pipe.names):
        raise ValueError("Explicit face volume flux for every declared phase required")
    f = pipe.fluid
    count = len(state.cells)
    nc = len(f.names)
    flux = torch.stack([f.tensor(face_volume_fluxes_m3_s[n]) for n in pipe.names], dim=-1)
    if tuple(flux.shape) != (count, len(pipe.names)) or not bool(torch.isfinite(flux).all().cpu()):
        raise ValueError("One finite flux per phase per periodic face required")
    if not bool(
        (flux.sum(-1).abs() <= 64 * torch.finfo(flux.dtype).eps * flux.abs().sum(-1)).all().cpu()
    ):
        raise ConservationError("Phase volume fluxes must balance at each fixed-volume face")
    rows = []
    for cell in state.cells:
        phases = dict(cell)
        rows.append(
            [
                [
                    *(phases[n].component_moles),
                    phases[n].volume_m3,
                    phases[n].momentum_kg_m_s,
                    phases[n].total_energy_j,
                ]
                if n in phases
                else [0.0] * (nc + 3)
                for n in pipe.names
            ]
        )
    stocks = f.tensor(rows)
    volume = stocks[..., nc]
    right = torch.roll(stocks, -1, 0)
    donor = torch.where((flux >= 0)[..., None], stocks, right)
    donor_volume = donor[..., nc]
    if bool(((flux != 0) & (donor_volume <= 0)).any().cpu()):
        raise PropertyModelError("Absent phase cannot donate a material parcel")
    outgoing = dt * (flux.clamp_min(0) + (-torch.roll(flux, 1, 0)).clamp_min(0))
    if bool((outgoing > volume).any().cpu()):
        raise HydraulicConvergenceError("Parcel outflow exceeds donor inventory; reduce step")
    # No division by a fictitious residual volume: zero-flux absent phases
    # use denominator 1 and remain exactly zero.
    ratio = torch.where(
        donor_volume > 0,
        flux / torch.where(donor_volume > 0, donor_volume, torch.ones_like(donor_volume)),
        torch.zeros_like(flux),
    )
    transport = donor * ratio[..., None]
    updated = stocks - dt * (transport - torch.roll(transport, 1, 0))
    data = updated.cpu().tolist()
    locations = []
    numbers = []
    volumes = []
    momenta = []
    energies = []
    roots = []
    guesses = []
    previous_cells = [dict(c) for c in state.cells]
    first_temperature = {}
    for cell in previous_cells:
        for name, phase in cell.items():
            first_temperature.setdefault(name, phase.temperature_k)
    for i, cell in enumerate(data):
        for j, row in enumerate(cell):
            if all(v == 0.0 for v in row):
                continue
            if min(row[:nc]) < 0 or row[nc] <= 0 or not math.fsum(row[:nc]) > 0:
                raise PropertyModelError("Parcel update left a nonphysical active phase")
            name = pipe.names[j]
            previous = previous_cells[i].get(name)
            guess = previous.temperature_k if previous else first_temperature[name]
            locations.append((i, name))
            numbers.append(row[:nc])
            volumes.append(row[nc])
            momenta.append(row[nc + 1])
            energies.append(row[nc + 2])
            roots.append(pipe.slots[j]["root"])
            guesses.append(guess)
    phases = recover_homogeneous_phases(f, numbers, volumes, momenta, energies, roots, guesses)
    cells = [[] for _ in state.cells]
    for (i, name), phase in zip(locations, phases, strict=True):
        cells[i].append((name, phase))
    new = replace(state, cells=tuple(tuple(c) for c in cells))
    after = pipe.inspect(new)
    production = after["entropy"] - old["entropy"]
    if production < -entropy_allowance(after["entropy"], old["entropy"]):
        raise ConservationError("Parcel remap decreased PR entropy")
    residual = pipe.ledger(state, new)
    return new, {
        "dt_s": dt,
        "state": new.to_dict(),
        "face_volume_fluxes_m3_s": flux.cpu().tolist(),
        "face_stock_fluxes": transport.cpu().tolist(),
        "entropy_production_j_k": production,
        "conservation_residuals": residual,
    }


def transfer_film_increment(pipe, state, film, film_state, dt_s, *, liquid_slot, gas_slot):
    """Material/thermal split of an unforced 1D film increment into PR stocks.

    This operator supplies parcel transport and the capillary heat ledger.
    It does NOT supply capillary traction to the inertial PR momentum
    equations and is not a complete coupled pipe time integrator.
    """
    from .thin_film_dynamics import ThinFilmDynamics

    if not isinstance(film, ThinFilmDynamics):
        raise TypeError("The overdamped transfer cannot discard an inertial film's kinetic energy")
    pipe.inspect(state)
    film.inspect(film_state)
    if liquid_slot == gas_slot or set(pipe.names) != {liquid_slot, gas_slot}:
        raise ValueError("Film transfer requires declared liquid and gas slots only")
    if (
        pipe.slots[pipe.names.index(liquid_slot)]["root"] != "liquid"
        or pipe.slots[pipe.names.index(gas_slot)]["root"] != "vapor"
    ):
        raise ValueError("Film transfer requires liquid/vapor roots")
    if film.shape != (len(state.cells),) or any(
        abs(dx - film.spacing[0]) > 1e-12 * dx for dx in pipe.lengths
    ):
        raise ValueError("Film and periodic pipe axial meshes must coincide")
    if film.width > math.pi * pipe.diameter * (1 + 1e-12):
        raise ValueError("Film strip exceeds pipe wall perimeter")
    if abs(film_state.time_s - state.time_s) > 1e-14:
        raise ValueError("Film and PR clocks differ")
    old_h = film.tensor(film_state.thickness_m)
    if bool((old_h / pipe.diameter > 0.005).any().cpu()):
        raise ValueError("Planar strip approximation requires h/R <=.01")
    for i, h in enumerate(film_state.thickness_m):
        p = dict(state.cells[i]).get(liquid_slot)
        expected = h * film.cell_area
        if p is None or abs(p.volume_m3 - expected) > 1e-24 + 1e-10 * expected:
            raise ConservationError("Wall geometry and actual PR liquid stock differ")
    new_film, fr = film.step(film_state, dt_s)
    if any(h / pipe.diameter > 0.005 for h in new_film.thickness_m):
        raise PropertyModelError("Film increment exceeds the thin planar strip domain")
    flux = [v * film.width for v in fr["face_volume_flux_m2_s"][0]]
    advected, ar = remap_phase_volumes(
        pipe, state, dt_s, {liquid_slot: flux, gas_slot: [-v for v in flux]}
    )
    face_heat = film.tensor(fr["face_dissipation_j"][0])
    heat = ((face_heat + torch.roll(face_heat, 1, 0)) / 2).cpu().tolist()
    phases = [dict(c)[liquid_slot] for c in advected.cells]
    recovered = recover_homogeneous_phases(
        pipe.fluid,
        [p.component_moles for p in phases],
        [p.volume_m3 for p in phases],
        [p.momentum_kg_m_s for p in phases],
        [p.total_energy_j + q for p, q in zip(phases, heat, strict=True)],
        ["liquid"] * len(phases),
        [p.temperature_k for p in phases],
    )
    cells = []
    for cell, p in zip(advected.cells, recovered, strict=True):
        cells.append(tuple((n, p if n == liquid_slot else v) for n, v in cell))
    end = replace(advected, cells=tuple(cells), time_s=new_film.time_s)
    after = pipe.inspect(end)
    before = pipe.inspect(state)
    if after["entropy"] - before["entropy"] < -entropy_allowance(
        after["entropy"], before["entropy"]
    ):
        raise ConservationError("Film material/thermal transfer decreased PR entropy")
    residual = pipe.ledger(state, end, external_heat_j=math.fsum(heat))
    joined_energy = residual[-2] + fr["free_energy_change_j"] + math.fsum(heat)
    roundoff = 64 * math.fsum(
        math.ulp(p.total_energy_j) for s in (state, end) for c in s.cells for _, p in c
    )
    if abs(joined_energy) > roundoff + 1e-18 + 1e-9 * (
        abs(fr["free_energy_change_j"]) + math.fsum(heat)
    ):
        raise ConservationError("Joint PR/film energy balance failed")
    for i, h in enumerate(new_film.thickness_m):
        expected = h * film.cell_area
        if abs(dict(end.cells[i])[liquid_slot].volume_m3 - expected) > 1e-24 + 1e-10 * expected:
            raise ConservationError("PR transfer did not follow film volume flux")
    return (
        end,
        new_film,
        {
            "film": fr,
            "parcel_transfer": ar,
            "state": end.to_dict(),
            "film_state": new_film.to_dict(),
            "phase_heat_j": heat,
            "joint_energy_residual_j": joined_energy,
            "conservation_residuals": residual,
        },
    )
