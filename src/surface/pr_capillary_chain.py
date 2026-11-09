"""Shared capillary geometry and work ports along a continued physical line.

Assembly concatenates existing cells without introducing physical stocks.
Normal pressure/heat/chemical unknowns on both sides of each continuation are
solved in one Newton. This is not an axial capillary transport integrator.
"""

import math
from dataclasses import replace

from .liquid import ConservationError
from .pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from .pr_capillary_pressure import PRCapillaryPressure


def assemble(network, state, pipe_order, parameters):
    network.inspect(state)
    names = tuple(pipe_order)
    if not names or len(set(names)) != len(names) or set(names) - set(network.pipes):
        raise ValueError("Unique known continued pipe order required")
    for a, b in zip(names, names[1:]):
        linkname = network.pipes[a]["boundary_nodes"][1]
        if (
            linkname not in network.continuations
            or network.pipes[b]["boundary_nodes"][0] != linkname
        ):
            raise ValueError("Capillary chain must follow direct continuations")
        link = network.continuations[linkname]
        if (link.left_pipe, link.right_pipe) != (a, b):
            raise ValueError("Capillary chain orientation differs")
    for link in network.continuations.values():
        if (link.left_pipe in names) != (link.right_pipe in names):
            raise ValueError("Capillary assembly must include the whole continuation component")
    # Only an actual last-to-first continuation closes the geometry.
    closing = network.pipes[names[-1]]["boundary_nodes"][1]
    periodic = (
        closing in network.continuations
        and network.continuations[closing].right_pipe == names[0]
        and network.pipes[names[0]]["boundary_nodes"][0] == closing
    )
    pipes = [network.pipes[n]["pipe"] for n in names]
    first = pipes[0]
    if any(p.diameter != first.diameter or p.slots != first.slots for p in pipes):
        raise ValueError("One common section and phase registry required")
    lengths = [length for p in pipes for length in p.lengths]
    elevations = list(first.elevations)
    for p in pipes[1:]:
        if elevations[-1] != p.elevations[0]:
            raise ValueError("One coordinate for each shared capillary face required")
        elevations.extend(p.elevations[1:])
    assembled = PRActiveMultifluidPipe(
        first.fluid,
        first.slots,
        lengths,
        first.diameter,
        boundary_mode="periodic" if periodic else "walls",
        face_elevations_m=elevations,
        gravity_m_s2=first.gravity,
        wave_speed_bound_m_s=first.wave_speed,
        maximum_courant=min(p.courant for p in pipes),
    )
    inventory = dict(state.pipes)
    cells = tuple(cell for name in names for cell in inventory[name].cells)
    view = PRActiveMultifluidState(cells, state.time_s, assembled.model_id)
    assembled.inspect(view)
    source = PRCapillaryPressure(assembled, **parameters)
    return assembled, view, source


def relax(network, state, dt_s, pipe_order, parameters):
    order = tuple(pipe_order)
    assembled, old, source = assemble(network, state, order, parameters)
    new, record = source.apply(old, dt_s)
    inventory = dict(state.pipes)
    offsets = {}
    cursor = 0
    for name in order:
        count = len(inventory[name].cells)
        offsets[name] = (cursor, cursor + count)
        inventory[name] = replace(inventory[name], cells=new.cells[cursor : cursor + count])
        cursor += count
    end = replace(state, pipes=tuple(sorted(inventory.items())))
    network.inspect(end)
    f = assembled.fluid
    oldh = f.tensor([dict(c)[source.liquid].volume_m3 for c in old.cells]) / source.area
    newh = f.tensor([dict(c)[source.liquid].volume_m3 for c in new.cells]) / source.area
    mid = (oldh + newh) / 2
    delta = newh - oldh
    owners = [name for name in order for _ in inventory[name].cells]
    local_before = {n: 0.0 for n in order}
    local_after = {n: 0.0 for n in order}
    work = {n: 0.0 for n in order}
    ports = {}
    edges = [(i, i + 1) for i in range(len(old.cells) - 1)]
    if assembled.boundary_mode == "periodic":
        edges.append((len(old.cells) - 1, 0))
    for edge, (i, j) in enumerate(edges):
        weight = float(source.weights[edge].cpu())
        a = owners[i]
        b = owners[j]
        ea = 0.5 * weight * float((oldh[j] - oldh[i]).square().cpu())
        eb = 0.5 * weight * float((newh[j] - newh[i]).square().cpu())
        local_before[a] += 0.5 * ea
        local_before[b] += 0.5 * ea
        local_after[a] += 0.5 * eb
        local_after[b] += 0.5 * eb
        if a != b or (i == len(old.cells) - 1 and j == 0):
            linkname = network.pipes[a]["boundary_nodes"][1]
            if linkname not in network.continuations:
                raise ValueError("Missing physical continuation at assembled surface edge")
            force = weight * float((mid[j] - mid[i]).cpu())
            port_work = force * float(((delta[i] + delta[j]) / 2).cpu())
            work[a] += port_work
            work[b] -= port_work
            ports[linkname] = {
                "left_pipe": a,
                "right_pipe": b,
                "height_conjugate_force_n": force,
                "work_into_left_j": port_work,
                "work_into_right_j": -port_work,
                "surface_energy_before_j": ea,
                "surface_energy_after_j": eb,
            }
    block_residuals = {}
    for name in order:
        i, j = offsets[name]
        bulk = math.fsum(
            b.total_energy_j - a.total_energy_j
            for ca, cb in zip(old.cells[i:j], new.cells[i:j], strict=True)
            for (_, a), (_, b) in zip(ca, cb, strict=True)
        )
        residual = bulk + local_after[name] - local_before[name] - work[name]
        allowance = (
            128
            * math.fsum(
                math.ulp(p.total_energy_j)
                for view in (old, new)
                for c in view.cells[i:j]
                for _, p in c
            )
            + 1e-22
        )
        if not math.isfinite(residual) or abs(residual) > allowance:
            raise ConservationError("Capillary continuation work/segment energy balance failed")
        block_residuals[name] = residual
    before = network.totals(state)
    after = network.totals(end)
    before[-1] += record["surface_energy_before_j"]
    after[-1] += record["surface_energy_after_j"]
    residual = [b - a for a, b in zip(before, after, strict=True)]
    for i, value in enumerate(residual):
        if abs(value) > 256 * (math.ulp(before[i]) + math.ulp(after[i])) + 1e-20:
            raise ConservationError("Shared capillary graph source ledger failed")
    return end, {
        "state": end.to_dict(),
        "dt_s": record["dt_s"],
        "clock_advanced": False,
        "pipe_order": list(order),
        "assembly_pipe": assembled.to_dict(),
        "assembly_initial": old.to_dict(),
        "source": record,
        "capillary_ports": ports,
        "segment_energy_residuals_j": block_residuals,
        "surface_energy_before_j": record["surface_energy_before_j"],
        "surface_energy_after_j": record["surface_energy_after_j"],
        "conservation_residuals": residual,
        "qualification": "Shared normal/chemical capillary geometry and work ports; axial capillary transport and topology change remain separate",
    }
