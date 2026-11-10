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


def assemble(network, state, pipe_order, parameters, *, port_openings=None):
    network.inspect(state)
    openings = {} if port_openings is None else dict(port_openings)
    if set(openings) - set(network.continuations):
        raise ValueError("Unknown capillary continuation opening")
    if any(type(value) is not bool for value in openings.values()):
        raise ValueError("Capillary continuation openings must be boolean")
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
        if link.left_pipe in names and not openings.get(
            network.pipes[link.left_pipe]["boundary_nodes"][1], True
        ):
            raise ValueError("Closed continuation requires separate capillary geometries")
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


def relax(network, state, dt_s, pipe_order, parameters, *, port_openings=None):
    order = tuple(pipe_order)
    assembled, old, source = assemble(
        network, state, order, parameters, port_openings=port_openings
    )
    if any(h is not None for h in source.boundary_heights):
        raise ValueError("Fixed face geometry requires explicit external surface-work ports")
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
    if source.wetting is not None:
        phi0 = (source.area * source.wetting.potential(oldh)).cpu().tolist()
        phi1 = (source.area * source.wetting.potential(newh)).cpu().tolist()
        for i, name in enumerate(owners):
            local_before[name] += phi0[i]
            local_after[name] += phi1[i]
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


def transport_work(
    network,
    before,
    after,
    pipe_order,
    parameters,
    integrated_liquid_flux_m3,
    *,
    volume_source_m3=None,
    port_openings=None,
):
    """Constitutive axial work on real graph states, not a state integrator."""
    from .pr_capillary_transport import PRCapillaryTransportWork

    order = tuple(pipe_order)
    _, old, source = assemble(network, before, order, parameters, port_openings=port_openings)
    _, new, _ = assemble(network, after, order, parameters, port_openings=port_openings)
    record = PRCapillaryTransportWork(source).evaluate(
        old, new, integrated_liquid_flux_m3, volume_source_m3=volume_source_m3
    )
    periodic = source.pipe.boundary_mode == "periodic"
    segments, ports = {}, {}
    cursor = 0
    for name in order:
        count = len(dict(before.pipes)[name].cells)
        left = (cursor - 1) % len(old.cells) if periodic else cursor
        right = (cursor + count - 1) % len(old.cells) if periodic else cursor + count
        segments[name] = {
            "surface_energy_before_j": math.fsum(
                record["surface_energy_before_j"][cursor : cursor + count]
            ),
            "surface_energy_after_j": math.fsum(
                record["surface_energy_after_j"][cursor : cursor + count]
            ),
            "axial_bulk_work_j": math.fsum(record["axial_bulk_work_j"][cursor : cursor + count]),
            "normal_bulk_work_j": math.fsum(record["normal_bulk_work_j"][cursor : cursor + count]),
            "surface_energy_in_j": record["face_surface_energy_transfer_j"][left],
            "surface_energy_out_j": record["face_surface_energy_transfer_j"][right],
            "energy_residual_j": math.fsum(
                record["local_energy_residual_j"][cursor : cursor + count]
            ),
        }
        node = network.pipes[name]["boundary_nodes"][1]
        if node in network.continuations:
            ports[node] = {
                "left_pipe": name,
                "right_pipe": network.continuations[node].right_pipe,
                "liquid_volume_transfer_m3": record["integrated_liquid_flux_m3"][right],
                "surface_energy_into_right_j": record["face_surface_energy_transfer_j"][right],
                "surface_energy_into_left_j": -record["face_surface_energy_transfer_j"][right],
            }
        cursor += count
    return {**record, "pipe_order": list(order), "segments": segments, "continuations": ports}


def advance_line(
    network,
    state,
    duration_s,
    pipe_order,
    parameters,
    *,
    port_openings=None,
    normal_relaxation=True,
    require_periodic=False,
    maintained_faces=False,
    volume_diffusion="acoustic",
    **step_options,
):
    """Advance a complete isolated uniform continuation line or loop."""
    from .pr_capillary_axial import PRCapillaryAxialPipe

    order = tuple(pipe_order)
    if (
        set(order) != set(network.pipes)
        or network.columns
        or (network.reservoirs and not maintained_faces)
        or network.equipment is not None
        or network.walls
    ):
        raise ValueError("The capillary line must be the complete isolated graph")
    _, initial, source = assemble(network, state, order, parameters, port_openings=port_openings)
    if require_periodic and source.pipe.boundary_mode != "periodic":
        raise ValueError("The capillary loop requires an actual closing continuation")
    boundaries = [None, None]
    if maintained_faces:
        for side, name in enumerate((order[0], order[-1])):
            pipe = network.pipes[name]["pipe"]
            node = network.pipes[name]["boundary_nodes"][side]
            if node in network.reservoirs:
                reservoir = network.reservoirs[node]
                boundaries[side] = network._assign(
                    pipe, (), reservoir.phases, reservoir.phase_roots
                )
        if not any(row is not None for row in boundaries):
            raise ValueError("Open capillary line requires a maintained reservoir endpoint")
    model = PRCapillaryAxialPipe(source, boundaries=boundaries, volume_diffusion=volume_diffusion)
    source = model.source
    step_options.setdefault("initial_step_s", state.next_step_s)
    final, record = model.advance(
        initial, duration_s, normal_relaxation=normal_relaxation, **step_options
    )
    inventory, cursor = dict(state.pipes), 0
    for name in order:
        count = len(inventory[name].cells)
        inventory[name] = replace(
            inventory[name], cells=final.cells[cursor : cursor + count], time_s=final.time_s
        )
        cursor += count
    end = replace(
        state,
        pipes=tuple(sorted(inventory.items())),
        time_s=final.time_s,
        next_step_s=record["next_step_s"],
    )
    network.inspect(end)
    return end, {
        "state": end.to_dict(),
        "pipe_order": list(order),
        "assembly_pipe": source.pipe.to_dict(),
        "assembly_initial": initial.to_dict(),
        "source_model": source.to_dict(),
        "axial_model": model.to_dict(),
        "maintained_faces": maintained_faces,
        "advance": record,
        "normal_relaxation": normal_relaxation,
        "qualification": "Isolated uniform two-phase continuation line or loop; open-flow/branching/topology-changing capillary graph remains unsupported",
    }


def advance_loop(network, state, duration_s, pipe_order, parameters, **options):
    return advance_line(
        network, state, duration_s, pipe_order, parameters, require_periodic=True, **options
    )
