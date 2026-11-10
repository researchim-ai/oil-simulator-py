"""Independent shared-edge work audit; no runtime capillary force helpers."""

import math

import numpy as np
from check_pr_capillary_pressure import audit as audit_source
from check_pr_capillary_pressure import close

from surface.operation import _digest
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from surface.pr_capillary_pressure import PRCapillaryPressure


def audit(network, before, after, report, parameters):
    network.inspect(before)
    network.inspect(after)
    order = tuple(report["pipe_order"])
    if not order or len(set(order)) != len(order):
        raise ValueError("Invalid capillary chain order")
    initial, final = dict(before.pipes), dict(after.pipes)
    pipe = PRActiveMultifluidPipe.from_dict(report["assembly_pipe"])
    old = PRActiveMultifluidState.from_dict(report["assembly_initial"])
    new = PRActiveMultifluidState.from_dict(report["source"]["state"])
    source = PRCapillaryPressure.from_dict(pipe, report["source"]["source_model"])
    expected_source = PRCapillaryPressure(pipe, **parameters)
    if source.model_id != expected_source.model_id:
        raise ValueError("Capillary constitutive parameters differ")
    audit_source(source, old, new, report["source"])
    for view, graph in ((old, initial), (new, final)):
        cells = [c for name in order for c in graph[name].cells]
        if view.cells != tuple(cells):
            raise ValueError("Assembly is not the actual graph inventory")
    if _digest(after.to_dict()) != _digest(report["state"]) or before.time_s != after.time_s:
        raise ValueError("Graph identity or clock differs")
    for name in set(initial) - set(order):
        if initial[name] != final[name]:
            raise ValueError("Unselected pipe changed")
    for field in ("columns", "equipment"):
        if getattr(before, field) != getattr(after, field):
            raise ValueError("Non-pipe inventory changed")
    lengths = [x for name in order for x in network.pipes[name]["pipe"].lengths]
    close(pipe.lengths, lengths, "assembled lengths", atol=0, rtol=0)
    first = network.pipes[order[0]]["pipe"]
    elevations = list(first.elevations)
    for name in order[1:]:
        elevations.extend(network.pipes[name]["pipe"].elevations[1:])
    close(pipe.elevations, elevations, "assembled elevations", atol=0, rtol=0)
    if (pipe.diameter, pipe.gravity, pipe.slots, pipe.fluid.to_dict()) != (
        first.diameter,
        first.gravity,
        first.slots,
        first.fluid.to_dict(),
    ):
        raise ValueError("Assembly physical model differs")
    ending = network.pipes[order[-1]]["boundary_nodes"][1]
    periodic = (
        ending in network.continuations and network.continuations[ending].right_pipe == order[0]
    )
    if (pipe.boundary_mode == "periodic") != periodic:
        raise ValueError("Assembly boundary topology differs")
    owners = [name for name in order for _ in initial[name].cells]
    heights = [
        [
            dict(c)[source.liquid].volume_m3 / (source.width * dx)
            for c, dx in zip(view.cells, lengths, strict=True)
        ]
        for view in (old, new)
    ]
    edges = list(zip(range(len(lengths) - 1), range(1, len(lengths))))
    if pipe.boundary_mode == "periodic":
        edges.append((len(lengths) - 1, 0))
    energies = [{n: [] for n in order} for _ in range(2)]
    works = {n: [] for n in order}
    expected_ports = {}
    for i, j in edges:
        a, b = owners[i], owners[j]
        weight = source.sigma * source.width / ((lengths[i] + lengths[j]) / 2)
        ee = [0.5 * weight * (h[j] - h[i]) ** 2 for h in heights]
        for e, energy in zip(ee, energies, strict=True):
            energy[a].append(e / 2)
            energy[b].append(e / 2)
        if a != b or (i == len(lengths) - 1 and j == 0):
            link = network.pipes[a]["boundary_nodes"][1]
            if link not in network.continuations:
                raise ValueError("No physical shared edge")
            connection = network.continuations[link]
            if (connection.left_pipe, connection.right_pipe) != (a, b):
                raise ValueError("Shared edge orientation differs")
            h0, h1 = heights
            force = weight * ((h1[j] + h0[j]) - (h1[i] + h0[i])) / 2
            work = force * ((h1[i] - h0[i]) + (h1[j] - h0[j])) / 2
            works[a].append(work)
            works[b].append(-work)
            expected_ports[link] = (force, work, ee)
    if set(expected_ports) != set(report["capillary_ports"]):
        raise ValueError("Missing or extra capillary port")
    for name, (force, work, ee) in expected_ports.items():
        port = report["capillary_ports"][name]
        link = network.continuations[name]
        if (port["left_pipe"], port["right_pipe"]) != (link.left_pipe, link.right_pipe):
            raise ValueError("Reported port orientation differs")
        close(port["height_conjugate_force_n"], force, "port force", atol=1e-22)
        close(port["work_into_left_j"], work, "left port work", atol=1e-25)
        close(port["work_into_right_j"], -work, "right port work", atol=1e-25)
        close(
            [port["surface_energy_before_j"], port["surface_energy_after_j"]],
            ee,
            "edge energy",
            atol=1e-25,
        )
    if source.wetting is not None:
        from reference_wetting_potential import potential as wall_energy

        for i, h in enumerate(heights):
            values = (
                source.width
                * np.array(lengths)
                * wall_energy(source.wetting.to_dict(), source.sigma, h)
            )
            for cell, name in enumerate(owners):
                energies[i][name].append(values[cell])
    for name in order:
        change = math.fsum(
            v.total_energy_j - u.total_energy_j
            for c, d in zip(initial[name].cells, final[name].cells, strict=True)
            for (_, u), (_, v) in zip(c, d, strict=True)
        )
        defect = (
            change
            + math.fsum(energies[1][name])
            - math.fsum(energies[0][name])
            - math.fsum(works[name])
        )
        allowance = (
            128
            * math.fsum(
                math.ulp(p.total_energy_j)
                for graph in (initial, final)
                for c in graph[name].cells
                for _, p in c
            )
            + 1e-22
        )
        close(defect, 0, "segment augmented energy", atol=allowance, rtol=0)
        close(
            report["segment_energy_residuals_j"][name],
            defect,
            "reported segment defect",
            atol=allowance,
            rtol=0,
        )
    return {"accepted": True, "shared_edges": len(expected_ports)}
