"""Explicit conservative mapping of one horizontal pipe cell to a Darcy link.

The cell's volume, component stocks and total energy are assigned to the
receiving finite vessel, then a UV flash reconstructs its equilibrium state.
Only a restart checkpoint can be transformed; the original request is kept.
"""

import argparse
import copy
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from surface.pr_darcy_connection import PRDarcyConnection
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState


def merge_pipe_stocks(t, target, body, pipes):
    """Retain all N/E in a receiver whose volume already includes the pipes."""
    f = t.fluid
    node = t.nodes[t.names.index(target)]
    if body.momentum_kg_m_s != 0 or any(p.momentum_kg_m_s != 0 for p in pipes):
        raise ValueError("Mapping requires resting initial stocks")
    n = tuple(
        math.fsum(row) for row in zip(body.component_moles, *(p.component_moles for p in pipes))
    )
    energy = math.fsum([body.total_energy_j, *(p.total_energy_j for p in pipes)])
    mass = math.fsum(v * c.molar_mass_kg_mol for v, c in zip(n, f.components))
    internal = energy - mass * t.gravity_m_s2 * (node.elevation_m - t.reference_elevation_m)
    q = f.flash_uv(
        n,
        internal,
        node.volume_m3,
        pressure_guess_pa=body.pressure_pa,
        temperature_guess_k=body.temperature_k,
    )
    count = math.fsum(n)
    amounts = tuple(
        tuple((count * p["molar_fraction"] * p["mole_fractions"]).cpu().tolist())
        for p in q["phases"]
    )
    return replace(
        body,
        component_moles=n,
        total_energy_j=energy,
        phase_component_moles=amounts,
        phase_roots=tuple(t.cells[t.names.index(target)]._tags(q)),
        pressure_pa=float(q["pressure_pa"]),
        temperature_k=float(q["temperature_k"]),
    )


def reduce_cell(config, name="gas_recycle_0"):
    original = PRControlledSurface.from_dict(config["model"])
    state = PRControlledSurfaceState.from_dict(config["initial"]["state"])
    original.inspect(state)
    request = copy.deepcopy(config)
    plant = request["model"]["plant"]
    transport = plant["transport"]
    nodes = {n["name"]: n for n in transport["nodes"]}
    cell = nodes[name]
    if cell["kind"] != "pipe_cell" or any(
        cell.get(k)
        for k in (
            "wall",
            "drift_flux",
            "droplet_population",
            "elevation_change_m",
            "hydrostatic_reconstruction",
            "distribution_strength",
            "buoyancy_relaxation_s",
        )
    ):
        raise ValueError("Reduction requires a simple horizontal viscous pipe cell")
    links = [l for l in transport["links"] if name in (l["source"], l["target"])]
    left = [l for l in links if l["target"] == name and l["target_port"] == "left"]
    right = [l for l in links if l["source"] == name and l["source_port"] == "right"]
    if len(links) != 2 or len(left) != 1 or len(right) != 1:
        raise ValueError("Reduction needs exactly one oriented two-ended cell")
    source, target = left[0]["source"], right[0]["target"]
    if nodes[source]["kind"] != "vessel" or nodes[target]["kind"] != "vessel":
        raise ValueError("Reduced endpoints must be finite stirred vessels")
    if not nodes[source]["elevation_m"] == nodes[target]["elevation_m"] == cell["elevation_m"]:
        raise ValueError("Reduction requires a common gravitational datum")
    references = [
        *plant["ports"],
        *plant.get("heat_links", []),
        *transport.get("pump_connections", []),
        *transport.get("compressors", []),
        *transport.get("restrictions", []),
        *transport.get("heat_links", []),
    ]
    if any(name in json.dumps(item) for item in references):
        raise ValueError("Cell participates in another physical connection")
    link_name = name + "_darcy"
    connection = PRDarcyConnection(
        link_name,
        source,
        target,
        cell["length_m"],
        cell["diameter_m"],
        cell["dynamic_viscosity_pa_s"],
        roughness_m=cell["roughness_m"],
        local_loss_coefficient=1.0 + cell["local_loss_coefficient"],
    )
    plant.setdefault("darcy_connections", []).append(asdict(connection))
    transport["nodes"] = [n for n in transport["nodes"] if n["name"] != name]
    removed = {l["name"] for l in links}
    transport["links"] = [l for l in transport["links"] if l["name"] not in removed]
    for command in request["schedule"]:
        openings = command.get("openings", {})
        modes = {openings.get(n, True) for n in removed}
        if len(modes) != 1:
            raise ValueError("Partial isolation of stored pipe fluid has no zero-holdup equivalent")
        if removed & openings.keys():
            for n in removed:
                openings.pop(n, None)
            openings[link_name] = modes.pop()
    nodes[target]["volume_m3"] += cell["volume_m3"]
    reduced = PRControlledSurface.from_dict(request["model"])
    t = reduced.plant.transport
    old = dict(state.plant.transport.nodes)
    body, pipe = old[target], old[name]
    if len(body.phase_roots) != 1 or len(pipe.phase_roots) != 1:
        raise ValueError("Mapping requires single-phase initial stocks")
    merged = merge_pipe_stocks(t, target, body, [pipe])
    if len(merged.phase_roots) != 1:
        raise ValueError("Merged Darcy receiver is not single phase")
    ts = replace(
        state.plant.transport,
        nodes=tuple((n, merged if n == target else old[n]) for n in t.names),
        model_id=t.model_id,
    )
    physical = replace(state.plant, transport=ts, model_id=reduced.plant.model_id)
    checkpoint = replace(state, plant=physical, model_id=reduced.model_id)
    reduced.inspect(checkpoint)
    request["initial"] = {"state": checkpoint.to_dict()}
    mapping = {
        "removed_cell": name,
        "receiving_vessel": target,
        "connection": link_name,
        "relocated_volume_m3": cell["volume_m3"],
        "relocated_component_moles": pipe.component_moles,
        "relocated_total_energy_j": pipe.total_energy_j,
        "scope": "Explicit reduced model; fast cell inertia and spatial line storage are not resolved. Total initial N/E/V are retained in finite vessels.",
    }
    return request, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request, mapping = reduce_cell(json.loads(args.config.read_text()))
    with args.output.open("x") as handle:
        json.dump(request, handle, indent=2, allow_nan=False)
    with args.output.with_suffix(".mapping.json").open("x") as handle:
        json.dump(mapping, handle, indent=2, allow_nan=False)
    print(json.dumps(mapping))


if __name__ == "__main__":
    main()
