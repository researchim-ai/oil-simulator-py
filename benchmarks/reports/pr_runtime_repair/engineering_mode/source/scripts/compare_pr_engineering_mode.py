"""Endpoint diagnostics for an explicitly reduced PR model.

This reports differences, not an automatic qualification. Removed pipe mean
velocity and algebraic connection velocity need not agree during startup.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

from surface.pr_surface_control import PRControlledSurface


def read(path):
    path = Path(path)
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
        value = json.load(stream)
    return value.get("report", value)


def compare(reference, reduced, mapping):
    a = reference["intervals"][-1]["state"]["plant"]["transport"]
    b = reduced["intervals"][-1]["state"]["plant"]["transport"]
    if a["time_s"] != b["time_s"]:
        raise ValueError("Compare the same physical endpoint")
    model = PRControlledSurface.from_dict(reduced["model"])
    reference_model = PRControlledSurface.from_dict(reference["model"])
    fluid = model.plant.transport.fluid
    if reference_model.plant.transport.fluid.to_dict() != fluid.to_dict():
        raise ValueError("Compare the same component registry and thermodynamics")
    nodes = {n.name: n for n in model.plant.transport.nodes}
    groups = {n: [n] for n in b["nodes"]}
    removed = set()
    for item in mapping["cells"]:
        if item["removed_cell"] in removed:
            raise ValueError("Duplicate removed cell in mapping")
        groups[item["receiving_vessel"]].append(item["removed_cell"])
        removed.add(item["removed_cell"])
    if set(a["nodes"]) != set(b["nodes"]) | removed:
        raise ValueError("Mapping does not cover the reference node registry")
    differences = {}
    for name, names in groups.items():
        old, new = a["nodes"][name], b["nodes"][name]
        differences[name] = {
            "pressure_pa": abs(old["pressure_pa"] - new["pressure_pa"]),
            "temperature_k": abs(old["temperature_k"] - new["temperature_k"]),
            "group_component_moles_max": max(
                abs(math.fsum(a["nodes"][n]["component_moles"][i] for n in names) - v)
                for i, v in enumerate(new["component_moles"])
            ),
            "group_energy_j": abs(
                math.fsum(a["nodes"][n]["total_energy_j"] for n in names) - new["total_energy_j"]
            ),
        }
    velocities = {}
    connections = {c.name: c for c in model.plant.darcy_connections}
    final_openings = reduced["config"]["schedule"][-1].get("openings", {})
    for item in mapping["cells"]:
        c = connections[item["connection"]]
        dp = b["nodes"][c.source]["pressure_pa"] - b["nodes"][c.target]["pressure_pa"]
        if not c.allow_reverse:
            dp = max(0.0, dp)
        donor = c.source if dp >= 0 else c.target
        rho = (
            math.fsum(
                n * p.molar_mass_kg_mol
                for n, p in zip(b["nodes"][donor]["component_moles"], fluid.components)
            )
            / nodes[donor].volume_m3
        )
        velocity = (
            float(c.velocity(fluid.tensor(dp), fluid.tensor(rho)))
            if final_openings.get(c.name, c.open)
            else 0.0
        )
        pipe = a["nodes"][item["removed_cell"]]
        mass = math.fsum(
            n * p.molar_mass_kg_mol for n, p in zip(pipe["component_moles"], fluid.components)
        )
        detailed = pipe["momentum_kg_m_s"] / mass
        velocities[c.name] = {
            "detailed_cell_mean_m_s": detailed,
            "reduced_algebraic_m_s": velocity,
            "absolute_difference_m_s": abs(detailed - velocity),
        }
    return {
        "time_s": a["time_s"],
        "nodes": differences,
        "velocity_diagnostics": velocities,
        "qualification": "Endpoint diagnostics only; no claim of equivalent startup, spatial holdup, controller trajectory or accumulated interfacial flux. Engineering budgets and settled-regime validation are required.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "reduced", "mapping"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read(args.reference), read(args.reduced), json.loads(args.mapping.read_text()))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
