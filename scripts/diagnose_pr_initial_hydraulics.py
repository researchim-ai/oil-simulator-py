"""Read-only inventory and hydraulic time-scale diagnosis of a PR checkpoint.

Pressure-head and viscous scales are screening estimates, not replacements for
Riemann fluxes, phase stability or a transient solution. No state is modified.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState


def diagnose(config):
    model = PRControlledSurface.from_dict(config["model"])
    state = PRControlledSurfaceState.from_dict(config["initial"]["state"])
    before = state.to_dict()
    model.inspect(state)
    transport = model.plant.transport
    stocks = dict(state.plant.transport.nodes)
    nodes = {n.name: n for n in transport.nodes}
    rows = []
    with torch.no_grad():
        for i, node in enumerate(transport.nodes):
            if node.kind != "pipe_cell":
                continue
            s = stocks[node.name]
            data = transport._state_data(i, s)
            rho = float(data["rho"])
            speed = float(data["q"]["frozen_sound_speed_m_s"].min())
            adjacent = []
            for link in transport.links:
                if node.name not in (link.source, link.target):
                    continue
                other = link.target if link.source == node.name else link.source
                neighbor = stocks[other]
                adjacent.append(
                    {
                        "link": link.name,
                        "neighbor": other,
                        "pressure_difference_neighbor_minus_cell_pa": neighbor.pressure_pa
                        - s.pressure_pa,
                        "neighbor_kind": nodes[other].kind,
                    }
                )
            mu = node.dynamic_viscosity_pa_s
            row = {
                "name": node.name,
                "volume_m3": node.volume_m3,
                "length_m": node.length_m,
                "diameter_m": node.diameter_m,
                "pressure_pa": s.pressure_pa,
                "temperature_k": s.temperature_k,
                "velocity_m_s": float(data["u"]),
                "density_kg_m3": rho,
                "phase_roots": list(s.phase_roots),
                "frozen_acoustic_transit_s": node.length_m / speed,
                "declared_wave_bound_transit_s": node.length_m / transport.wave_speed_bound_m_s,
                "laminar_momentum_relaxation_s": rho * node.diameter_m**2 / (32 * mu)
                if mu
                else None,
                "adjacent_pressure_heads": adjacent,
            }
            rows.append(row)
    if state.to_dict() != before:
        raise RuntimeError("Read-only diagnosis changed checkpoint")
    return {
        "format": "pr_initial_hydraulics_screen_v1",
        "state_unchanged": True,
        "pipes": sorted(rows, key=lambda r: r["volume_m3"]),
        "scope": "Screening scales only; multiphase acoustics and nonlinear hydraulic equilibrium are not certified by this report",
        "initial_state_sha256": hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()
        ).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = diagnose(json.loads(args.config.read_text()))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pipes": len(result["pipes"]), "state_unchanged": result["state_unchanged"]}))


if __name__ == "__main__":
    main()
