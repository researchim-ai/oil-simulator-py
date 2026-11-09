"""Measured scaling of a genuinely coupled PR pipe/column/pump/rotor network.

Runs real CFL-limited substeps, not prescribed manufactured flows. Reports
physical duration explicitly: these short runs are scaling/latency benchmarks,
not proof of long-horizon stability or industrial throughput. Stage timings
are inclusive host wall times; nested stages must not be added together.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_active_surface import PRActiveColumnPort, PRActiveSurfaceNetwork
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
from surface.pr_interphase_relaxation import PRRelaxationPhase
from surface.separator_geometry import SeparatorGeometry
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
from surface.vessel_wall import VesselWall

ROOT = Path(__file__).resolve().parents[1]


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def build(cells, branches, device):
    if type(cells) is not int or cells < branches * 2 or branches < 4 or cells % branches:
        raise ValueError("At least four branches, two cells/branch, exact partition required")
    base = json.loads((ROOT / "configs/surface_pr_equilibrium_euler_plant_pump.json").read_text())
    raw = base["model"]["transport"]
    raw.update(
        gravity_m_s2=9.80665,
        wave_speed_bound_m_s=4000.0,
        initial_step_s=1e-4,
        maximum_step_s=0.001,
        minimum_step_s=1e-12,
    )
    for node in raw["nodes"]:
        node["volume_m3"] = 10.0
    equipment = PREquilibriumTransportNetwork.from_dict(raw, device=device)
    f = equipment.fluid
    initial = {
        n: {"pressure_pa": 1e6, "temperature_k": 350.0, "mole_fractions": [0.0, 1.0, 0.0]}
        for n in equipment.names
    }
    es = equipment.initialize(initial)
    columns = {}
    cs = {}
    for name, pressure in [("separator", 1e6), ("receiver", 1.0002e6)]:
        column = StratifiedPRColumn(
            f,
            SeparatorGeometry(2.0, 5.0, "vertical"),
            quadrature_order=2,
            maximum_quadrature_order=8,
            elevation_m=-2.5,
        )
        columns[name] = column
        cs[name] = StratifiedPRState.from_dict(
            column.initialize(pressure, 350.0, [0.2, 0.8, 0.0])["state"]
        )
    q = f.flash_pt(1e6, 350.0, [0.2, 0.8, 0.0])
    if len(q["phases"]) != 2:
        raise ValueError("Benchmark explicitly requires two physical PR branches")
    pipes = {}
    ports = []
    states = {}
    per_branch = cells // branches
    for i in range(branches):
        name = f"line_{i:03d}"
        if i % 4 == 0:
            ends = (f"{name}_source", "feed")
            bindings = [(ends[0], "separator")]
        elif i % 4 == 1:
            ends = ("return", f"{name}_target")
            bindings = [(ends[1], "receiver")]
        else:
            ends = (f"{name}_source", f"{name}_target")
            bindings = [(ends[0], "separator"), (ends[1], "receiver")]
        ports.extend(PRActiveColumnPort(n, c, 2.5) for n, c in bindings)
        survey = [0.02 * math.sin(math.pi * j / per_branch) for j in range(per_branch + 1)]
        survey[0] = survey[-1] = 0.0
        pipe = PRActiveMultifluidPipe(
            f,
            [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
            [10.0] * per_branch,
            0.05,
            face_elevations_m=survey,
            gravity_m_s2=9.80665,
            wall_friction={
                "model": "darcy_volume_wetted_v1",
                "viscosity_pa_s": {"liquid": 0.001, "gas": 1e-5},
                "roughness_m": 1e-6,
            },
        )
        cell = {}
        for phase in q["phases"]:
            tag = "liquid" if float(phase["density_kg_m3"].cpu()) > 100 else "vapor"
            slot = "liquid" if tag == "liquid" else "gas"
            fraction = float(
                (
                    phase["molar_fraction"]
                    * phase["molar_volume_m3_mol"]
                    / q["molar_volume_m3_mol"]
                ).cpu()
            )
            volume = pipe.volumes[0] * fraction
            amount = volume / float(phase["molar_volume_m3_mol"].cpu())
            x = phase["mole_fractions"].cpu().tolist()
            mass = amount * sum(v * c.molar_mass_kg_mol for v, c in zip(x, f.components))
            velocity = 0.01 if i % 2 == 0 else -0.01
            cell[slot] = PRRelaxationPhase(
                tuple(amount * v for v in x),
                volume,
                1e6,
                350.0,
                mass * velocity,
                amount * float(phase["molar_internal_energy_j_mol"].cpu())
                + 0.5 * mass * velocity**2,
            )
        states[name] = pipe.initialize([cell] * per_branch)
        pipes[name] = {"pipe": pipe, "boundary_nodes": ends}
    model = PRActiveSurfaceNetwork(
        pipes,
        columns,
        ports,
        equipment=equipment,
        column_walls={
            n: VesselWall(1e5, 100.0, 10.0, 355.0, max_temperature_change_k=2.0) for n in columns
        },
    )
    state = model.initialize(cs, states, es, wall_temperatures_k={n: 352.0 for n in columns})
    return model, state


class Timings:
    def __init__(self):
        self.values = {}
        self.restores = []

    def wrap(self, obj, method, label):
        original = getattr(obj, method)

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                row = self.values.setdefault(label, {"calls": 0, "seconds": 0.0})
                row["calls"] += 1
                row["seconds"] += time.perf_counter() - start

        setattr(obj, method, timed)
        self.restores.append((obj, method, original))

    def restore(self):
        for obj, name, method in reversed(self.restores):
            setattr(obj, name, method)


def independent_totals(model, state):
    n = [0.0] * len(model.fluid.names)
    energy = []
    for name, s in state.pipes:
        pipe = model.pipes[name]["pipe"]
        for cell, z in zip(s.cells, pipe.centers):
            for _, p in cell:
                for j, v in enumerate(p.component_moles):
                    n[j] += v
                mass = math.fsum(
                    v * c.molar_mass_kg_mol
                    for v, c in zip(p.component_moles, model.fluid.components)
                )
                energy.append(p.total_energy_j + mass * model.gravity * z)
    for name, s in state.columns:
        for j, v in enumerate(s.component_moles):
            n[j] += v
        energy.append(s.total_energy_j)
    for name, t in state.wall_temperatures:
        energy.append(
            model.walls[name].heat_capacity_j_k * (t - model.walls[name].reference_temperature_k)
        )
    for node, (_, s) in zip(model.equipment.nodes, state.equipment.nodes):
        for j, v in enumerate(s.component_moles):
            n[j] += v
        mass = math.fsum(
            v * c.molar_mass_kg_mol for v, c in zip(s.component_moles, model.fluid.components)
        )
        energy.append(
            s.total_energy_j + mass * model.gravity * model.equipment.reference_elevation_m
        )
        if node.wall is not None:
            energy.append(node.wall.heat_capacity_j_k * s.wall_temperature_k)
    for name, speed in state.equipment.rotors:
        energy.append(0.5 * model.equipment._drives[name].drive.inertia_kg_m2 * speed**2)
    return n, math.fsum(energy)


def run(cells, branches, steps, device):
    synchronize(device)
    start = time.perf_counter()
    model, initial = build(cells, branches, device)
    synchronize(device)
    setup = time.perf_counter() - start
    timing = Timings()
    timing.wrap(model, "_pipe_step", "pipes_and_apertures_inclusive")
    timing.wrap(model, "inspect", "whole_network_inspect_inclusive")
    timing.wrap(model.equipment, "_step", "equipment_newton_inclusive")
    for c in model.columns.values():
        timing.wrap(c, "sample", "column_aperture_sampling_inclusive")
        timing.wrap(c, "equilibrate", "column_inventory_recovery_inclusive")
    for t in model.thermal.values():
        timing.wrap(t, "_step", "column_thermal_newton_inclusive")
    for e in model.pipes.values():
        p = e["pipe"]
        timing.wrap(p, "inspect", "pipe_eos_inspect_inclusive")
        timing.wrap(p, "_operator", "pipe_face_operator")
        timing.wrap(p.friction, "apply", "pipe_friction_inclusive")
    state = initial
    reports = []
    durations = []
    dt = 0.25 * model.maximum_step_s
    try:
        for index in range(steps):
            synchronize(device)
            start = time.perf_counter()
            state, r = model.step(state, dt)
            synchronize(device)
            elapsed = time.perf_counter() - start
            reports.append(r)
            durations.append(elapsed)
            print(
                json.dumps(
                    {
                        "cells": cells,
                        "step": index + 1,
                        "seconds": elapsed,
                        "simulated_time_s": state.time_s,
                    }
                ),
                flush=True,
            )
    finally:
        timing.restore()
    a, b = independent_totals(model, initial), independent_totals(model, state)
    external = math.fsum(r["external_energy_j"] for r in reports)
    nres = [y - x for x, y in zip(a[0], b[0])]
    eres = b[1] - a[1] - external
    limits = [max(1e-7, 256 * math.ulp(max(abs(x), abs(y), 1.0))) for x, y in zip(a[0], b[0])]
    elimit = max(1e-5, 256 * math.ulp(max(abs(a[1]), abs(b[1]), 1.0)))
    if any(abs(r) > tol for r, tol in zip(nres, limits)) or abs(eres) > elimit:
        raise ValueError("Independent large-network global N/E ledger failed")
    # Full scalar spatial/path/profile audit only on the smallest connected mesh.
    full_audit = None
    if cells <= 64:
        from check_pr_active_surface import audit

        old = initial
        for r in reports:
            new = type(state).from_dict(r["state"])
            audit(model, old, new, r)
            old = new
        full_audit = True
    return {
        "cells": cells,
        "branches": branches,
        "columns": len(model.columns),
        "equipment_vessels": len(model.equipment.nodes),
        "pump_rotors": len(state.equipment.rotors),
        "active_phases": sum(len(c) for _, s in state.pipes for c in s.cells),
        "setup_seconds": setup,
        "step_seconds": durations,
        "mean_step_seconds": sum(durations) / steps,
        "physical_step_s": dt,
        "simulated_duration_s": state.time_s - initial.time_s,
        "simulated_seconds_per_wall_second": (state.time_s - initial.time_s) / sum(durations),
        "timings_inclusive": timing.values,
        "independent_component_residuals_mol": nres,
        "independent_energy_residual_j": eres,
        "energy_roundoff_limit_j": elimit,
        "full_scalar_spatial_audit": full_audit,
        "global_ledger_accepted": True,
        "final_pump_speed_rad_s": dict(state.equipment.rotors),
        "final_state": state.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        raise ValueError("Positive steps and new output required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Explicit CUDA device unavailable")
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__),
        ROOT / "configs/surface_pr_equilibrium_euler_plant_pump.json",
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = {
        "format": "pr_active_surface_scaling_v1",
        "device": args.device,
        "torch": torch.__version__,
        "hardware": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
        "timing_scope": "inclusive host wall timings; CUDA synchronized at global step boundaries",
        "source_sha256": hashes,
        "cases": [],
    }
    for cells in args.cells:
        result["cases"].append(run(cells, args.branches, args.steps, args.device))
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError("Physical sources changed during benchmark")


if __name__ == "__main__":
    main()
