"""Actual CPU/CUDA black-oil FIM → producing wells ↔ physical gathering tree.

Pressure counterfactuals start from the same reservoir state. No Eclipse or
Norne benchmark is implied; rejected substeps are handled by Simulator.
"""

import argparse
import contextlib
import copy
import io
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.fluid import Fluid
from simulator.reservoir import Reservoir
from simulator.simulation import Simulator
from simulator.well import WellManager


def run_case(config, outlet_pa, days, report_days=1.0, device_name="cpu", initial_rs=None):
    c = copy.deepcopy(config)
    network = c["simulation"]["physical_surface_network"]["network"]
    boundaries = [node for node in network["nodes"] if node.get("pressure_pa") is not None]
    if len(boundaries) != 1:
        raise ValueError("Expected one production receiver")
    boundaries[0]["pressure_pa"] = outlet_pa
    device = torch.device(device_name)
    log = io.StringIO()
    steps = []
    interval_timings = []
    with contextlib.redirect_stdout(log):
        r = Reservoir(c["reservoir"], device=device)
        f = Fluid(c["fluid"], r, device=device)
        if initial_rs is not None:
            if not math.isfinite(initial_rs) or initial_rs < 0 or bool((f.s_g != 0).any()):
                raise ValueError(
                    "initial_rs requires finite nonnegative Rs and zero initial free gas"
                )
            f.rs = torch.full_like(f.pressure, initial_rs, dtype=torch.float64)
        sim = Simulator(r, f, WellManager(c["wells"], r), c["simulation"], device=device)
        s = sim.fim_solver
        if initial_rs is not None and bool((f.rs - initial_rs).abs().gt(1e-8).any()):
            raise ValueError("initial_rs exceeds initial saturated Rs")

        def mass():
            return s.masses(f.pressure.flatten(), f.s_w.flatten(), s.x, s.sat, s.ug).sum(1)

        old = mass().detach().cpu()
        previous = [0.0]

        def accepted(time_days):
            dt = (time_days - previous[0]) * 86400
            previous[0] = time_days
            report = copy.deepcopy(s.wells.surface_report)
            rates = torch.tensor(
                [w["component_mass_rates_kg_s"] for w in report["wells"].values()],
                dtype=torch.float64,
            ).sum(0)
            steps.append(
                {
                    "time_days": time_days,
                    "dt_s": dt,
                    "withdrawn_component_mass_kg": (rates * dt).tolist(),
                    "newton_iterations": s.last_newton_iters,
                    "surface": report,
                }
            )

        reservoir_hook = accepted

        def accepted_with_facility(time_days):
            reservoir_hook(time_days)
            if sim.facility_coupling is not None:
                steps[-1]["facility_coupling"] = copy.deepcopy(sim.facility_coupling.report)

        sim.substep_hook = accepted_with_facility
        end = days * 86400
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        while sim.current_time < end:
            interval_start = time.perf_counter()
            if not sim.run_step(min(report_days * 86400, end - sim.current_time)):
                raise RuntimeError("Physical black-oil interval failed")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            interval_timings.append(
                {
                    "time_days": sim.current_time / 86400,
                    "wall_s": time.perf_counter() - interval_start,
                }
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        new = mass().detach().cpu()
    withdrawn = torch.tensor(
        [step["withdrawn_component_mass_kg"] for step in steps], dtype=torch.float64
    ).sum(0)
    return {
        "outlet_pressure_pa": outlet_pa,
        "initial_rs_m3_m3": initial_rs,
        "reservoir_device": str(device),
        "solve_wall_s": elapsed,
        "interval_timings": interval_timings,
        "field_graphs_active": s._graphs is not None if device.type == "cuda" else False,
        "field_graph_cache_entries": len(s._graphs) if s._graphs is not None else 0,
        "elapsed_days": sim.current_time / 86400,
        "component_order": ["oil", "water", "gas"],
        "initial_component_mass_kg": old.tolist(),
        "final_component_mass_kg": new.tolist(),
        "withdrawn_component_mass_kg": withdrawn.tolist(),
        "mass_balance_error_kg": (old - new - withdrawn).tolist(),
        "accepted_substeps": steps,
        "final_pressure_pa": f.pressure.flatten().tolist(),
        "final_s_w": f.s_w.flatten().tolist(),
        "final_s_g": f.s_g.flatten().tolist(),
        "final_rs": f.rs.flatten().tolist(),
        "solver_log": log.getvalue(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--outlet-mpa", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    parser.add_argument(
        "--initial-rs",
        type=float,
        help="Explicit dissolved gas ratio; requires zero initial free gas",
    )
    parser.add_argument("--days", type=float, default=1)
    parser.add_argument("--report-days", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--network-backend", choices=["cpu_reference", "torch_device"], default="cpu_reference"
    )
    args = parser.parse_args()
    try:
        if (
            not math.isfinite(args.days)
            or not math.isfinite(args.report_days)
            or not args.days > 0
            or not args.report_days > 0
        ):
            raise ValueError("days must be positive")
        c = json.loads(args.config.read_text())
        c["simulation"]["physical_surface_network"]["network_backend"] = args.network_backend
        p = Path(c["fluid"]["pvt_path"])
        if not p.is_absolute():
            c["fluid"]["pvt_path"] = str(Path(__file__).resolve().parents[1] / p)
        result = {
            "model": "coupled_blackoil_fim_homogeneous_gathering",
            "device": args.device,
            "network_backend": args.network_backend,
            "cases": [
                run_case(c, p * 1e6, args.days, args.report_days, args.device, args.initial_rs)
                for p in args.outlet_mpa
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
