"""CPU verification of actual water FIM/surface pressure feedback, from identical initial states."""

import argparse
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.fluid import Fluid
from simulator.reservoir import Reservoir
from simulator.simulation import Simulator
from simulator.well import WellManager


def run_case(config, outlet_pa, outlet_node, days, device_name="cpu"):
    cfg = copy.deepcopy(config)
    surface = cfg["simulation"].get("physical_surface_network")
    if surface is None:
        raise ValueError("Configuration requires physical_surface_network")
    matches = [node for node in surface["network"]["nodes"] if node["name"] == outlet_node]
    if len(matches) != 1 or matches[0].get("pressure_pa") is None:
        raise ValueError("Outlet must name an existing fixed-pressure surface node")
    matches[0]["pressure_pa"] = outlet_pa
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; no implicit CPU fallback")
        reservoir = Reservoir(cfg["reservoir"], device=device)
        fluid = Fluid(cfg["fluid"], reservoir, device=device)
        wells = WellManager(cfg["wells"], reservoir)
        sim = Simulator(reservoir, fluid, wells, cfg["simulation"], device=device)
        solver = sim.fim_solver
        initial_mass = sum(
            float(v.sum())
            for v in solver.calc_masses(fluid.pressure.reshape(-1), fluid.s_w.reshape(-1))
        )
        steps = []
        previous_time = [0.0]

        def accepted(time_days):
            dt = (time_days - previous_time[0]) * 86400
            previous_time[0] = time_days
            report = copy.deepcopy(solver.wells.surface_report)
            rate = sum(w["water_mass_rate_kg_s"] for w in report["wells"].values())
            steps.append(
                {
                    "time_days": time_days,
                    "dt_s": dt,
                    "mass_rate_kg_s": rate,
                    "produced_mass_kg": rate * dt,
                    "net_well_mass_kg": float(
                        solver._last_well_sys["q_o"].sum() + solver._last_well_sys["q_w"].sum()
                    )
                    * dt,
                    "newton_iterations": solver.last_newton_iters,
                    "surface": report,
                }
            )

        sim.substep_hook = accepted
        if not sim.run_step(days * 86400):
            raise RuntimeError("Coupled FIM interval did not converge")
        final_mass = sum(
            float(v.sum())
            for v in solver.calc_masses(fluid.pressure.reshape(-1), fluid.s_w.reshape(-1))
        )
    produced = sum(step["produced_mass_kg"] for step in steps)
    net_withdrawal = sum(step["net_well_mass_kg"] for step in steps)
    return {
        "outlet_pressure_pa": outlet_pa,
        "execution": {"reservoir_device": str(device), "network_device": "cpu",
                      "schur_device": str(solver._last_well_sys["D_full"].device)},
        "elapsed_days": sim.current_time / 86400,
        "accepted_substeps": steps,
        "produced_water_mass_kg": produced,
        "reservoir_mass_loss_kg": initial_mass - final_mass,
        "net_well_mass_withdrawal_kg": net_withdrawal,
        "mass_balance_error_kg": initial_mass - final_mass - net_withdrawal,
        "final_pressure_pa": fluid.pressure.reshape(-1).tolist(),
        "final_water_saturation": fluid.s_w.reshape(-1).tolist(),
        "solver_log": log.getvalue(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--outlet-mpa", type=float, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--outlet-node", default="outlet")
    parser.add_argument("--days", type=float, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    try:
        if not 0 < args.days < float("inf") or any(
            not 0 < p < float("inf") for p in args.outlet_mpa
        ):
            raise ValueError("Days and outlet pressures must be finite and positive")
        cfg = json.loads(args.config.read_text())
        result = {
            "model": "water_only_fim_physical_surface_coupling",
            "cases": [run_case(cfg, p * 1e6, args.outlet_node, args.days, args.device) for p in args.outlet_mpa],
        }
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end="")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
