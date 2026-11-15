"""CPU work/precision for production midpoint heat transport vs shared Gauss."""

import argparse
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_liquid_thermal_collocation import ConservativeReference, case

from surface import ThermalLiquidCollocation
from surface.torch_liquid import TorchLiquidEquations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--baseline-tolerance-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.pairs < 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("positive pair count and CUDA_VISIBLE_DEVICES='' required")
    if (
        not math.isfinite(args.baseline_tolerance_scale)
        or not 0 < args.baseline_tolerance_scale <= 1
    ):
        parser.error("baseline tolerance scale must be in (0,1]")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__).resolve(),
        ROOT / "scripts/check_liquid_thermal_collocation.py",
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    _, heat, p, t = case()
    reference = ConservativeReference(heat)
    y0 = reference.encode(p, t)
    solution = solve_ivp(
        reference, (0.0, 1.0), y0, method="DOP853", rtol=1e-12, atol=1e-9, dense_output=True
    )
    assert solution.success

    def run(method, duration):
        m, h, physical, thermal = case()
        if method == "midpoint" and args.baseline_tolerance_scale != 1:
            for name in (
                "relative_tolerance",
                "pressure_tolerance_pa",
                "flow_tolerance_m3_s",
                "speed_tolerance_rad_s",
            ):
                setattr(m, name, getattr(m, name) * args.baseline_tolerance_scale)
            # Reinitialize after changing only numerical settings, before any
            # kernel exists. Physical coefficients and boundary data match.
            physical = m.initialize(
                dict(physical.pressures_pa),
                dict(physical.pipe_volume_rates_m3_s),
                dict(physical.speeds_rad_s),
            )
            thermal = h.initialize(
                physical, dict(thermal.temperatures_k), dict(thermal.wall_temperatures_k)
            )
        solver = ThermalLiquidCollocation(h) if method == "gauss4" else None
        start = time.perf_counter()
        if solver:
            result = solver.advance(
                physical, thermal, duration, motor_torques_nm={"a": 5.0, "b": 7.0}
            )
        else:
            hydraulics = m.advance(physical, duration, motor_torques_nm={"a": 5.0, "b": 7.0})
            caloric = h.advance_records(thermal, hydraulics["steps"])
            result = {
                "state": {"hydraulics": hydraulics["state"], "heat": caloric["state"]},
                "hydraulics": hydraulics,
                "caloric": caloric,
                "heat_model": h.to_dict(),
                "initial_heat": thermal.to_dict(),
            }
        return time.perf_counter() - start, result, h

    for method in ("midpoint", "gauss4"):
        run(method, 0.005)
    results = []
    for pair in range(args.pairs):
        for method in ("midpoint", "gauss4") if pair % 2 == 0 else ("gauss4", "midpoint"):
            elapsed, result, heat = run(method, 1.0)
            e = TorchLiquidEquations(heat.physics)
            ht, hy = [], []
            for step in result["hydraulics"]["steps"]:
                ht.append(step.get("end_time_s", step["time_s"] + step["dt_s"]))
                if method == "gauss4":
                    z = np.array(step["scaled_state"])
                    hy.append(z[[0, 1, 2, 6, 7]] * np.r_[[e.ps] * 2, e.qs, [100, 110]])
                else:
                    hy.append(
                        [step["final_pressures_pa"][n] for n in ("h", "t")]
                        + [step["final_pipe_volume_rates_m3_s"]["line"]]
                        + [step["final_speeds_rad_s"][n] for n in ("a", "b")]
                    )
            expected_h = np.array([reference.observe(solution.sol(v))[:5] for v in ht])
            hyd_error = abs(np.array(hy) - expected_h).max(0)
            caloric = result["thermal_steps"] if method == "gauss4" else result["caloric"]["steps"]
            tt, temperatures = [], []
            for step in caloric:
                tt.append(step.get("end_time_s", step["time_s"] + step["dt_s"]))
                tk = step["temperatures_k"] if method == "gauss4" else step["final_temperatures_k"]
                tw = (
                    step["wall_temperatures_k"]
                    if method == "gauss4"
                    else step["final_wall_temperatures_k"]
                )
                temperatures.append([tk[n] for n in heat.names] + [tw[n] for n in heat.wall_names])
            ntemp = len(heat.names) + len(heat.wall_names)
            expected_t = np.array([reference.observe(solution.sol(v))[5 : 5 + ntemp] for v in tt])
            thermal_error = abs(np.array(temperatures) - expected_t).max(0)
            final_mass = result["state"]["heat"]["masses_kg"]["in"]
            mass_error = abs(final_mass - reference.decode(solution.y[:, -1])[0]["in"])
            combined = (
                [s["combined_energy_residual_j"] for s in caloric]
                if method == "gauss4"
                else [s["combined_energy_residual_j"] for s in result["caloric"]["audits"]]
            )
            row = {
                "pair": pair,
                "method": method,
                "elapsed_s": elapsed,
                "hydraulic_steps": len(ht),
                "thermal_steps": len(tt),
                "hydraulic_maximum_errors": hyd_error.tolist(),
                "maximum_temperature_error_k": float(max(thermal_error)),
                "temperature_errors_k": thermal_error.tolist(),
                "source_inventory_error_kg": mass_error,
                "max_combined_energy_residual_j": max(map(abs, combined)),
                "hydraulic_tolerance_scale": args.baseline_tolerance_scale
                if method == "midpoint"
                else 1.0,
            }
            if method == "gauss4":
                row.update(
                    rejected_steps=result["hydraulics"]["rejected_steps"],
                    thermal_linear_solves=result["thermal_linear_solves"],
                )
            row["within_requested_budgets"] = bool(
                np.all(hyd_error <= [1.0, 1.0, 1e-8, 1e-4, 1e-4])
                and max(thermal_error) <= 1e-4
                and mass_error < 1e-7
            )
            with gzip.open(args.output / f"pair{pair}_{method}.json.gz", "wt") as stream:
                json.dump(
                    {
                        "report": result,
                        "hydraulic_times_s": ht,
                        "hydraulic_states": np.array(hy).tolist(),
                        "temperature_times_s": tt,
                        "temperature_states": temperatures,
                    },
                    stream,
                    allow_nan=False,
                )
            results.append(row)
            print(json.dumps(row), flush=True)
            if method == "gauss4" or args.baseline_tolerance_scale < 1:
                assert row["within_requested_budgets"], row
    assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == v for n, v in hashes.items())
    medians = {
        m: statistics.median(r["elapsed_s"] for r in results if r["method"] == m)
        for m in ("midpoint", "gauss4")
    }
    summary = {
        "scope": "Constant-property liquid hydraulics plus finite source, pipe caloric stocks and wall. NOT the PR facility.",
        "error_sampling": "All accepted endpoints; finite source mass at the final endpoint",
        "reference": "Independent conservative ODE, DOP853 rtol=1e-12 atol=1e-9",
        "cpu_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "results": results,
        "median_elapsed_s": medians,
        "speedup": medians["midpoint"] / medians["gauss4"],
        "source_sha256": hashes,
        "baseline_tolerance_scale": args.baseline_tolerance_scale,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"medians_s": medians, "speedup": summary["speedup"]}), flush=True)


if __name__ == "__main__":
    main()
