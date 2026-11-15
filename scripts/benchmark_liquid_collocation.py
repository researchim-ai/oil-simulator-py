"""Alternating CPU work/precision qualification of opt-in Torch Gauss4.

The comparator is the production adaptive midpoint, with the same physical
model, controls, maximum step and configured observable tolerances.
"""

import argparse
import gzip
import hashlib
import json
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
from check_liquid_transient import coupled_model, coupled_rhs

from surface import LiquidCollocation, LiquidTransientNetwork


def configured():
    data = coupled_model(0.02).to_dict()
    data.update(
        temporal_error_control=True,
        relative_tolerance=1e-8,
        pressure_tolerance_pa=0.1,
        flow_tolerance_m3_s=1e-11,
        speed_tolerance_rad_s=1e-6,
    )
    model = LiquidTransientNetwork.from_dict(data)
    state = model.initialize(
        {"h": 550000.0, "t": 500000.0}, {"line": 0.002}, {"a": 100.0, "b": 110.0}
    )
    return model, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("positive pair count required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("explicitly set CUDA_VISIBLE_DEVICES='' for CPU qualification")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__).resolve()]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    initial = np.array([550000.0, 500000.0, 0.002, 100.0, 110.0])
    reference = solve_ivp(
        coupled_rhs,
        (0.0, 1.0),
        initial,
        method="DOP853",
        rtol=1e-12,
        atol=[1e-7, 1e-7, 1e-15, 1e-11, 1e-11],
        dense_output=True,
    )
    assert reference.success
    budget = np.array([1.0, 1.0, 1e-8, 1e-4, 1e-4])

    def run(method, duration):
        model, state = configured()
        integrator = LiquidCollocation(model) if method == "gauss4" else None
        start = time.perf_counter()
        if integrator is not None:
            report = integrator.advance(state, duration, motor_torques_nm={"a": 5.0, "b": 7.0})
        else:
            report = model.advance(state, duration, motor_torques_nm={"a": 5.0, "b": 7.0})
        elapsed = time.perf_counter() - start
        return elapsed, report, integrator

    # Both root solvers and AD shapes are warmed outside measured runs.
    run("midpoint", 0.005)
    run("gauss4", 0.005)
    rows = []
    for pair in range(args.pairs):
        for method in ("midpoint", "gauss4") if pair % 2 == 0 else ("gauss4", "midpoint"):
            elapsed, report, integrator = run(method, 1.0)
            times, states = [0.0], [initial.tolist()]
            for step in report["steps"]:
                times.append(step.get("end_time_s", step["time_s"] + step["dt_s"]))
                if integrator is not None:
                    e = integrator.e
                    z = np.array(step["scaled_state"])
                    states.append(
                        (z[[0, 1, 2, 6, 7]] * np.r_[[e.ps] * 2, e.qs, [100, 110]]).tolist()
                    )
                else:
                    states.append(
                        [step["final_pressures_pa"][n] for n in ("h", "t")]
                        + [step["final_pipe_volume_rates_m3_s"]["line"]]
                        + [step["final_speeds_rad_s"][n] for n in ("a", "b")]
                    )
            times, states = np.array(times), np.array(states)
            error = abs(states - reference.sol(times).T).max(axis=0)
            row = {
                "pair": pair,
                "method": method,
                "elapsed_s": elapsed,
                "accepted_steps": len(report["steps"]),
                "maximum_observable_errors": error.tolist(),
                "maximum_budget_fraction": float(max(error / budget)),
                "max_energy_residual_j": max(abs(s["energy_residual_j"]) for s in report["steps"]),
                "max_mass_residual_kg": max(abs(s["mass_residual_kg"]) for s in report["steps"]),
                "median_step_s": float(np.median(np.diff(times))),
                "minimum_step_s": float(min(np.diff(times))),
                "maximum_step_s": float(max(np.diff(times))),
            }
            if integrator is not None:
                row.update(
                    nonlinear_evaluations=report["nonlinear_evaluations"],
                    jacobian_builds=report["jacobian_builds"],
                    rejected_steps=report["rejected_steps"],
                    nonlinear_solves=report["attempts"],
                )
            with gzip.open(args.output / f"pair{pair}_{method}.json.gz", "wt") as f:
                json.dump(
                    {
                        "report": report,
                        "times_s": times.tolist(),
                        "physical_states": states.tolist(),
                    },
                    f,
                    allow_nan=False,
                )
            assert np.all(error <= budget), row
            rows.append(row)
            print(json.dumps(row), flush=True)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in sources.items())
    medians = {
        m: statistics.median(r["elapsed_s"] for r in rows if r["method"] == m)
        for m in ("midpoint", "gauss4")
    }
    summary = {
        "scope": "Single-phase liquid/pump DAE; NOT the multicomponent PR facility",
        "cpu_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "reference": "Independent eliminated ODE, DOP853 rtol=1e-12",
        "error_sampling": "Every accepted endpoint; no continuous-output certification",
        "observable_error_budget": budget.tolist(),
        "results": rows,
        "median_elapsed_s": medians,
        "median_speedup": medians["midpoint"] / medians["gauss4"],
        "source_sha256": sources,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"medians_s": medians, "speedup": summary["median_speedup"]}), flush=True)


if __name__ == "__main__":
    main()
