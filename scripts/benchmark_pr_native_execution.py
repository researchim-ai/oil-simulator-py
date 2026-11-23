"""Paired complete PR transients with native kernels, initialization and audits separated.

The two-phase pipe has finite plenums: it is a comparable workload, not identical
to NeqSim's pressure reservoirs or independently resolved phase momenta.
"""

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.equilibrium_properties import EquilibriumComponent, EquilibriumFluid
from surface.native_execution import native_kernel_execution
from surface.pr_equilibrium_transport import PREquilibriumNode
from surface.pr_momentum_bdf import PRMomentumBDF
from surface.pr_momentum_pipe import PRMomentumPipe
from surface.pr_transport_builder import (
    PREquilibriumPipe,
    build_equilibrium_transport,
    initialize_grouped,
)
from surface.torch_nonlinear import TorchNewton


def build_case(args, *, fused, integrator="backward_euler", refinement=1):
    methane = EquilibriumComponent("methane", 190.564, 4599200, 0.01142, 0.0160428, 35.7, 250, 500)
    pentane = EquilibriumComponent("n-pentane", 469.7, 3370000, 0.251, 0.0721488, 167.2, 250, 500)
    fluid = EquilibriumFluid([methane, pentane], [[0.0, 0.0], [0.0, 0.0]])
    fluid._compiled_full_eos_cpu = fused
    if args.case == "bdf-wave":
        model = PRMomentumPipe(
            fluid,
            length_m=10,
            diameter_m=0.1,
            cells=args.cells,
            wave_speed_bound_m_s=1000,
            boundary_mode="periodic",
            initial_step_s=1e-4,
            maximum_step_s=0.1,
            relative_tolerance=1e-5,
            temperature_tolerance_k=0.0002,
            velocity_tolerance_m_s=0.0001,
            maximum_attempts=10000,
        )
        solver = PRMomentumBDF(model)
        state = solver.initialize(
            [
                5e6 + 1000 * math.cos(2 * math.pi * (i + 0.5) / args.cells)
                for i in range(args.cells)
            ],
            300,
            [1.0, 0.0],
            0.0,
        )
        from check_pr_momentum_bdf import audit

        return (
            lambda: solver.advance(state, args.duration_s),
            audit,
            {"model": solver.to_dict(), "initial": state.to_dict()},
        )
    nodes = [PREquilibriumNode("inlet", 100.0), PREquilibriumNode("export", 100.0)]
    model = build_equilibrium_transport(
        fluid,
        nodes,
        [PREquilibriumPipe("line", "inlet", "export", 10.0, 0.1, args.cells)],
        wave_speed_bound_m_s=2000,
        initial_step_s=1e-4,
        maximum_step_s=0.1 / refinement,
        relative_tolerance=1e-4 / refinement**3,
        temperature_tolerance_k=0.002 / refinement**3,
        velocity_tolerance_m_s=0.001 / refinement**3,
        maximum_attempts=10000,
        dense_limit=768,
    )
    common = {
        "pressure_pa": 5e6,
        "temperature_k": 300.0,
        "mole_fractions": [0.8, 0.2],
        "velocity_m_s": 0.0,
    }
    initial = {"inlet": {**common, "pressure_pa": 5.001e6}, "export": common, "line": common}
    state = initialize_grouped(model, initial)
    solver = model
    if integrator != "backward_euler":
        from surface.pr_collocation import PRCollocation

        solver = PRCollocation(
            model,
            scheme="radau3",
            error_control="embedded" if integrator == "radau3-embedded" else "step_doubling",
        )
    phases = {n: len(v.phase_roots) for n, v in state.nodes}
    if min(phases.values()) != 2:
        raise ValueError("The benchmark requires two occupied phases")
    request = {
        "model": model.to_dict(),
        "initial": {"state": state.to_dict()},
        "schedule": [{"duration_s": args.duration_s}],
    }

    def run():
        result = solver.advance(state, args.duration_s)
        return {
            "config": request,
            "model": model.to_dict(),
            "initial": state.to_dict(),
            "intervals": [result],
            "conservation_residuals": result["conservation_residuals"],
            "total_heat_j": result["total_heat_j"],
            "total_work_j": result["total_work_j"],
        }

    from check_pr_equilibrium_transport import audit

    def check(result):
        return audit(json.loads(json.dumps(request)), json.loads(json.dumps(result)))

    return run, check, {"request": request, "phase_counts": phases}


def reference_errors(result, reference):
    """Compare accepted checkpoints to a refined trajectory, including startup.

    Linear interpolation of the refined reference is explicit in the report;
    it is a temporal-refinement check, not independent physical validation.
    """
    import numpy as np

    mass = np.array([c["molar_mass_kg_mol"] for c in result["model"]["fluid"]["components"]])

    def fields(state):
        values = []
        for row in state["nodes"].values():
            n = np.array(row["component_moles"])
            values.extend(
                [
                    row["pressure_pa"],
                    row["temperature_k"],
                    row["momentum_kg_m_s"] / (n @ mass),
                    *n,
                    row["total_energy_j"],
                ]
            )
        return np.array(values)

    def trajectory(report):
        states = [report["initial"], *[s["state"] for i in report["intervals"] for s in i["steps"]]]
        return np.array([s["time_s"] for s in states]), np.stack([fields(s) for s in states])

    times, values = trajectory(result)
    rt, rv = trajectory(reference)
    expected = np.stack([np.interp(times, rt, column) for column in rv.T], axis=1)
    delta = abs(values - expected)
    size = len(mass) + 4
    scale = 1e-4 * np.maximum(abs(values), abs(expected))
    scale[:, 0::size] += 1.0
    scale[:, 1::size] = 0.002
    scale[:, 2::size] = 0.001
    for k in range(len(mass)):
        scale[:, 3 + k :: size] += 1e-10
    scale[:, size - 1 :: size] += 1e-6
    return {
        "comparison": "accepted checkpoints against linearly interpolated refined Radau trajectory",
        "maximum_requested_error_norm": float((delta / scale).max()),
        "endpoint_requested_error_norm": float((delta[-1] / scale[-1]).max()),
        "maximum_pressure_difference_pa": float(delta[:, 0::size].max()),
        "maximum_temperature_difference_k": float(delta[:, 1::size].max()),
        "maximum_velocity_difference_m_s": float(delta[:, 2::size].max()),
        "reference_steps": len(rt) - 1,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("bdf-wave", "two-phase-pipe"), default="bdf-wave")
    p.add_argument("--duration-s", type=float, default=1.0)
    p.add_argument("--cells", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--without-fused-eos", action="store_true")
    p.add_argument(
        "--reference-refinement",
        type=int,
        default=0,
        help="Optional two-phase refined Radau check; >=2, max step /factor, tolerances /factor^3",
    )
    p.add_argument(
        "--integrator",
        choices=("backward_euler", "radau3", "radau3-embedded"),
        default="backward_euler",
        help="Native path integrator; the paired reference remains backward Euler",
    )
    args = p.parse_args()
    if args.repeats < 1 or args.duration_s <= 0 or args.cells < 1:
        p.error("Positive duration/cells and at least one warm repeat required")
    if args.reference_refinement and (
        args.reference_refinement < 2 or args.case != "two-phase-pipe"
    ):
        p.error("Reference refinement requires two-phase-pipe and a factor >=2")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [
        *sorted((root / "src/surface").glob("*.py")),
        Path(__file__),
        root / "scripts/check_pr_equilibrium_transport.py",
        root / "scripts/audit_pr_collocation.py",
        root / "scripts/check_pr_momentum_bdf.py",
    ]
    hashes = {str(v.relative_to(root)): hashlib.sha256(v.read_bytes()).hexdigest() for v in sources}
    (args.output_dir / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    EquilibriumFluid._compiled_chemical_cpu = True
    TorchNewton._dense_jacobian_mode = "forward"
    all_runs = {}
    for name in ["eager", "native"]:
        runs = []
        engine = None
        from contextlib import ExitStack

        with ExitStack() as scope:
            if name == "native":
                engine = scope.enter_context(
                    native_kernel_execution(progress=lambda e: print(json.dumps(e), flush=True))
                )
            for repeat in range(args.repeats + 1):
                entered = time.perf_counter()
                run, audit, spec = build_case(
                    args,
                    fused=name == "native" and not args.without_fused_eos,
                    integrator=args.integrator if name == "native" else "backward_euler",
                )
                initialization = time.perf_counter() - entered
                print(
                    json.dumps(
                        {
                            "event": "begin",
                            "path": name,
                            "repeat": repeat,
                            "initialization_wall_s": initialization,
                        }
                    ),
                    flush=True,
                )
                entered = time.perf_counter()
                result = run()
                wall = time.perf_counter() - entered
                # Full result is retained before auditing; failures never count
                # as completed, qualified speedups.
                with gzip.open(args.output_dir / f"{name}_{repeat}.json.gz", "wt") as f:
                    json.dump(
                        {"result": result, "wall_s": wall, "initialization_wall_s": initialization},
                        f,
                        allow_nan=False,
                    )
                checked = audit(result)
                intervals = result.get("intervals", [result])
                state = intervals[-1]["state"]
                actual = (
                    state["time_s"]
                    if "time_s" in state
                    else state["history"][-1]["state"]["time_s"]
                )
                start = spec.get(
                    "initial", spec.get("request", {}).get("initial", {}).get("state", {})
                )
                t0 = start.get(
                    "time_s",
                    start.get("history", [{"state": {"time_s": 0.0}}])[-1]["state"]["time_s"],
                )
                if not math.isclose(actual - t0, args.duration_s, rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError("Incomplete interval")
                record = {
                    "repeat": repeat,
                    "wall_s": wall,
                    "initialization_wall_s": initialization,
                    "simulated_s": actual - t0,
                    "independent_audit": checked,
                    "accepted_steps": sum(len(r["steps"]) for r in intervals),
                    "rejected_attempts": sum(r.get("rejected_attempts", 0) for r in intervals),
                    "fallback_intervals": sum(r.get("fallback_intervals", 0) for r in intervals),
                    "collocation_nonlinear_solves": sum(
                        r.get("collocation_nonlinear_solves", 0) for r in intervals
                    ),
                    "execution": engine.statistics() if engine else None,
                }
                runs.append(record)
                print(json.dumps({"event": "completed", "path": name, **record}), flush=True)
                (args.output_dir / "request.json").write_text(
                    json.dumps(spec, indent=2, allow_nan=False) + "\n"
                )
        all_runs[name] = {
            "cold": runs[0],
            "warm": runs[1:],
            "warm_median_wall_s": statistics.median(r["wall_s"] for r in runs[1:]),
        }
    differences = {}
    with gzip.open(args.output_dir / f"eager_{args.repeats}.json.gz", "rt") as handle:
        a = json.load(handle)["result"]
    with gzip.open(args.output_dir / f"native_{args.repeats}.json.gz", "rt") as handle:
        b = json.load(handle)["result"]
    if args.case == "bdf-wave":
        for key in ["pressure_pa", "temperature_k", "velocity_m_s"]:
            differences[key] = max(
                abs(x - y) for x, y in zip(a["final_summary"][key], b["final_summary"][key])
            )
    else:
        x, y = a["intervals"][-1]["state"], b["intervals"][-1]["state"]
        for key in ["pressure_pa", "temperature_k", "total_energy_j", "momentum_kg_m_s"]:
            differences[key] = max(abs(x["nodes"][n][key] - y["nodes"][n][key]) for n in x["nodes"])
    accuracy = None
    if args.reference_refinement:
        # Reference values/derivatives use the original eager operator, not
        # the native engine being qualified, and no fused full-EOS kernel.
        run, check, _ = build_case(
            args,
            fused=False,
            integrator="radau3-embedded",
            refinement=args.reference_refinement,
        )
        entered = time.perf_counter()
        refined = run()
        wall = time.perf_counter() - entered
        with gzip.open(args.output_dir / "refined.json.gz", "wt") as handle:
            json.dump({"result": refined, "wall_s": wall}, handle, allow_nan=False)
        checked = check(refined)
        accuracy = {
            "refinement_factor": args.reference_refinement,
            "execution": "eager_derivatives_unfused_full_eos",
            "independent_audit": checked,
            "eager": reference_errors(a, refined),
            "native": reference_errors(b, refined),
        }
        print(json.dumps({"event": "accuracy", **accuracy}), flush=True)
    if any(hashlib.sha256((root / v).read_bytes()).hexdigest() != h for v, h in hashes.items()):
        raise RuntimeError("Sources changed during measurement")
    from surface.torch_backend import runtime_fingerprint

    report = {
        "scope": "Paired CPU standalone pipes; not a coupled-facility or identical NeqSim model comparison",
        "case": args.case,
        "duration_s": args.duration_s,
        "cells": args.cells,
        "native_integrator": args.integrator,
        "runtime": runtime_fingerprint(),
        "policy": {
            "compiled_chemical_cpu_both_paths": True,
            "compiled_full_eos_native": not args.without_fused_eos,
            "dense_ad": "forward",
            "tolerances_relaxed": False,
        },
        "temporal_refinement": accuracy,
        "runs": all_runs,
        "observed_speedup": all_runs["eager"]["warm_median_wall_s"]
        / all_runs["native"]["warm_median_wall_s"],
        "endpoint_differences": differences,
        "source_sha256": hashes,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
