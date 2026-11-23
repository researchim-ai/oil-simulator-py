"""Compare the same single-phase PR/Rusanov method-of-lines with OpenModelica BDF.

This is an integrator/implementation experiment, not a Modelica.Fluid multiphase
product benchmark. No SciPy or external integrator enters production code.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import surface.pr_momentum_pipe as pipe_module
from surface.equilibrium_properties import EquilibriumComponent, EquilibriumFluid
from surface.peng_robinson import R
from surface.pr_momentum_pipe import PRMomentumPipe


def modelica_source(fluid, concentrations, length, diameter):
    c = fluid.components[0]
    k = float(fluid.kappa[0])
    n = len(concentrations)
    initial = ",".join(format(v, ".17g") for v in concentrations)
    return f"""model PRPeriodicPipe
  parameter Integer N={n};
  parameter Real L={length:.17g}, D={diameter:.17g}, dx=L/N;
  parameter Real gasR={R:.17g}, mw={c.molar_mass_kg_mol:.17g};
  parameter Real b={c.b:.17g}, a0={c.attraction_coefficient:.17g};
  parameter Real Tc={c.critical_temperature_k:.17g}, kappa={k:.17g};
  parameter Real cp={c.ideal_cp_j_mol_k:.17g}, Tref={fluid.tref:.17g};
  parameter Real c0[N]={{{initial}}};
  Real concentration[N](each nominal=2000, start=c0, each fixed=true);
  Real T[N](each nominal=300, each start=300, each fixed=true);
  Real velocity[N](each nominal=1, each start=0, each fixed=true);
  Real pressure[N](each nominal=5e6);
  Real rootAlpha[N], attraction[N], attractionT[N], molarVolume[N];
  Real molarEnergy[N], U[N,3], F[N,3], face[N,3];
  Real rhs[N,3];
  Real totalMoles, totalMomentum, totalEnergy;
equation
  for i in 1:N loop
    rootAlpha[i] = 1+kappa*(1-sqrt(T[i]/Tc));
    attraction[i] = a0*rootAlpha[i]^2;
    attractionT[i] = -a0*kappa*rootAlpha[i]/(Tc*sqrt(T[i]/Tc));
    molarVolume[i] = 1/concentration[i];
    pressure[i] = gasR*T[i]/(molarVolume[i]-b)
      -attraction[i]/(molarVolume[i]^2+2*b*molarVolume[i]-b^2);
    molarEnergy[i] = cp*(T[i]-Tref)-gasR*T[i]
      +(T[i]*attractionT[i]-attraction[i])/(2*sqrt(2)*b)
       *log((molarVolume[i]+(1+sqrt(2))*b)/(molarVolume[i]+(1-sqrt(2))*b));
    U[i,1] = concentration[i];
    U[i,2] = mw*concentration[i]*velocity[i];
    U[i,3] = concentration[i]*molarEnergy[i]+0.5*mw*concentration[i]*velocity[i]^2;
    F[i,1] = U[i,1]*velocity[i];
    F[i,2] = U[i,2]*velocity[i]+pressure[i];
    F[i,3] = (U[i,3]+pressure[i])*velocity[i];
    for j in 1:3 loop
      face[i,j] = (F[i,j]+F[if i==N then 1 else i+1,j])/2
        -500*(U[if i==N then 1 else i+1,j]-U[i,j])/2;
      rhs[i,j] = (face[if i==1 then N else i-1,j]-face[i,j])/dx;
      der(U[i,j]) = rhs[i,j];
    end for;
  end for;
  totalMoles=sum(U[:,1])*dx*Modelica.Constants.pi*D^2/4;
  totalMomentum=sum(U[:,2])*dx*Modelica.Constants.pi*D^2/4;
  totalEnergy=sum(U[:,3])*dx*Modelica.Constants.pi*D^2/4;
end PRPeriodicPipe;
"""


def read_csv(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return [{k: float(v) for k, v in row.items()} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--omc-runtime",
        type=Path,
        required=True,
        help="Extracted OpenModelica prefix with root/usr and env.sh",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--duration-s", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--newton", choices=("legacy", "enhanced"), default="enhanced")
    parser.add_argument("--integrator", choices=("be", "bdf2"), default="be")
    args = parser.parse_args()
    if args.integrator == "bdf2" and args.newton != "enhanced":
        parser.error("The BDF2 adapter uses the enhanced Newton")
    if args.newton == "enhanced":
        from surface.torch_nonlinear import TorchNewton
    else:
        from surface.torch_linear import TorchNewton
    pipe_module.TorchNewton = TorchNewton
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    build_dir = Path(tempfile.mkdtemp(prefix="oil-pr-bdf-"))
    fluid = EquilibriumFluid(
        [
            EquilibriumComponent("methane", 190.564, 4599200, 0.01142, 0.0160428, 35.7, 250, 500),
            EquilibriumComponent("nitrogen", 126.192, 3395800, 0.0372, 0.0280134, 29.1, 250, 500),
        ],
        [[0.0, 0.0], [0.0, 0.0]],
    )
    pipe = PRMomentumPipe(
        fluid,
        length_m=10,
        diameter_m=0.1,
        cells=args.cells,
        wave_speed_bound_m_s=500,
        boundary_mode="periodic",
        initial_step_s=1e-4,
        maximum_step_s=0.01,
        relative_tolerance=args.relative_tolerance,
        temperature_tolerance_k=0.002 * (args.relative_tolerance / 1e-4),
        velocity_tolerance_m_s=0.001 * (args.relative_tolerance / 1e-4),
        maximum_attempts=10000,
    )
    pressures = [
        5e6 + 1000 * math.cos(2 * math.pi * (i + 0.5) / args.cells) for i in range(args.cells)
    ]
    initial = pipe.initialize(pressures, 300, [1.0, 0.0], 0.0)
    if args.integrator == "bdf2":
        from surface.pr_momentum_bdf import PRMomentumBDF, PRMomentumBDFState

        integrator = PRMomentumBDF(pipe)
        initial_bdf = PRMomentumBDFState(
            (integrator._first(initial),), None, integrator.integrator_id
        )
    concentrations = [row[0] / pipe.volume for row in initial.component_moles]
    (out / "PRPeriodicPipe.mo").write_text(modelica_source(fluid, concentrations, 10, 0.1))
    # No library installation; use an already available read-only standard library.
    libraries = list((args.omc_runtime / "libraries").glob("*/Modelica/package.mo"))
    if len(libraries) != 1:
        raise RuntimeError("Expected one existing Modelica standard library")
    library = libraries[0].parent.parent
    mos = f'''loadFile("{library}/Complex.mo", uses=false);
loadFile("{library}/ModelicaServices/package.mo", uses=false);
loadFile("{library}/Modelica/package.mo", uses=false);
loadFile("{out}/PRPeriodicPipe.mo");
setCommandLineOptions("--generateSymbolicJacobian");
setCompiler("gcc");
setCXXCompiler("g++");
buildModel(PRPeriodicPipe, stopTime={args.duration_s}, numberOfIntervals=100, tolerance=1e-8, outputFormat="csv");
getErrorString();
'''
    (out / "build.mos").write_text(mos)
    start = time.perf_counter()
    compiled = subprocess.run(
        ["bash", str(args.omc_runtime.resolve() / "env.sh"), str(out / "build.mos")],
        cwd=build_dir,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    compile_s = time.perf_counter() - start
    (out / "build.log").write_text(compiled.stdout + compiled.stderr)
    binary = build_dir / "PRPeriodicPipe"
    if compiled.returncode or not binary.exists():
        raise RuntimeError(f"OpenModelica compilation failed; inspect {out / 'build.log'}")
    env = dict(os.environ)
    prefix = args.omc_runtime.resolve() / "root/usr"
    env["LD_LIBRARY_PATH"] = (
        f"{prefix}/lib/x86_64-linux-gnu:{prefix}/lib/x86_64-linux-gnu/omc:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    modelica_runs = []
    for solver, tolerance in [("dassl", 1e-8), ("ida", 1e-8), ("ida", 1e-10)]:
        samples = []
        for repeat in range(args.repeats + 1):
            csv_path = out / f"{solver}_{tolerance}_{repeat}.csv"
            start = time.perf_counter()
            result = subprocess.run(
                [
                    str(binary),
                    "-s=" + solver,
                    f"-tolerance={tolerance}",
                    "-r=" + str(csv_path),
                    "-lv=LOG_STATS",
                ],
                cwd=build_dir,
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            elapsed = time.perf_counter() - start
            (out / f"{solver}_{tolerance}_{repeat}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError("OpenModelica integration failed")
            rows = read_csv(csv_path)
            with gzip.open(str(csv_path) + ".gz", "wb") as compressed:
                compressed.write(csv_path.read_bytes())
            csv_path.unlink()
            if not math.isclose(rows[-1]["time"], args.duration_s, abs_tol=1e-12):
                raise RuntimeError("OpenModelica integration was incomplete")
            for row in rows:
                for i in range(1, args.cells + 1):
                    if not (
                        250 < row[f"T[{i}]"] < 500
                        and 0 < row[f"pressure[{i}]"] < 1e9
                        and row[f"concentration[{i}]"] > 0
                        and math.isfinite(row[f"velocity[{i}]"])
                    ):
                        raise RuntimeError("Native trajectory left the single-phase test domain")
            stats = {}
            for key, pattern in [
                ("steps", r"(\d+) steps taken"),
                ("function_evaluations", r"(\d+) calls of functionODE"),
                ("jacobian_evaluations", r"(\d+) evaluations of jacobian"),
                ("error_test_failures", r"(\d+) error test failures"),
            ]:
                found = re.search(pattern, result.stdout)
                if found:
                    stats[key] = int(found[1])
            samples.append(
                {
                    "process_wall_s": elapsed,
                    "initial": rows[0],
                    "final": rows[-1],
                    "solver_statistics": stats,
                }
            )
        modelica_runs.append(
            {
                "solver": solver,
                "relative_tolerance": tolerance,
                "cold": samples[0],
                "warm": samples[1:],
                "median_process_wall_s": statistics.median(
                    s["process_wall_s"] for s in samples[1:]
                ),
            }
        )
    torch_runs = []
    for repeat in range(args.repeats + 1):
        work = {
            "newton_solves": 0,
            "jacobian_builds": 0 if args.newton == "enhanced" else None,
            "newton_wall_s": 0.0,
            "nonlinear_evaluations_total": 0,
            "physical_validation_calls": 0,
            "physical_validation_wall_s": 0.0,
        }
        original_solve = TorchNewton.solve
        original_validate = pipe._validate_data

        def counted_solve(solver, *arguments, work=work, original_solve=original_solve, **kwargs):
            entered = time.perf_counter()
            value = original_solve(solver, *arguments, **kwargs)
            work["newton_solves"] += 1
            work["nonlinear_evaluations_total"] += value[-1]
            if work["jacobian_builds"] is not None:
                work["jacobian_builds"] += solver.jacobian_builds
            work["newton_wall_s"] += time.perf_counter() - entered
            return value

        def counted_validate(*arguments, work=work, original_validate=original_validate, **kwargs):
            entered = time.perf_counter()
            value = original_validate(*arguments, **kwargs)
            work["physical_validation_calls"] += 1
            work["physical_validation_wall_s"] += time.perf_counter() - entered
            return value

        TorchNewton.solve = counted_solve
        pipe._validate_data = counted_validate
        start = time.perf_counter()
        try:
            result = (
                integrator.advance(initial_bdf, args.duration_s)
                if args.integrator == "bdf2"
                else pipe.advance(initial, args.duration_s)
            )
        finally:
            TorchNewton.solve = original_solve
            pipe._validate_data = original_validate
        wall = time.perf_counter() - start
        if args.integrator == "bdf2":
            from check_pr_momentum_bdf import audit

            independent = audit(result)
        else:
            from check_pr_momentum_pipe import audit

            request = {
                "model": pipe.to_dict(),
                "initial": {
                    "pressure_pa": pressures,
                    "temperature_k": 300,
                    "mole_fractions": [1.0, 0.0],
                    "velocity_m_s": 0.0,
                },
                "schedule": [{"duration_s": args.duration_s}],
            }
            independent = audit(
                request,
                {
                    "config": request,
                    "model": pipe.to_dict(),
                    "initial": initial.to_dict(),
                    "intervals": [result],
                },
            )
        with gzip.open(out / f"torch_{args.integrator}_{repeat}.json.gz", "wt") as handle:
            json.dump(result, handle, allow_nan=False)
        torch_runs.append(
            {
                "wall_s": wall,
                "accepted_steps": len(result["steps"]),
                "attempts": result["attempts"],
                "nonlinear_evaluations": sum(
                    s.get(
                        "nonlinear_evaluations", s.get("work", {}).get("nonlinear_evaluations", 0)
                    )
                    for s in result["steps"]
                ),
                "conservation_residuals": result["conservation_residuals"],
                "final": result["final_summary"],
                "work": work,
                "independent_audit": independent,
            }
        )
        print(
            json.dumps(
                {
                    "engine": "torch",
                    "repeat": repeat,
                    "wall_s": wall,
                    "attempts": result["attempts"],
                }
            ),
            flush=True,
        )
    # Cross-check independently written PR caloric formulas and flux divergence.
    data = pipe._decode(
        fluid.tensor(initial.solution_variables), pipe._support(initial.component_moles)
    )
    faces, _ = pipe._faces(data, (None, None))
    rhs = ((faces[:-1] - faces[1:]) / pipe.dx).detach().cpu().numpy()[:, [0, -2, -1]]
    reference = modelica_runs[-1]["warm"][-1]
    initial_row = reference["initial"]
    initial_rhs = np.array(
        [[initial_row[f"rhs[{i + 1},{j + 1}]"] for j in range(3)] for i in range(args.cells)]
    )
    checks = {"initial_rhs_max_abs_difference": np.abs(rhs - initial_rhs).max(axis=0).tolist()}
    checks["initial_rhs_scaled_difference"] = (
        np.abs(rhs - initial_rhs).max(axis=0) / np.maximum(1, np.abs(rhs).max(axis=0))
    ).tolist()
    snapshot_checks = []
    for row in (rows[0], rows[len(rows) // 2], rows[-1]):
        variables = fluid.tensor(
            [
                [
                    math.log(row[f"pressure[{i}]"]),
                    row[f"T[{i}]"] / fluid.tref,
                    row[f"velocity[{i}]"] / pipe.maximum_velocity_m_s,
                ]
                for i in range(1, args.cells + 1)
            ]
        ).flatten()
        decoded = pipe._decode(variables, pipe._support(initial.component_moles))
        face, _ = pipe._faces(decoded, (None, None))
        actual = ((face[:-1] - face[1:]) / pipe.dx).detach().cpu().numpy()[:, [0, -2, -1]]
        expected = np.array(
            [[row[f"rhs[{i},{j}]"] for j in range(1, 4)] for i in range(1, args.cells + 1)]
        )
        conserved = decoded["u"].detach().cpu().numpy()[:, [0, -2, -1]]
        native = np.array(
            [[row[f"U[{i},{j}]"] for j in range(1, 4)] for i in range(1, args.cells + 1)]
        )
        snapshot_checks.append(
            {
                "time_s": row["time"],
                "max_scaled_rhs_difference": (
                    np.abs(actual - expected).max(axis=0)
                    / np.maximum(1, np.abs(expected).max(axis=0))
                ).tolist(),
                "max_scaled_conserved_difference": (
                    np.abs(conserved - native).max(axis=0)
                    / np.maximum(1, np.abs(native).max(axis=0))
                ).tolist(),
            }
        )
    checks["operator_snapshots"] = snapshot_checks
    if (
        max(
            max(s["max_scaled_rhs_difference"] + s["max_scaled_conserved_difference"])
            for s in snapshot_checks
        )
        > 1e-7
    ):
        raise RuntimeError("Native and Torch spatial/caloric operators differ")
    reference_final = reference["final"]
    comparisons = []
    for run in modelica_runs:
        end = run["warm"][-1]["final"]
        differences = {
            name: max(
                abs(end[f"{name}[{i + 1}]"] - reference_final[f"{name}[{i + 1}]"])
                for i in range(args.cells)
            )
            for name in ["pressure", "T", "velocity"]
        }
        ledger = {
            name: end[name] - run["warm"][-1]["initial"][name]
            for name in ["totalMoles", "totalMomentum", "totalEnergy"]
        }
        comparisons.append(
            {
                "solver": run["solver"],
                "tolerance": run["relative_tolerance"],
                "max_difference_from_tight_ida": differences,
                "conservation_changes": ledger,
            }
        )
    last = torch_runs[-1]["final"]
    checks["torch_max_difference_from_tight_ida"] = {
        key: max(abs(last[key][i] - reference_final[f"{name}[{i + 1}]"]) for i in range(args.cells))
        for key, name in [
            ("pressure_pa", "pressure"),
            ("temperature_k", "T"),
            ("velocity_m_s", "velocity"),
        ]
    }
    # Compare accepted trajectory points as well as the possibly fortuitous
    # final acoustic phase. Native output is interpolated with four-point
    # cubic Lagrange polynomials, outside all measured integration times.
    unique = {row["time"]: row for row in rows}
    times = np.array(sorted(unique))
    trajectory_errors = {"pressure_pa": 0.0, "temperature_k": 0.0, "velocity_m_s": 0.0}
    for step in result["steps"]:
        state = step["point"]["state"] if args.integrator == "bdf2" else step["state"]
        time_s = state["time_s"]
        k = max(0, min(len(times) - 4, int(np.searchsorted(times, time_s)) - 2))
        nodes = times[k : k + 4]
        weights = np.array(
            [
                math.prod((time_s - nodes[j]) / (nodes[i] - nodes[j]) for j in range(4) if j != i)
                for i in range(4)
            ]
        )
        y = np.array(state["solution_variables"]).reshape(args.cells, 3)
        physical = [np.exp(y[:, 0]), y[:, 1] * fluid.tref, y[:, 2] * pipe.maximum_velocity_m_s]
        for (key, name), actual in zip(
            [("pressure_pa", "pressure"), ("temperature_k", "T"), ("velocity_m_s", "velocity")],
            physical,
        ):
            expected = weights @ np.array(
                [[unique[t][f"{name}[{i + 1}]"] for i in range(args.cells)] for t in nodes]
            )
            trajectory_errors[key] = max(
                trajectory_errors[key], float(np.max(np.abs(actual - expected)))
            )
    checks["max_at_saved_steps_vs_cubic_tight_ida"] = trajectory_errors
    root = Path(__file__).resolve().parents[1]
    sources = list((root / "src/surface").glob("*.py")) + [
        Path(__file__),
        out / "PRPeriodicPipe.mo",
        root / "scripts/check_pr_momentum_bdf.py",
        root / "scripts/check_pr_momentum_pipe.py",
        root / "scripts/check_stratified_pr.py",
        root / "scripts/check_stratified_pr_flow.py",
        root / "scripts/check_stratified_pr_network.py",
    ]
    report = {
        "scope": "Identical periodic single-phase PR caloric EOS, initial conserved stocks, first-order Rusanov spatial flux (500 m/s), zero friction/heat; native BDF vs the selected Torch integrator. Not a multiphase or whole-facility product comparison.",
        "request": {
            "cells": args.cells,
            "duration_s": args.duration_s,
            "length_m": 10,
            "diameter_m": 0.1,
            "torch_relative_tolerance": args.relative_tolerance,
            "torch_newton": args.newton,
            "torch_integrator": args.integrator,
        },
        "environment": {
            "openmodelica_version": subprocess.run(
                ["bash", str(args.omc_runtime.resolve() / "env.sh"), "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip(),
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "standard_library_sha256": hashlib.sha256(libraries[0].read_bytes()).hexdigest(),
        },
        "openmodelica_compile_wall_s": compile_s,
        "openmodelica": modelica_runs,
        "torch": {
            "cold": torch_runs[0],
            "warm": torch_runs[1:],
            "warm_median_wall_s": statistics.median(s["wall_s"] for s in torch_runs[1:]),
        },
        "checks": checks,
        "comparisons": comparisons,
        "source_sha256": {
            str(p.relative_to(root)) if p.is_relative_to(root) else str(p): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": checks, "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
