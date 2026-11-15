"""Same-equation diagonal/node-two-stage BlockJacobi work comparison.

Fixed-step stiff pure-methane PR thermal rings isolate algebraic conditioning.
They do not qualify time-discretization accuracy, a full facility or GPU speed.
"""

import argparse
import gc
import gzip
import hashlib
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from audit_pr_collocation import audit_gauss_step
from check_pr_collocation import ROOT, thermal_fixture

from surface import PRCollocation, PREquilibriumNode, PREquilibriumTransportNetwork
from surface.liquid import HydraulicConvergenceError
from surface.pr_collocation import TorchNewton
from surface.pr_heat_link import PRHeatLink
from surface.torch_backend import runtime_fingerprint
from surface.vessel_wall import VesselWall

BUDGETS = {
    "pressure_pa": 1e-5,
    "temperature_k": 1e-9,
    "wall_temperature_k": 1e-9,
    "component_moles": 1e-11,
    "total_energy_j": 1e-7,
    "momentum_kg_m_s": 1e-11,
}


def fixture(count, *, homogeneous=False):
    base, _ = thermal_fixture()
    nodes = [
        PREquilibriumNode(
            f"n{i:03}",
            0.001,
            wall=VesselWall(
                40.0 if homogeneous else 20.0 + i,
                1e5 if homogeneous else 1e3 * 10 ** (i % 4),
                max_temperature_change_k=20.0,
            ),
        )
        for i in range(count)
    ]
    links = [
        PRHeatLink(f"h{i:03}", f"n{i:03}", f"n{(i + 1) % count:03}", 5.0) for i in range(count)
    ]
    model = PREquilibriumTransportNetwork(
        base.fluid,
        nodes,
        heat_links=links,
        dense_limit=1,
        initial_step_s=0.02,
        maximum_step_s=0.25,
        minimum_step_s=1e-8,
    )
    state = model.initialize(
        {
            name: {
                "pressure_pa": 1e6,
                "temperature_k": 320.0 + (10 * (i % 2) if homogeneous else (7 * i) % 13),
                "wall_temperature_k": 325.0 + (10 * (i % 2) if homogeneous else (7 * i) % 13),
                "mole_fractions": [1.0, 0.0],
            }
            for i, name in enumerate(model.names)
        }
    )
    return model, state


@contextmanager
def trace_linear_solves():
    rows = []
    original = TorchNewton.solve

    def observed(self, fn, initial, *args, **kwargs):
        result = original(self, fn, initial, *args, **kwargs)
        blocks = kwargs.get("blocks") or []
        rows.append(
            {
                "unknowns": initial.numel(),
                "stored_entries": self.matrix.values.numel() if self.matrix is not None else None,
                "block_count": len(blocks),
                "largest_block": max(map(len, blocks), default=0),
                "factor_block_sizes": []
                if self.block_preconditioner is None
                else [list(f.shape) for f in self.block_preconditioner.factors],
                "jacobian_builds": self.jacobian_builds,
                "jacobian_reuses": self.jacobian_reuses,
                "nonlinear_evaluations": result[3],
                "stop_reason": self.stop_reason,
                "success": bool(result[2]),
            }
        )
        return result

    TorchNewton.solve = observed
    try:
        yield rows
    finally:
        TorchNewton.solve = original


def step(model, state, scheme, enabled):
    solver = PRCollocation(model, scheme=scheme, phase_fallback=False)
    solver._block_preconditioning = enabled
    return solver._step(
        state,
        0.02,
        model.default_openings,
        [0.0] * len(model.nodes),
        [s.temperature_k for _, s in state.nodes],
        {},
    )


def difference(a, b):
    errors = {k: 0.0 for k in BUDGETS}
    for (na, x), (nb, y) in zip(a.nodes, b.nodes, strict=True):
        assert na == nb
        for key, previous in errors.items():
            xx, yy = getattr(x, key), getattr(y, key)
            delta = (
                max(abs(u - v) for u, v in zip(xx, yy, strict=True))
                if key == "component_moles"
                else abs(xx - yy)
            )
            errors[key] = max(previous, delta)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if torch.get_num_threads() != 1 or not 1 <= args.repeats <= 5:
        parser.error("One CPU thread and 1..5 repeats required")
    if (args.output / "summary.json").exists():
        parser.error("Choose a fresh output directory")
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    pairs = []
    cases = [
        ("homogeneous16", 16, True, "radau3"),
        ("heterogeneous32", 32, False, "radau3"),
        ("heterogeneous32", 32, False, "gauss4"),
        ("heterogeneous64", 64, False, "radau3"),
    ]
    # Warm separate models; no measured model/factor can inherit this solve.
    for enabled in (False, True):
        m, s = fixture(4)
        step(m, s, "radau3", enabled)
    for label, count, homogeneous, scheme in cases:
        for pair in range(args.repeats):
            results = {}
            for enabled in (False, True) if pair % 2 == 0 else (True, False):
                mode = "block" if enabled else "diagonal"
                model, state = fixture(count, homogeneous=homogeneous)
                gc.collect()
                record = None
                error = None
                with trace_linear_solves() as trace:
                    start = time.perf_counter()
                    try:
                        final, record = step(model, state, scheme, enabled)
                    except HydraulicConvergenceError as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    elapsed = time.perf_counter() - start
                if record is not None:
                    assert audit_gauss_step(
                        model.to_dict(),
                        state.to_dict(),
                        record,
                        model.fluid,
                        {"openings": model.default_openings},
                    )["accepted"]
                    results[mode] = final
                row = {
                    "case": label,
                    "nodes": count,
                    "method": scheme,
                    "mode": mode,
                    "pair": pair,
                    "success": record is not None,
                    "elapsed_to_success_s": elapsed if record is not None else None,
                    "elapsed_to_failure_s": elapsed if record is None else None,
                    "error": error,
                    "linear_solves": trace,
                    "jacobian_builds": sum(x["jacobian_builds"] for x in trace),
                    "nonlinear_evaluations": sum(x["nonlinear_evaluations"] for x in trace),
                    "independent_step_audit_passed": record is not None,
                }
                rows.append(row)
                with gzip.open(args.output / f"{label}_{scheme}_{pair}_{mode}.json.gz", "wt") as f:
                    json.dump(
                        {
                            "model": model.to_dict(),
                            "initial": state.to_dict(),
                            "record": record,
                            "measurement": row,
                        },
                        f,
                        allow_nan=False,
                    )
                print(json.dumps(row), flush=True)
            assert "block" in results, "Block solve must complete the qualified case"
            if "diagonal" in results:
                errors = difference(results["diagonal"], results["block"])
                assert all(errors[k] <= BUDGETS[k] for k in errors), errors
                pairs.append({"case": label, "method": scheme, "pair": pair, "errors": errors})
    medians = {}
    for label, _count, _homogeneous, method in cases:
        key = f"{label}_{method}"
        medians[key] = {}
        for mode in ("diagonal", "block"):
            data = [
                r
                for r in rows
                if r["case"] == label and r["method"] == method and r["mode"] == mode
            ]
            ok = all(r["success"] for r in data)
            medians[key][mode] = {
                "successful_runs": sum(r["success"] for r in data),
                "median_elapsed_s": statistics.median(r["elapsed_to_success_s"] for r in data)
                if ok
                else None,
                "jacobian_builds": [r["jacobian_builds"] for r in data],
            }
        a, b = (
            medians[key]["diagonal"]["median_elapsed_s"],
            medians[key]["block"]["median_elapsed_s"],
        )
        medians[key]["speedup"] = a / b if a is not None and b is not None else None
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    summary = {
        "scope": __doc__,
        "fixed_step_s": 0.02,
        "repeats": args.repeats,
        "budgets_between_solved_discrete_states": BUDGETS,
        "comparisons": pairs,
        "medians": medians,
        "runs": rows,
        "runtime": runtime_fingerprint(),
        "cpu_threads": 1,
        "source_sha256": hashes,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
