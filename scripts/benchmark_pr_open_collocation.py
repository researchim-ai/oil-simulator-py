"""Paired work/precision of a real open six-node two-phase PR graph.

All methods use identical input settings. A twice-refined Radau trajectory
is an internal temporal reference, not an independent exact flow solution.
Every accepted step additionally passes the scalar EOS/HLLC/boundary audit.
"""

import argparse
import copy
import gc
import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from check_pr_collocation import ROOT
from check_pr_open_transport import audit

from surface.pr_equilibrium_transport import PREquilibriumTransportState
from surface.pr_open_transport import PROpenTransportNetwork
from surface.torch_backend import runtime_fingerprint

BUDGETS = {
    "pressure_pa": 1.0,
    "temperature_k": 1e-4,
    "component_moles": 1e-8,
    "momentum_kg_m_s": 1e-8,
    "total_energy_j": 1e-5,
}


def prepare(config):
    model = PROpenTransportNetwork.from_dict(config["model"])
    if set(config["initial"]) != {"nodes"}:
        raise ValueError("Benchmark requires explicit node initialization")
    return model, model.transport.initialize(config["initial"]["nodes"])


def integrate(model, state, schedule, method, *, timings=None):
    results = []
    for command in schedule:
        before = state
        start = time.perf_counter() if timings is not None else None
        result = model.advance(state, **command, integrator=method)
        if timings is not None:
            timings.append(time.perf_counter() - start)
        state = PREquilibriumTransportState.from_dict(result["state"])
        results.append((before, result))
    return results


def differences(result, reference):
    values = {k: 0.0 for k in BUDGETS}
    for (_, a), (_, b) in zip(result, reference, strict=True):
        for name, x in a["state"]["nodes"].items():
            y = b["state"]["nodes"][name]
            for key, previous in values.items():
                delta = (
                    max(abs(u - v) for u, v in zip(x[key], y[key]))
                    if key == "component_moles"
                    else abs(x[key] - y[key])
                )
                values[key] = max(previous, delta)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if torch.get_num_threads() != 1 or not 1 <= args.repeats <= 5:
        parser.error("Use one CPU thread and 1..5 pairs")
    config = json.loads(args.config.read_text())
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        args.config.resolve(),
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=True)
    references = []
    for dt in (2e-6, 1e-6):
        fine = copy.deepcopy(config)
        opts = fine["model"]["transport"]
        opts.update(
            initial_step_s=dt,
            maximum_step_s=dt,
            relative_tolerance=opts["relative_tolerance"] * 0.1,
            temperature_tolerance_k=opts["temperature_tolerance_k"] * 0.1,
            velocity_tolerance_m_s=opts["velocity_tolerance_m_s"] * 0.1,
        )
        model, state = prepare(fine)
        reference = integrate(model, state, fine["schedule"], "radau3")
        for command, (before, report) in zip(fine["schedule"], reference):
            audit(
                model.to_dict(),
                before.to_dict(),
                report,
                **{k: v for k, v in command.items() if k != "duration_s"},
            )
        references.append(reference)
        with gzip.open(args.output / f"reference_{dt}.json.gz", "wt") as stream:
            json.dump(
                {"config": fine, "intervals": [r for _, r in reference]}, stream, allow_nan=False
            )
    refinement = differences(*references)
    assert all(refinement[k] <= 0.1 * BUDGETS[k] for k in BUDGETS), refinement
    methods = ["backward_euler", "gauss4", "radau3"]
    rows = []
    for method in methods:
        m, s = prepare(config)
        integrate(m, s, config["schedule"][:1], method)
    for pair in range(args.repeats):
        order = methods[pair:] + methods[:pair]
        for method in order:
            model, state = prepare(config)
            gc.collect()
            start = time.perf_counter()
            result = integrate(model, state, config["schedule"], method)
            elapsed = time.perf_counter() - start
            for command, (before, report) in zip(config["schedule"], result):
                audit(
                    model.to_dict(),
                    before.to_dict(),
                    report,
                    **{k: v for k, v in command.items() if k != "duration_s"},
                )
            errors = differences(result, references[-1])
            passed = all(errors[k] <= BUDGETS[k] for k in BUDGETS)
            row = {
                "pair": pair,
                "method": method,
                "elapsed_s": elapsed,
                "errors": errors,
                "within_budgets": passed,
                "accepted_substeps": sum(len(r["steps"]) for _, r in result),
                "attempts": sum(r["attempts"] for _, r in result),
                "collocation_nonlinear_solves": sum(
                    r.get("collocation_nonlinear_solves", 0) for _, r in result
                )
                if method != "backward_euler"
                else None,
                "fallback_intervals": sum(r.get("fallback_intervals", 0) for _, r in result),
            }
            rows.append(row)
            with gzip.open(args.output / f"pair{pair}_{method}.json.gz", "wt") as stream:
                json.dump(
                    {
                        "config": config,
                        "initial": state.to_dict(),
                        "intervals": [r for _, r in result],
                        "measurement": row,
                    },
                    stream,
                    allow_nan=False,
                )
            print(json.dumps(row), flush=True)
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]
        for p in paths
    ):
        raise RuntimeError("Sources changed during open benchmark")
    medians = {
        method: statistics.median(row["elapsed_s"] for row in rows if row["method"] == method)
        for method in methods
    }
    summary = {
        "scope": "CPU six-node open two-phase PR network, reverse reservoir flow, branch closing and reopening. Short numerical benchmark; not full facility/GPU/commercial qualification",
        "duration_s": sum(c["duration_s"] for c in config["schedule"]),
        "budgets": BUDGETS,
        "reference_refinement_errors": refinement,
        "reference": "Radau3 dt=2us and 1us with 10x tighter temporal tolerances; not an independent exact trajectory",
        "all_accepted_steps_independently_audited": True,
        "median_elapsed_s": medians,
        "all_methods_within_budgets": all(r["within_budgets"] for r in rows),
        "gauss_to_radau_speedup": medians["gauss4"] / medians["radau3"]
        if all(r["within_budgets"] for r in rows if r["method"] in ("gauss4", "radau3"))
        else None,
        "repeats": args.repeats,
        "runtime": runtime_fingerprint(),
        "cpu_threads": torch.get_num_threads(),
        "runs": rows,
        "source_sha256": hashes,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("runs", "source_sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
