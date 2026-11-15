"""Paired CPU work/precision for a closed pure-PR isochoric heat contact.

Not a full-network, GPU, reservoir or commercial-simulator comparison.
Run without concurrent jobs for qualified timing. The same physical model
and numerical tolerances are used by both solvers; no coefficient fitting.
"""

import argparse
import gc
import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from check_pr_collocation import ROOT, IsochoricReference, thermal_fixture
from scipy.integrate import solve_ivp

from surface import PRCollocation, PREquilibriumTransportState
from surface.torch_backend import runtime_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--methods",
        nargs=2,
        choices=["backward_euler", "gauss4", "radau3"],
        default=["backward_euler", "gauss4"],
    )
    parser.add_argument("--fixture", choices=["contact", "walls"], default="contact")
    args = parser.parse_args()
    if not 0 < args.duration <= 0.5 or not 1 <= args.repeats <= 10:
        parser.error("Duration must be (0, .5], repeats 1..10")
    if len(set(args.methods)) != 2:
        parser.error("Choose two different methods")
    if torch.get_num_threads() != 1:
        parser.error("Set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1")
    sources = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        ROOT / "tests/data/surface/pr_mixture_coolprop.json",
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    fixture, reference_type = thermal_fixture, IsochoricReference
    if args.fixture == "walls":
        from check_pr_collocation_walls import WallReference, wall_fixture

        fixture, reference_type = wall_fixture, WallReference
    model, state = fixture()
    ref = reference_type(model, state)
    initial = [s.temperature_k for _, s in state.nodes]
    if args.fixture == "walls":
        initial += [s.wall_temperature_k for _, s in state.nodes] + [0.0]
    reference = solve_ivp(
        ref.rhs,
        (0.0, args.duration),
        initial,
        method="DOP853",
        rtol=1e-13,
        atol=1e-12,
        dense_output=True,
    )
    assert reference.success
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    # Warm both implementations outside the measured intervals.
    for method in args.methods:
        m, s = fixture()
        solver = (
            m
            if method == "backward_euler"
            else PRCollocation(m, scheme=method, phase_fallback=False)
        )
        solver.advance(s, min(0.001, args.duration))
    for pair in range(args.repeats):
        methods = args.methods if pair % 2 == 0 else list(reversed(args.methods))
        for method in methods:
            m, s = fixture()
            solver = (
                m
                if method == "backward_euler"
                else PRCollocation(m, scheme=method, phase_fallback=False)
            )
            gc.collect()
            start = time.perf_counter()
            report = solver.advance(s, args.duration)
            elapsed = time.perf_counter() - start
            error_t = error_p = 0.0
            for step in report["steps"]:
                endpoint = PREquilibriumTransportState.from_dict(step["state"])
                expected = reference.sol(endpoint.time_s)
                error_t = max(
                    error_t,
                    float(
                        abs(
                            np.array([cell.temperature_k for _, cell in endpoint.nodes])
                            - expected[:2]
                        ).max()
                    ),
                )
                error_p = max(
                    error_p,
                    float(
                        abs(
                            np.array([cell.pressure_pa for _, cell in endpoint.nodes])
                            - ref.pressure(expected[:2])
                        ).max()
                    ),
                )
                if args.fixture == "walls":
                    error_t = max(
                        error_t,
                        float(
                            abs(
                                np.array([cell.wall_temperature_k for _, cell in endpoint.nodes])
                                - expected[2:4]
                            ).max()
                        ),
                    )
            row = {
                "pair": pair,
                "method": method,
                "elapsed_s": elapsed,
                "accepted_substeps": len(report["steps"]),
                "attempts": report["attempts"],
                "temperature_error_k": error_t,
                "pressure_error_pa": error_p,
                "fallback_intervals": report.get("fallback_intervals", 0),
                "within_requested_budgets": error_t <= 1e-3 and error_p <= 5.0,
            }
            rows.append(row)
            with gzip.open(args.output / f"pair{pair}_{method}.json.gz", "wt") as stream:
                json.dump(
                    {
                        "model": m.to_dict(),
                        "initial": s.to_dict(),
                        "result": report,
                        "measurement": row,
                    },
                    stream,
                    allow_nan=False,
                )
            print(json.dumps(row), flush=True)
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]
        for p in sources
    ):
        raise RuntimeError("Sources changed during paired timing")
    medians = {
        name: statistics.median(row["elapsed_s"] for row in rows if row["method"] == name)
        for name in args.methods
    }
    passed = all(row["within_requested_budgets"] for row in rows)
    summary = {
        "scope": "Same-input paired CPU synthetic pure-PR heat-contact benchmark; not full facility/GPU/OPM/Eclipse performance",
        "methods": args.methods,
        "fixture": args.fixture,
        "simulated_duration_s": args.duration,
        "pairs": args.repeats,
        "runtime": runtime_fingerprint(),
        "cpu_threads": torch.get_num_threads(),
        "budgets": {"temperature_k": 1e-3, "pressure_pa": 5.0},
        "reference": "Independent scalar isochoric PR Cv/pressure, DOP853 rtol=1e-13 atol=1e-12, all accepted endpoints checked",
        "runs": rows,
        "median_elapsed_s": medians,
        "both_within_requested_budgets": passed,
        "qualified_speedup": medians[args.methods[0]] / medians[args.methods[1]]
        if passed
        else None,
        "source_sha256": hashes,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("source_sha256", "runs", "runtime")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
