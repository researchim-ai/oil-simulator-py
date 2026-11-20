"""Paired exact PR phase-flash sequences, including cold startup and all audits.

These are scalar thermodynamic calls, not physical seconds or a whole network.
Unique P/T points prevent exact-result cache hits from masquerading as speedup.
"""

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import torch

from surface.equilibrium_cache import cached_flashes
from surface.equilibrium_properties import EquilibriumFluid
from surface.peng_robinson import PengRobinsonFluid

ROOT = Path(__file__).resolve().parents[1]


def run_sequence(f, states, analytic, warm):
    f._lean_allocation = analytic
    f._analytic_chemical = analytic
    f._warm_equilibrium = warm
    if f.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with cached_flashes(f, maximum_entries=2) as cache:
        results = [f.flash_pt(p, t, z) for p, t, z in states]
        stats = cache.statistics()
    if f.device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start, results, stats


def compare(a, b):
    if a["phase_count"] != b["phase_count"]:
        raise ValueError("Phase counts differ")
    pairs = min(
        itertools.permutations(range(len(b["phases"]))),
        key=lambda order: sum(
            float((x["mole_fractions"] - b["phases"][j]["mole_fractions"]).abs().max().cpu())
            for x, j in zip(a["phases"], order)
        ),
    )
    composition = max(
        float((x["mole_fractions"] - b["phases"][j]["mole_fractions"]).abs().max().cpu())
        for x, j in zip(a["phases"], pairs)
    )
    fraction = max(
        float((x["molar_fraction"] - b["phases"][j]["molar_fraction"]).abs().cpu())
        for x, j in zip(a["phases"], pairs)
    )
    relative = max(
        float(((a[k] - b[k]).abs() / a[k].abs().clamp_min(1.0)).cpu())
        for k in (
            "molar_enthalpy_j_mol",
            "molar_internal_energy_j_mol",
            "molar_entropy_j_mol_k",
            "density_kg_m3",
        )
    )
    return {
        "max_composition_absolute_difference": composition,
        "max_phase_fraction_absolute_difference": fraction,
        "max_bulk_scaled_difference": relative,
        "agree": max(composition, fraction, relative) < 1e-7,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--compiled-cpu-chemical", action="store_true")
    p.add_argument("--points", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 2 <= args.points <= 100:
        p.error("2..100 unique points required")
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        p.error("Explicit GPU required")
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__)]
    hashes = {str(q.relative_to(ROOT)): hashlib.sha256(q.read_bytes()).hexdigest() for q in paths}
    data = json.loads((ROOT / "tests/data/surface/pr_mixture_coolprop.json").read_text())
    factories = [
        (
            "hydrocarbon_vle",
            lambda: EquilibriumFluid(
                [PengRobinsonFluid(**c) for c in data["components"]],
                [[0, 0], [0, 0]],
                device=args.device,
            ),
            196.2,
            [0.5, 0.5],
        ),
        (
            "synthetic_three_liquids",
            lambda: EquilibriumFluid(
                [PengRobinsonFluid(n, 500, 5e6, 0.2, 0.1, 100, 250, 700) for n in ("a", "b", "c")],
                [[0, 0.6, 0.6], [0.6, 0, 0.6], [0.6, 0.6, 0]],
                device=args.device,
            ),
            300.0,
            [1 / 3] * 3,
        ),
    ]
    result = {
        "device": args.device,
        "points_per_sequence": args.points,
        "cases": {},
        "source_sha256": hashes,
        "scope": "Paired scalar flash sequences, includes cold first call; not a network or experimental qualification",
    }
    for name, factory, t, z in factories:
        f = factory()
        f._compiled_chemical_cpu = args.compiled_cpu_chemical
        states = [(1e6 * (1 + i * 1e-5), t + i * 0.002, z) for i in range(args.points)]
        # Trigger code generation in both paths outside measured sequences.
        run_sequence(f, states[:2], False, False)
        run_sequence(f, states[:2], True, True)
        old, reference, oldstats = run_sequence(f, states, False, False)
        new, actual, newstats = run_sequence(f, states, True, True)
        checks = [compare(a, b) for a, b in zip(reference, actual)]
        result["cases"][name] = {
            "reference_s": old,
            "accelerated_s": new,
            "observed_speedup": old / new,
            "reference_ms_per_flash": old / len(states) * 1000,
            "accelerated_ms_per_flash": new / len(states) * 1000,
            "comparisons": checks,
            "reference_statistics": oldstats,
            "accelerated_statistics": newstats,
            "phase_counts": [r["phase_count"] for r in actual],
        }
    if any(
        hashlib.sha256(q.read_bytes()).hexdigest() != hashes[str(q.relative_to(ROOT))]
        for q in paths
    ):
        raise RuntimeError("Source changed during paired measurement")
    result["algorithms_agree"] = all(
        row["agree"] for case in result["cases"].values() for row in case["comparisons"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                name: {
                    k: v
                    for k, v in case.items()
                    if k not in ("comparisons", "reference_statistics", "accelerated_statistics")
                }
                for name, case in result["cases"].items()
            },
            indent=2,
        )
    )
    return 0 if result["algorithms_agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
