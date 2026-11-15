"""Same-source scalar/grouped PR audit comparison, with unchanged physics.

Model setup is outside timing. Every accepted step is independently audited;
physical states and the complete accepted time mesh must match exactly.
No experimental/GPU/full-field speed qualification is implied.
"""

import argparse
import gc
import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from benchmark_pr_open_collocation import BUDGETS, differences, integrate
from check_pr_collocation import ROOT
from check_pr_open_transport import audit

from surface.pr_open_transport import PROpenTransportNetwork
from surface.torch_backend import runtime_fingerprint


def prepare(config, enabled, optimization="characteristics"):
    model = PROpenTransportNetwork.from_dict(config["model"])
    if optimization == "characteristics":
        model.transport._batch_characteristics = enabled
    elif optimization == "boundary":
        model._reuse_boundary_properties = enabled
    elif optimization == "stages":
        model.transport._batch_collocation_stages = enabled
    else:
        raise ValueError("Unknown execution optimization")
    if set(config["initial"]) != {"nodes"}:
        raise ValueError("Explicit initial nodes required")
    return model, model.transport.initialize(config["initial"]["nodes"])


def physical_signature(results):
    return [
        {
            "state": r["state"],
            "steps": [(s["dt_s"], s["state"], s["entropy_production_j_k"]) for s in r["steps"]],
        }
        for _, r in results
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=["backward_euler", "gauss4", "radau3"], default=["radau3"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--optimization",
        choices=["characteristics", "boundary", "stages"],
        default="characteristics",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Previously archived refined trajectory, explicitly hash-bound",
    )
    args = parser.parse_args()
    if (
        torch.get_num_threads() != 1
        or not 1 <= args.repeats <= 5
        or len(set(args.methods)) != len(args.methods)
    ):
        parser.error("One CPU thread, 1..5 repeats, unique methods required")
    if (args.output / "summary.json").exists():
        parser.error("Choose a fresh output directory; completed measurements are immutable")
    config_payload = args.config.read_bytes()
    config = json.loads(config_payload)
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        args.config.resolve(),
    ]

    def key(p):
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

    hashes = {key(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes[key(args.config.resolve())] = hashlib.sha256(config_payload).hexdigest()
    reference = None
    refhash = None
    if args.reference:
        reference_payload = args.reference.read_bytes()
        refhash = hashlib.sha256(reference_payload).hexdigest()
        archived = json.loads(gzip.decompress(reference_payload))
        reference = [(None, r) for r in archived["intervals"]]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in args.methods:
        for enabled in (False, True):
            m, s = prepare(config, enabled, args.optimization)
            integrate(m, s, config["schedule"][:1], method)
        for pair in range(args.repeats):
            paired = {}
            for enabled in (False, True) if pair % 2 == 0 else (True, False):
                model, state = prepare(config, enabled, args.optimization)
                gc.collect()
                intervals = []
                start = time.perf_counter()
                result = integrate(model, state, config["schedule"], method, timings=intervals)
                elapsed = time.perf_counter() - start
                for command, (before, report) in zip(config["schedule"], result):
                    audit(
                        model.to_dict(),
                        before.to_dict(),
                        report,
                        **{k: v for k, v in command.items() if k != "duration_s"},
                    )
                mode = (
                    ("batched" if enabled else "scalar")
                    if args.optimization != "boundary"
                    else ("reused" if enabled else "uncached")
                )
                errors = differences(result, reference) if reference is not None else None
                within = (
                    all(errors[k] <= BUDGETS[k] for k in errors) if errors is not None else None
                )
                row = {
                    "method": method,
                    "mode": mode,
                    "pair": pair,
                    "elapsed_s": elapsed,
                    "interval_elapsed_s": intervals,
                    "boundary_preparation": model._prepared_boundaries.statistics(),
                    "accepted_substeps": sum(len(r["steps"]) for _, r in result),
                    "attempts": sum(r["attempts"] for _, r in result),
                    "errors_vs_archived_refinement": errors,
                    "within_reference_budgets": within,
                }
                paired[mode] = result
                rows.append(row)
                with gzip.open(args.output / f"{method}_pair{pair}_{mode}.json.gz", "wt") as stream:
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
            names = (
                ("scalar", "batched") if args.optimization != "boundary" else ("uncached", "reused")
            )
            assert json.dumps(
                physical_signature(paired[names[0]]), sort_keys=True, allow_nan=False
            ) == json.dumps(
                physical_signature(paired[names[1]]), sort_keys=True, allow_nan=False
            ), f"Physical trajectory/time mesh changed: {method} pair {pair}"
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[key(p)] for p in paths):
        raise RuntimeError("Sources changed during measurement")
    mode_names = (
        ("scalar", "batched") if args.optimization != "boundary" else ("uncached", "reused")
    )
    medians = {
        method: {
            mode: statistics.median(
                r["elapsed_s"] for r in rows if r["method"] == method and r["mode"] == mode
            )
            for mode in (
                ("scalar", "batched") if args.optimization != "boundary" else ("uncached", "reused")
            )
        }
        for method in args.methods
    }
    summary = {
        "scope": "Same-source exact PR execution optimization A/B; CPU six-node open binary PR graph. Not a full facility, GPU or commercial comparison",
        "model_time_s": sum(c["duration_s"] for c in config["schedule"]),
        "repeats": args.repeats,
        "median_elapsed_s": medians,
        "optimization": args.optimization,
        "speedup": {m: v[mode_names[0]] / v[mode_names[1]] for m, v in medians.items()},
        "all_physical_states_time_mesh_and_entropy_bitwise_equal": True,
        "all_accepted_steps_independently_audited": True,
        "reference_artifact": str(args.reference) if args.reference else None,
        "reference_artifact_sha256": refhash,
        "all_within_reference_budgets": all(r["within_reference_budgets"] for r in rows)
        if reference is not None
        else None,
        "budgets": BUDGETS if reference is not None else None,
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
