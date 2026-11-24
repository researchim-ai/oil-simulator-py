"""Compare accepted connected intervals with BE versus Radau fast dynamics.

Physics, initial inventories, controls and error tolerances are identical.
Each complete trajectory is independently audited, including Radau stages.
This is a shared-host observation, not a real-time performance guarantee.
"""

import argparse
import copy
import gc
import gzip
import hashlib
import json
import math
import statistics
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from check_pr_controlled_surface import audit
from run_pr_controlled_surface import run

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState
from surface.torch_nonlinear import TorchNewton


def prepare(config, scheme):
    """Change only the integrator selector and its enclosing checkpoint ID."""
    request = copy.deepcopy(config)
    request["model"].pop("fast_time_scheme", None)
    if scheme != "backward_euler":
        request["model"]["fast_time_scheme"] = scheme
    model = PRControlledSurface.from_dict(request["model"])
    if set(request["initial"]) != {"state"}:
        raise ValueError("An explicit physical checkpoint is required for a paired run")
    original = PRControlledSurfaceState.from_dict(request["initial"]["state"])
    request["initial"] = {"state": replace(original, model_id=model.model_id).to_dict()}
    return json.loads(json.dumps(request))


def measure(request, *, native, progress, profiler=None):
    started = time.perf_counter()
    gc_starts, gc_times, gc_counts = {}, [0.0] * 3, [0] * 3

    def observe_gc(phase, info):
        generation = info["generation"]
        if phase == "start":
            gc_starts[generation] = time.perf_counter()
        elif phase == "stop":
            begin = gc_starts.pop(generation, None)
            if begin is not None:
                gc_times[generation] += time.perf_counter() - begin
                gc_counts[generation] += 1

    gc.callbacks.append(observe_gc)
    if profiler is not None:
        profiler.enable()
    try:
        report = run(request, native_kernels=native, progress=progress)
    finally:
        if profiler is not None:
            profiler.disable()
        gc.callbacks.remove(observe_gc)
    elapsed = time.perf_counter() - started
    report = json.loads(json.dumps(report))
    if len(report["intervals"]) != len(request["schedule"]):
        raise ValueError("Incomplete requested schedule")
    first = PRControlledSurfaceState.from_dict(report["initial"])
    final = PRControlledSurfaceState.from_dict(report["intervals"][-1]["state"])
    requested = math.fsum(row["duration_s"] for row in request["schedule"])
    if not math.isclose(final.time_s - first.time_s, requested, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Incomplete physical interval")
    fast = [
        stage["report"]
        for interval in report["intervals"]
        for step in interval["steps"]
        for stage in step["local_subcycling"]["stages"]
        if stage["operator"] == "fast"
    ]
    return report, {
        "wall_s": elapsed,
        "gc_wall_s_by_generation": gc_times,
        "gc_collections_by_generation": gc_counts,
        "simulated_duration_s": final.time_s - first.time_s,
        "attempts": sum(v["attempts"] for v in report["intervals"]),
        "accepted_macro_steps": sum(len(v["steps"]) for v in report["intervals"]),
        "accepted_fast_steps": sum(len(r["steps"]) for r in fast),
        "fast_nonlinear_solves": sum(r.get("collocation_nonlinear_solves", 0) for r in fast),
        "fast_fallback_intervals": sum(r.get("fallback_intervals", 0) for r in fast),
        "execution": report["numerical_policy"].get("native_execution"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiled-cpu-chemical", action="store_true")
    parser.add_argument("--compiled-full-eos", action="store_true")
    parser.add_argument("--reverse-dense-jacobian", action="store_true")
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["backward_euler", "radau3"],
        choices=["backward_euler", "radau3", "radau3-embedded"],
    )
    parser.add_argument(
        "--native-kernels",
        action="store_true",
        help="Native CPU execution for non-BE candidates only",
    )
    parser.add_argument("--maximum-native-kernels", type=int, default=16)
    parser.add_argument(
        "--repeats", type=int, default=3, help="Warm fresh-start repeats, plus one first run"
    )
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.maximum_native_kernels < 1
        or len(set(args.schemes)) != len(args.schemes)
    ):
        parser.error("Positive budgets/repeats and unique schemes required")
    if "backward_euler" not in args.schemes:
        parser.error("Include backward_euler as the common reference")
    config = json.loads(args.config.read_text())
    if config["model"]["model"] != "pr_subcycled_surface_v1":
        raise ValueError("A subcycled controlled surface is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "src/surface").glob("*.py")),
        Path(__file__),
        root / "scripts/run_pr_controlled_surface.py",
        root / "scripts/check_pr_controlled_surface.py",
        root / "scripts/check_pr_subcycled_surface.py",
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": hashes,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    EquilibriumFluid._compiled_chemical_cpu = args.compiled_cpu_chemical
    EquilibriumFluid._compiled_full_eos_cpu = args.compiled_full_eos
    TorchNewton._dense_jacobian_mode = "reverse" if args.reverse_dense_jacobian else "forward"
    from surface.native_execution import native_kernel_execution
    from surface.torch_backend import runtime_fingerprint

    results, endpoints = {}, {}
    for scheme in args.schemes:
        native = args.native_kernels and scheme != "backward_euler"
        request = prepare(config, scheme)
        (args.output_dir / (scheme + ".request.json")).write_text(
            json.dumps(request, indent=2) + "\n"
        )
        runs = []
        last_progress = [0.0]

        def progress(event, scheme=scheme, last_progress=last_progress):
            now = time.perf_counter()
            if event.get("event") == "native_kernel_compile" or now - last_progress[0] >= 15:
                print(json.dumps({"scheme": scheme, **event}), flush=True)
                last_progress[0] = now

        with ExitStack() as scope:
            if native:
                scope.enter_context(
                    native_kernel_execution(
                        maximum_kernels=args.maximum_native_kernels, progress=progress
                    )
                )
            for repeat in range(args.repeats + 1):
                print(
                    json.dumps({"event": "begin", "scheme": scheme, "repeat": repeat}), flush=True
                )
                report, record = measure(request, native=native, progress=progress)
                # Retain failed audit evidence too; only the summary qualifies a run.
                with gzip.open(args.output_dir / f"{scheme}_{repeat}.json.gz", "wt") as handle:
                    json.dump(
                        {"request": request, "report": report, **record}, handle, allow_nan=False
                    )
                checked = audit(request, report)
                if checked.get("accepted") is not True:
                    raise ValueError("Independent physical audit rejected the trajectory")
                record.update(repeat=repeat, audit=checked)
                runs.append(record)
                print(json.dumps({"event": "completed", "scheme": scheme, **record}), flush=True)
                endpoints[scheme] = [
                    PRControlledSurfaceState.from_dict(v["state"]) for v in report["intervals"]
                ]
        median = statistics.median(r["wall_s"] for r in runs[1:])
        results[scheme] = {
            "wall_s": median,
            "cold": runs[0],
            "warm": runs[1:],
            "warm_median_wall_s": median,
        }
    model = PRControlledSurface.from_dict(config["model"])
    differences = {
        scheme: [
            model._error(a, b) for a, b in zip(endpoints["backward_euler"], states, strict=True)
        ]
        for scheme, states in endpoints.items()
    }
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]
        for p in paths
    ):
        raise RuntimeError("Sources changed during paired measurement")
    summary = {
        "scope": "complete supplied schedule including model initialization; fresh physical states; audit outside timing",
        "runtime": runtime_fingerprint(),
        "results": results,
        "difference_at_each_report_in_original_error_norm": differences,
        "accepted": all(max(v) <= 1 for v in differences.values()),
        "independent_audits_passed": True,
        "observed_speedups": {
            k: results["backward_euler"]["wall_s"] / v["wall_s"] for k, v in results.items()
        },
        "source_sha256": hashes,
        "complete_operating_cycle_qualified": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("source_sha256", "results")}, indent=2
        )
    )
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
