"""Paired whole-schedule qualification of guarded Radau/probe optimizations.

Fresh physical initial conditions, one shared native context, alternating
policies. Compilation and independent audits are outside warm measurements.
No different time tolerances, geometry or thermodynamic shortcuts.
"""

import argparse
import cProfile
import gzip
import hashlib
import json
import pstats
import statistics
import tarfile
from pathlib import Path

from benchmark_pr_fast_integrator import measure
from check_pr_controlled_surface import audit

from surface import pr_collocation as collocation_module
from surface.equilibrium_properties import EquilibriumFluid
from surface.native_execution import native_kernel_execution
from surface.pr_collocation import PRCollocation
from surface.pr_gravity_transport import PRGravityTransportNetwork, _ColumnCoupling
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState
from surface.torch_backend import runtime_fingerprint
from surface.torch_nonlinear import TorchNewton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maximum-native-kernels", type=int, default=16)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile an additional warmed baseline outside timing pairs",
    )
    parser.add_argument(
        "--optimization",
        choices=(
            "chord-history",
            "temporal-predictor",
            "spatial-predictor",
            "stability-objective",
            "stability-carry",
            "closure-lifetime",
            "release-runtime",
        ),
        default="chord-history",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.maximum_native_kernels < 1:
        parser.error("Positive repeat count required")
    request = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "src/surface").glob("*.py")),
        Path(__file__),
        root / "scripts/benchmark_pr_fast_integrator.py",
        root / "scripts/run_pr_controlled_surface.py",
        root / "scripts/check_pr_controlled_surface.py",
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (args.output_dir / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    with tarfile.open(args.output_dir / "sources.tar.gz", "w:gz") as archive:
        for name in hashes:
            archive.add(root / name, arcname=name, recursive=False)
    EquilibriumFluid._compiled_chemical_cpu = True
    EquilibriumFluid._compiled_full_eos_cpu = True
    TorchNewton._dense_jacobian_mode = "reverse"
    model = PRControlledSurface.from_dict(request["model"])
    baseline, candidate = (
        ("fresh_solve", "guarded_history")
        if args.optimization == "chord-history"
        else ("fresh_probe", "warm_probe")
        if args.optimization == "temporal-predictor"
        else ("fresh_spatial", "warm_spatial")
        if args.optimization == "spatial-predictor"
        else ("original_objective", "fused_objective")
        if args.optimization == "stability-objective"
        else ("recomputed_values", "carried_values")
        if args.optimization == "stability-carry"
        else ("cyclic_closures", "acyclic_closures")
        if args.optimization == "closure-lifetime"
        else ("original_runtime", "updated_runtime")
    )
    results = {baseline: [], candidate: []}
    endpoints = {}
    original_marker = collocation_module.declare_runtime_residual
    original_prepare = _ColumnCoupling.prepare

    def cyclic_marker(function):
        # Reproduce the former protocol for a causal lifetime A/B only.
        function.runtime_residual = function
        return function

    def cyclic_prepare(self, *args, **kwargs):
        result = original_prepare(self, *args, **kwargs)
        self.evaluate = result[3]
        return result

    def execute(label, repeat, profiler=None):
        if args.optimization == "chord-history":
            PRCollocation._reuse_step_jacobian = label == candidate
        elif args.optimization == "temporal-predictor":
            PRControlledSurface._reuse_temporal_predictor = label == candidate
            PRGravityTransportNetwork._reuse_spatial_predictor = False
        elif args.optimization == "spatial-predictor":
            PRControlledSurface._reuse_temporal_predictor = False
            PRGravityTransportNetwork._reuse_spatial_predictor = label == candidate
        elif args.optimization == "stability-objective":
            EquilibriumFluid._fused_stability_objective_cpu = label == candidate
            EquilibriumFluid._reuse_stability_values = False
            PRControlledSurface._reuse_temporal_predictor = False
            PRGravityTransportNetwork._reuse_spatial_predictor = False
        elif args.optimization == "stability-carry":
            EquilibriumFluid._reuse_stability_values = label == candidate
            EquilibriumFluid._fused_stability_objective_cpu = False
            PRControlledSurface._reuse_temporal_predictor = False
            PRGravityTransportNetwork._reuse_spatial_predictor = False
        else:
            collocation_module.declare_runtime_residual = (
                original_marker if label == candidate else cyclic_marker
            )
            _ColumnCoupling.prepare = original_prepare if label == candidate else cyclic_prepare
            EquilibriumFluid._reuse_stability_values = (
                True if args.optimization == "closure-lifetime" else label == candidate
            )
            EquilibriumFluid._fused_stability_objective_cpu = False
            PRControlledSurface._reuse_temporal_predictor = False
            PRGravityTransportNetwork._reuse_spatial_predictor = False
        report, record = measure(
            request,
            native=True,
            progress=lambda e: print(json.dumps(e), flush=True),
            profiler=profiler,
        )
        with gzip.open(args.output_dir / f"{label}_{repeat}.json.gz", "wt") as handle:
            json.dump({"request": request, "report": report, **record}, handle, allow_nan=False)
        checked = audit(request, report)
        if checked.get("accepted") is not True:
            raise ValueError("Independent physical audit rejected trajectory")
        endpoints[label] = [
            PRControlledSurfaceState.from_dict(v["state"]) for v in report["intervals"]
        ]
        counters = {key: 0 for key in ("hits", "refreshes", "builds")}
        linear_work = {key: 0 for key in ("jacobian_builds", "jacobian_reuses")}
        for interval in report["intervals"]:
            for step in interval["steps"]:
                for stage in step["local_subcycling"]["stages"]:
                    if stage["operator"] == "fast":
                        for key, value in (
                            stage["report"].get("cross_step_linearization", {}).items()
                        ):
                            counters[key] += value
                        for key, value in stage["report"].get("linear_work", {}).items():
                            linear_work[key] += value
        record.update(
            audit=checked,
            policy=label,
            repeat=repeat,
            history=counters,
            linear_work=linear_work,
            transport_work=report["numerical_policy"].get(
                "transport_nonlinear_work_including_probes"
            ),
            temporal_predictor=report["numerical_policy"].get("temporal_probe_predictor_work"),
            chemical_work=report["numerical_policy"].get("chemical_kernels"),
        )
        print(json.dumps({"event": "completed", **record}), flush=True)
        return record

    old = PRCollocation._reuse_step_jacobian
    old_predictor = PRControlledSurface._reuse_temporal_predictor
    old_spatial = PRGravityTransportNetwork._reuse_spatial_predictor
    old_objective = EquilibriumFluid._fused_stability_objective_cpu
    old_carry = EquilibriumFluid._reuse_stability_values
    differences = []
    profile_record = None
    try:
        with native_kernel_execution(
            maximum_kernels=args.maximum_native_kernels,
            progress=lambda e: print(json.dumps(e), flush=True),
        ) as engine:
            preparation = execute(baseline, "preparation")
            settling = execute(candidate, "settling")
            for repeat in range(args.repeats):
                order = tuple(results) if repeat % 2 == 0 else tuple(reversed(results))
                for label in order:
                    results[label].append(execute(label, repeat))
                differences.append(
                    [
                        model._error(a, b)
                        for a, b in zip(endpoints[baseline], endpoints[candidate], strict=True)
                    ]
                )
            if args.profile:
                compilations_before = engine.statistics()["compilations"]
                previous_states = endpoints[baseline]
                profiler = cProfile.Profile()
                profile_record = execute(baseline, "profile", profiler=profiler)
                profiler.dump_stats(str(args.output_dir / "warm.prof"))
                with (args.output_dir / "warm_profile.txt").open("w") as stream:
                    stats = pstats.Stats(profiler, stream=stream).strip_dirs()
                    stats.sort_stats("cumulative").print_stats(65)
                    stats.sort_stats("tottime").print_stats(65)
                if endpoints[baseline] != previous_states:
                    raise ValueError("Profiling changed physical reporting states")
                profile_record["new_compilations"] = (
                    engine.statistics()["compilations"] - compilations_before
                )
            execution = engine.statistics()
    finally:
        PRCollocation._reuse_step_jacobian = old
        PRControlledSurface._reuse_temporal_predictor = old_predictor
        PRGravityTransportNetwork._reuse_spatial_predictor = old_spatial
        EquilibriumFluid._fused_stability_objective_cpu = old_objective
        EquilibriumFluid._reuse_stability_values = old_carry
        collocation_module.declare_runtime_residual = original_marker
        _ColumnCoupling.prepare = original_prepare
    unchanged = all(
        hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(root))]
        for p in paths
    )
    medians = {k: statistics.median(v["wall_s"] for v in rows) for k, rows in results.items()}
    summary = {
        "scope": __doc__,
        "optimization": args.optimization,
        "runtime": runtime_fingerprint(),
        "preparation": preparation,
        "settling": settling,
        "profile": profile_record,
        "results": results,
        "warm_median_wall_s": medians,
        "observed_speedup": medians[baseline] / medians[candidate],
        "difference_at_each_report_in_original_error_norm": differences,
        "independent_audits_passed": True,
        "sources_unchanged": unchanged,
        "native_execution": execution,
        "full_operating_cycle_qualified": False,
        "accepted": unchanged
        and max(map(max, differences)) <= 0.01
        and not execution["disabled"]
        and execution["fallback_calls"] == 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
