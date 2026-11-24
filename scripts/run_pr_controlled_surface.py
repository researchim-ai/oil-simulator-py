"""Source-bound low-drop valves / actuators / PI and gravitational transport."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(
    config,
    *,
    device="cpu",
    progress=None,
    cuda_graphs=False,
    native_kernels=False,
    maximum_native_kernels=16,
):
    if native_kernels:
        if device != "cpu" or cuda_graphs:
            raise ValueError("Native CPU kernels cannot be combined with CUDA execution")
        from surface.native_execution import native_kernel_execution

        with native_kernel_execution(
            maximum_kernels=maximum_native_kernels, progress=progress
        ) as execution:
            result = _run(config, device=device, progress=progress)
            result["numerical_policy"]["native_execution"] = execution.statistics()
            return result
    if not cuda_graphs:
        return _run(config, device=device, progress=progress)
    if device != "cuda":
        raise ValueError("CUDA graphs require the CUDA device")
    from surface.cuda_execution import cuda_graph_execution

    with cuda_graph_execution(progress=progress) as execution:
        result = _run(config, device=device, progress=progress)
        result["numerical_policy"]["cuda_execution"] = execution.statistics()
        return result


def _run(config, *, device="cpu", progress=None):
    from surface.equilibrium_properties import EquilibriumFluid
    from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
    from surface.pr_pump_connection import PumpConnectionBundle
    from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState
    from surface.stratified_pr import StratifiedPRState
    from surface.torch_backend import runtime_fingerprint
    from surface.torch_nonlinear import TorchNewton

    if (
        set(config) != {"model", "initial", "schedule"}
        or not isinstance(config["schedule"], list)
        or not config["schedule"]
    ):
        raise ValueError("Expected model, initial and nonempty schedule")
    model = PRControlledSurface.from_dict(config["model"], device=device)
    initial = config["initial"]
    if set(initial) == {"state"}:
        state = PRControlledSurfaceState.from_dict(initial["state"])
        model.inspect(state)
    else:
        if set(initial) != {"plant", "positions", "integrals"}:
            raise ValueError("Expected plant, positions and integrals")
        spec = initial["plant"]
        if hasattr(model.plant, "bridges"):
            if set(spec) != {"transport", "profiles"} or set(spec["profiles"]) != set(
                model.plant.names
            ):
                raise ValueError("Expected transport and exact profile registry")
            physical = model.plant.initialize(
                model.plant.transport.initialize(spec["transport"]),
                {
                    n: b.layered.initialize_equilibrium(**spec["profiles"][n])
                    for n, b in model.plant.bridges.items()
                },
            )
        else:
            if set(spec) != {"transport", "columns", "wall_temperatures"}:
                raise ValueError("Expected transport, columns and wall_temperatures")
            ts = model.plant.transport.initialize(spec["transport"])
            columns = {
                n: StratifiedPRState.from_dict(model.plant.columns[n].initialize(**v)["state"])
                for n, v in spec["columns"].items()
            }
            physical = model.plant.initialize(ts, columns, spec["wall_temperatures"])
        state = model.initialize(physical, initial["positions"], initial["integrals"])
    first = state
    if progress is not None:
        progress({"event": "initialized", "time_s": state.time_s})
    intervals = []
    for command in config["schedule"]:
        if (
            set(command)
            - {
                "duration_s",
                "openings",
                "heat_w",
                "utility_temperature_k",
                "loss_coefficients",
                "drive_torque_nm",
                "restriction_openings",
                "targets",
                "actuator_modes",
                "reset_trips",
                "profile_heat_w",
                "profile_utility_temperature_k",
                "profile_openings",
            }
            or "duration_s" not in command
        ):
            raise ValueError("Unknown/missing equilibrium transport command")
        report = model.advance(state, progress=progress, **command)
        state = PRControlledSurfaceState.from_dict(report["state"])
        intervals.append(report)
    heat = sum(r["total_heat_j"] for r in intervals)
    work = sum(r["total_work_j"] for r in intervals)
    return {
        "config": config,
        "model": model.to_dict(),
        "initial": first.to_dict(),
        "intervals": intervals,
        "conservation_residuals": model.plant._ledger(first.plant, state.plant, heat + work),
        "total_heat_j": heat,
        "total_work_j": work,
        "runtime": runtime_fingerprint(),
        "numerical_policy": {
            "shared_probe_quadrature": model._share_probe_quadrature,
            "temporal_probe_predictor": model._reuse_temporal_predictor,
            "spatial_probe_predictor": getattr(model.plant, "_reuse_spatial_predictor", False),
            "transport_nonlinear_work_including_probes": [
                {"nodes": len(t.nodes), "model_id": t.model_id, **t.nonlinear_work}
                for t in dict.fromkeys(
                    [model.plant.transport]
                    + ([model.fast, model.slow.plant.transport] if hasattr(model, "fast") else [])
                )
            ],
            "temporal_probe_predictor_work": [
                {
                    "attempts": p.temporal_predictor_attempts,
                    "uses": p.temporal_predictor_uses,
                    "vector_uses": p.temporal_vector_uses,
                    "spatial_vector_uses": p.spatial_vector_uses,
                }
                for p in dict.fromkeys(
                    [model.plant] + ([model.slow.plant] if hasattr(model, "slow") else [])
                )
                if hasattr(p, "temporal_predictor_attempts")
            ],
            "fast_prefix_reuse": getattr(model, "_reuse_fast_prefixes", False),
            "identical_trial_reuse": TorchNewton._reuse_identical_trials,
            "stagnation_patience": TorchNewton._stagnation_patience,
            "jacobian_reuse": TorchNewton._reuse_jacobian,
            "accepted_residual_reuse": TorchNewton._reuse_accepted_residual,
            "fugacity_ssi_predictor": EquilibriumFluid._fugacity_ssi_predictor,
            "node_validation_reuse": PREquilibriumTransportNetwork._reuse_node_validation,
            "node_validation": [
                {
                    "nodes": len(transport.nodes),
                    "hits": transport._verified_node_hits,
                    "misses": transport._verified_node_misses,
                    "entries": len(transport._verified_nodes),
                    "maximum_entries": transport._node_validation_entries,
                }
                for transport in dict.fromkeys(
                    [model.plant.transport]
                    + ([model.fast, model.slow.plant.transport] if hasattr(model, "fast") else [])
                )
            ],
            "dense_jacobian_mode": TorchNewton._dense_jacobian_mode,
            "fast_time_scheme": getattr(model, "fast_time_scheme", "backward_euler"),
            "fast_jacobian_history": getattr(
                getattr(model, "fast_integrator", None), "_reuse_step_jacobian", False
            ),
            "early_certified_roundoff": TorchNewton._early_certified_roundoff,
            "preserve_transport_endpoint_step": PREquilibriumTransportNetwork._preserve_endpoint_step,
            "analytic_single_phase_characteristics": PREquilibriumTransportNetwork._analytic_single_phase_characteristics,
            "pump_check_valve_active_set": PumpConnectionBundle._fixed_check_valves,
            "pump_check_valve_branch_reuse": PumpConnectionBundle._reuse_valve_branch,
            "pump_shutoff_valve_predictor": PumpConnectionBundle._shutoff_valve_predictor,
            "compiled_chemical": EquilibriumFluid._compiled_chemical,
            "compiled_cpu_chemical": EquilibriumFluid._compiled_chemical_cpu,
            "compiled_cpu_full_eos": EquilibriumFluid._compiled_full_eos_cpu,
            "compiled_cuda_full_eos": EquilibriumFluid._compiled_full_eos_cuda,
            "analytic_chemical": EquilibriumFluid._analytic_chemical,
            "minimization_line_width": EquilibriumFluid._minimization_line_width,
            "compact_chemical_cpu": EquilibriumFluid._compact_chemical_cpu,
            "warm_equilibrium": EquilibriumFluid._warm_equilibrium,
            "grouped_column_eos": {
                name: column._batch_column_eos
                for name, column in getattr(model.plant, "columns", {}).items()
            },
            "chemical_kernels": [
                fluid.chemical_kernel_statistics()
                for fluid in dict.fromkeys(
                    [model.plant.transport.fluid]
                    + [column.fluid for column in getattr(model.plant, "columns", {}).values()]
                )
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Reuse bounded CUDA graphs for dense Newton kernels; original physics and tolerances",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native-kernels",
        action="store_true",
        help="Compile exact CPU kernels; final Newton acceptance remains eager",
    )
    parser.add_argument("--progress", action="store_true", help="JSON progress on stderr")
    parser.add_argument(
        "--maximum-native-kernels",
        type=int,
        default=16,
        help="Bound native CPU kernel admission; an existing caller-owned context takes precedence",
    )
    parser.add_argument(
        "--no-fugacity-ssi",
        action="store_true",
        help="Reference: disable fugacity successive-substitution predictors",
    )
    parser.add_argument(
        "--reevaluate-accepted-residual",
        action="store_true",
        help="Reference: recompute the accepted scalar residual at every Newton iteration",
    )
    parser.add_argument(
        "--compiled-full-eos",
        action="store_true",
        help="Fuse exact full caloric EOS on the selected device; AD retains the eager kernel",
    )
    parser.add_argument(
        "--staged-stability-lines",
        action="store_true",
        help="Evaluate TPD line searches in kernel-sized groups, preserving first admissible scales",
    )
    parser.add_argument(
        "--full-cpu-chemical-tiles",
        action="store_true",
        help="Reference policy: pad small CPU chemical EOS batches to 64 rows",
    )
    parser.add_argument(
        "--shrink-at-interval-end",
        action="store_true",
        help="Use endpoint-clipped transport steps as next proposals (reference policy)",
    )
    parser.add_argument(
        "--refine-discarded-probe",
        action="store_true",
        help="Independently refine the discarded temporal probe for reference comparisons",
    )
    parser.add_argument(
        "--evaluate-identical-trials",
        action="store_true",
        help="Evaluate duplicate CPU trial vectors for reference/performance comparisons",
    )
    parser.add_argument(
        "--stagnation-patience",
        type=int,
        choices=(2, 4),
        default=2,
        help="Consecutive sub-1%% improvements inside the explicit roundoff budget",
    )
    parser.add_argument(
        "--confirm-roundoff-plateau",
        action="store_true",
        help="Reference Newton policy: confirm stagnation instead of stopping at certified arithmetic floor",
    )
    parser.add_argument(
        "--fresh-jacobians",
        action="store_true",
        help="Rebuild every Newton tangent for reference/performance comparisons",
    )
    parser.add_argument(
        "--reverse-dense-jacobian",
        action="store_true",
        help="Use reverse AD for small dense Newton blocks; sparse graph assembly is unchanged",
    )
    chemical_policy = parser.add_mutually_exclusive_group()
    chemical_policy.add_argument(
        "--compiled-cpu-chemical",
        action="store_true",
        help="Opt in to fused float64 CPU chemical EOS and local derivatives",
    )
    chemical_policy.add_argument(
        "--eager-chemical",
        action="store_true",
        help="Disable fused CUDA chemical EOS for reference/performance comparisons",
    )
    column_eos = parser.add_mutually_exclusive_group()
    column_eos.add_argument(
        "--scalar-column-eos",
        action="store_true",
        help="Evaluate each column/port EOS segment separately for reference comparisons",
    )
    column_eos.add_argument(
        "--group-column-eos",
        action="store_true",
        help="Force grouped column/port EOS (CPU default; CUDA requires separate timing)",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((root / "src/surface").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    from surface.equilibrium_properties import EquilibriumFluid
    from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
    from surface.torch_nonlinear import TorchNewton

    EquilibriumFluid._compiled_full_eos_cpu = args.compiled_full_eos and args.device == "cpu"
    EquilibriumFluid._fugacity_ssi_predictor = not args.no_fugacity_ssi
    TorchNewton._reuse_accepted_residual = not args.reevaluate_accepted_residual
    EquilibriumFluid._compiled_full_eos_cuda = args.compiled_full_eos and args.device == "cuda"
    EquilibriumFluid._compact_chemical_cpu = not args.full_cpu_chemical_tiles
    EquilibriumFluid._minimization_line_width = 4 if args.staged_stability_lines else 12

    PREquilibriumTransportNetwork._preserve_endpoint_step = not args.shrink_at_interval_end
    TorchNewton._reuse_jacobian = not args.fresh_jacobians
    TorchNewton._dense_jacobian_mode = "reverse" if args.reverse_dense_jacobian else "forward"
    TorchNewton._early_certified_roundoff = not args.confirm_roundoff_plateau
    from surface.pr_surface_control import PRControlledSurface

    PRControlledSurface._share_probe_quadrature = not args.refine_discarded_probe
    from surface.pr_subcycled_surface import PRSubcycledSurface

    PRSubcycledSurface._share_probe_quadrature = not args.refine_discarded_probe
    TorchNewton._reuse_identical_trials = not args.evaluate_identical_trials
    TorchNewton._stagnation_patience = args.stagnation_patience

    EquilibriumFluid._compiled_chemical_cpu = args.compiled_cpu_chemical
    EquilibriumFluid._compiled_chemical = not args.eager_chemical
    from surface.stratified_pr import StratifiedPRColumn

    StratifiedPRColumn._batch_column_eos = (
        False if args.scalar_column_eos else True if args.group_column_eos else None
    )
    result = run(
        json.loads(args.config.read_text()),
        device=args.device,
        cuda_graphs=args.cuda_graphs,
        native_kernels=args.native_kernels,
        maximum_native_kernels=args.maximum_native_kernels,
        progress=(lambda event: print(json.dumps(event), file=sys.stderr, flush=True))
        if args.progress
        else None,
    )
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]
        for p in paths
    ):
        raise RuntimeError("Transport sources changed during run")
    result["source_sha256"] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as handle:
            temp = handle.name
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp, args.output)
        temp = None
    finally:
        if temp:
            os.unlink(temp)
    print(args.output)


if __name__ == "__main__":
    main()
