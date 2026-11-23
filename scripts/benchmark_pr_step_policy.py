"""Paired accepted interval: certified Newton floor and split probe reuse.

Both policies use identical physics/tolerances/initial stocks and current PR
kernels. Independent physical audit runs outside solver timing. The complete
requested interval must finish; rejected trials alone cannot pass this probe.
"""

import argparse
import gzip
import hashlib
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from check_pr_controlled_surface import audit
from run_pr_controlled_surface import run

from surface import equilibrium_minimization_batch
from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
from surface.pr_pump_connection import PumpConnectionBundle
from surface.pr_subcycled_surface import PRSubcycledSurface
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState
from surface.torch_nonlinear import TorchNewton

ROOT = Path(__file__).resolve().parents[1]


def measure(config, enabled, policy="certified-probe", *, device="cpu"):
    calls = []
    line_work = {"candidate_points": 0, "batches": 0}
    first_acceptable = equilibrium_minimization_batch._first_acceptable

    def counted_lines(evaluate, *args, **kwargs):
        def counted(trials, *a, **kw):
            line_work["candidate_points"] += math.prod(trials.shape[:-1])
            line_work["batches"] += 1
            return evaluate(trials, *a, **kw)

        return first_acceptable(counted, *args, **kwargs)

    original = TorchNewton.solve

    def counted(self, *args, **kwargs):
        started = time.perf_counter()
        answer = original(self, *args, **kwargs)
        calls.append(
            {
                "unknowns": args[1].numel(),
                "evaluations": answer[3],
                "jacobian_builds": self.jacobian_builds,
                "accepted_residual_reuses": self.accepted_residual_reuses,
                "stop": self.stop_reason,
                "wall_s": time.perf_counter() - started,
            }
        )
        return answer

    with ExitStack() as context:
        if policy in ("accepted-residual", "pipe-algorithms"):
            context.enter_context(patch.object(TorchNewton, "_reuse_accepted_residual", enabled))
            context.enter_context(patch.object(TorchNewton, "_dense_jacobian_mode", "reverse"))
            if policy == "pipe-algorithms":
                context.enter_context(
                    patch.object(EquilibriumFluid, "_fugacity_ssi_predictor", enabled)
                )
        elif policy == "fugacity-ssi":
            context.enter_context(
                patch.object(EquilibriumFluid, "_fugacity_ssi_predictor", enabled)
            )
            context.enter_context(patch.object(TorchNewton, "_dense_jacobian_mode", "reverse"))
        elif policy == "node-validation":
            context.enter_context(
                patch.object(PREquilibriumTransportNetwork, "_reuse_node_validation", enabled)
            )
            context.enter_context(patch.object(TorchNewton, "_dense_jacobian_mode", "reverse"))
        elif policy == "endpoint-carry":
            context.enter_context(
                patch.object(PREquilibriumTransportNetwork, "_preserve_endpoint_step", enabled)
            )
        elif policy == "single-phase-acoustics":
            context.enter_context(
                patch.object(
                    PREquilibriumTransportNetwork, "_analytic_single_phase_characteristics", enabled
                )
            )
        elif policy == "fast-prefix":
            context.enter_context(patch.object(PRSubcycledSurface, "_reuse_fast_prefixes", enabled))
        elif policy == "shutoff-valve-predictor":
            context.enter_context(
                patch.object(PumpConnectionBundle, "_shutoff_valve_predictor", enabled)
            )
        elif policy in ("pump-valve-seeds", "pump-seeds-reverse-ad"):
            context.enter_context(
                patch.object(PumpConnectionBundle, "_reuse_valve_branch", enabled)
            )
            if policy == "pump-seeds-reverse-ad":
                context.enter_context(
                    patch.object(
                        TorchNewton, "_dense_jacobian_mode", "reverse" if enabled else "forward"
                    )
                )
        elif policy == "full-eos":
            context.enter_context(patch.object(EquilibriumFluid, "_compiled_full_eos_cpu", enabled))
        elif policy == "staged-stability-search":
            context.enter_context(
                patch.object(EquilibriumFluid, "_minimization_line_width", 4 if enabled else 12)
            )
            context.enter_context(patch.object(TorchNewton, "_dense_jacobian_mode", "reverse"))
        elif policy in ("compact-chemical-tiles", "tiled-stability"):
            context.enter_context(patch.object(EquilibriumFluid, "_compact_chemical_cpu", enabled))
            context.enter_context(patch.object(TorchNewton, "_dense_jacobian_mode", "reverse"))
            context.enter_context(
                patch.object(
                    EquilibriumFluid,
                    "_minimization_line_width",
                    4 if enabled and policy == "tiled-stability" else 12,
                )
            )
        elif policy in ("reverse-dense-ad", "fused-properties-reverse-ad"):
            if policy == "fused-properties-reverse-ad":
                context.enter_context(
                    patch.object(EquilibriumFluid, "_compiled_full_eos_cpu", enabled)
                )
            context.enter_context(
                patch.object(
                    TorchNewton, "_dense_jacobian_mode", "reverse" if enabled else "forward"
                )
            )
        else:
            context.enter_context(patch.object(TorchNewton, "_early_certified_roundoff", enabled))
            context.enter_context(
                patch.object(PRControlledSurface, "_share_probe_quadrature", enabled)
            )
            context.enter_context(
                patch.object(PRSubcycledSurface, "_share_probe_quadrature", enabled)
            )
        context.enter_context(patch.object(TorchNewton, "solve", counted))
        context.enter_context(
            patch.object(equilibrium_minimization_batch, "_first_acceptable", counted_lines)
        )
        start = time.perf_counter()
        report = run(config, device=device)
        elapsed = time.perf_counter() - start
    report = json.loads(json.dumps(report))
    checked = audit(config, report)
    if checked.get("accepted") is not True:
        raise ValueError("Independent physical audit rejected")
    return {
        "wall_s": elapsed,
        "report": report,
        "audit": checked,
        "solves": calls,
        "stability_line_search": line_work,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--compiled-cpu-chemical", action="store_true")
    p.add_argument("--accelerated-first", action="store_true")
    p.add_argument("--warmup", action="store_true", help="Audit both paths once before timing")
    p.add_argument(
        "--policy",
        choices=(
            "certified-probe",
            "node-validation",
            "fugacity-ssi",
            "accepted-residual",
            "pipe-algorithms",
            "endpoint-carry",
            "single-phase-acoustics",
            "fast-prefix",
            "shutoff-valve-predictor",
            "pump-valve-seeds",
            "pump-seeds-reverse-ad",
            "full-eos",
            "staged-stability-search",
            "compact-chemical-tiles",
            "tiled-stability",
            "reverse-dense-ad",
            "fused-properties-reverse-ad",
        ),
        default="certified-probe",
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    EquilibriumFluid._compiled_chemical_cpu = args.compiled_cpu_chemical
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__),
        ROOT / "scripts/run_pr_controlled_surface.py",
    ]
    hashes = {str(q.relative_to(ROOT)): hashlib.sha256(q.read_bytes()).hexdigest() for q in paths}
    results = {}
    if args.warmup:
        for enabled in (False, True):
            measure(config, enabled, args.policy, device=args.device)
    sequence = [("reference", False), ("accelerated", True)]
    if args.accelerated_first:
        sequence.reverse()
    for name, enabled in sequence:
        r = measure(config, enabled, args.policy, device=args.device)
        with gzip.open(args.output_dir / (name + ".json.gz"), "wt") as f:
            json.dump(r, f, allow_nan=False)
        results[name] = r
    a, b = (results[k] for k in ("reference", "accelerated"))
    model = PRControlledSurface.from_dict(config["model"])
    old = PRControlledSurfaceState.from_dict(a["report"]["intervals"][-1]["state"])
    new = PRControlledSurfaceState.from_dict(b["report"]["intervals"][-1]["state"])
    difference = model._error(old, new)
    if old.time_s != new.time_s:
        raise ValueError("End clocks differ")

    def stats(r):
        sizes = sorted({s["unknowns"] for s in r["solves"]})
        return {
            "wall_s": r["wall_s"],
            "nonlinear_solves": len(r["solves"]),
            "residual_evaluations": sum(x["evaluations"] for x in r["solves"]),
            "jacobian_builds": sum(x["jacobian_builds"] for x in r["solves"]),
            "accepted_residual_reuses": sum(x["accepted_residual_reuses"] for x in r["solves"]),
            "attempts": sum(x["attempts"] for x in r["report"]["intervals"]),
            "stability_line_search": r["stability_line_search"],
            "chemical_padded_points": sum(
                k["padded_points"] for k in r["report"]["numerical_policy"]["chemical_kernels"]
            ),
            "nonlinear_work_by_size": {
                str(size): {
                    "solves": sum(s["unknowns"] == size for s in r["solves"]),
                    "wall_s": sum(s["wall_s"] for s in r["solves"] if s["unknowns"] == size),
                }
                for size in sizes
            },
        }

    summary = {
        "scope": f"Paired {args.device.upper()} accepted schedule including initialization; shared host, not a statistical speedup guarantee",
        "simulated_duration_s": sum(c["duration_s"] for c in config["schedule"]),
        "warmup_runs_excluded": 2 if args.warmup else 0,
        "config": str(args.config),
        "policy": args.policy,
        "reference": stats(a),
        "accelerated": stats(b),
        "observed_speedup": a["wall_s"] / b["wall_s"],
        "end_time_s": new.time_s,
        "difference_in_original_error_norm": difference,
        "agree": difference < 0.01,
        "independent_audits_passed": True,
        "source_sha256": hashes,
        "complete_operating_cycle_qualified": False,
    }
    if any(
        hashlib.sha256(q.read_bytes()).hexdigest() != hashes[str(q.relative_to(ROOT))]
        for q in paths
    ):
        raise RuntimeError("Source changed during paired measurement")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2))
    return 0 if summary["agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
