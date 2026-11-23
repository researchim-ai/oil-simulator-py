"""Compare accepted connected intervals with BE versus Radau fast dynamics.

Physics, initial inventories, controls and error tolerances are identical.
Each complete trajectory is independently audited, including Radau stages.
This is a shared-host observation, not a real-time performance guarantee.
"""

import argparse
import copy
import gzip
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from check_pr_controlled_surface import audit
from run_pr_controlled_surface import run

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState
from surface.torch_nonlinear import TorchNewton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiled-cpu-chemical", action="store_true")
    parser.add_argument("--reverse-dense-jacobian", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["model"]["model"] != "pr_subcycled_surface_v1":
        raise ValueError("A subcycled controlled surface is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "src/surface").glob("*.py")), Path(__file__)]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    EquilibriumFluid._compiled_chemical_cpu = args.compiled_cpu_chemical
    TorchNewton._dense_jacobian_mode = "reverse" if args.reverse_dense_jacobian else "forward"
    results = {}
    for scheme in ("backward_euler", "radau3"):
        request = copy.deepcopy(config)
        request["model"].pop("fast_time_scheme", None)
        if scheme == "radau3":
            request["model"]["fast_time_scheme"] = scheme
        model = PRControlledSurface.from_dict(request["model"])
        if "state" in request["initial"]:
            original = PRControlledSurfaceState.from_dict(request["initial"]["state"])
            state = replace(original, model_id=model.model_id)
            model.inspect(state)
            request["initial"] = {"state": state.to_dict()}
        request = json.loads(json.dumps(request))
        start = time.perf_counter()
        report = run(request)
        elapsed = time.perf_counter() - start
        report = json.loads(json.dumps(report))
        checked = audit(request, report)
        if checked.get("accepted") is not True:
            raise ValueError("Independent audit rejected the trajectory")
        fast = [
            stage["report"]
            for interval in report["intervals"]
            for step in interval["steps"]
            for stage in step["local_subcycling"]["stages"]
            if stage["operator"] == "fast"
        ]
        results[scheme] = {
            "wall_s": elapsed,
            "accepted_fast_steps": sum(len(r["steps"]) for r in fast),
            "fast_fallback_intervals": sum(r.get("fallback_intervals", 0) for r in fast),
            "state": report["intervals"][-1]["state"],
            "audit": checked,
        }
        with gzip.open(args.output_dir / f"{scheme}.json.gz", "wt") as handle:
            json.dump(
                {"request": request, "report": report, "audit": checked}, handle, allow_nan=False
            )
        print(scheme, elapsed, flush=True)
    old, new = [PRControlledSurfaceState.from_dict(r["state"]) for r in results.values()]
    if old.time_s != new.time_s:
        raise ValueError("Endpoints differ")
    difference = model._error(old, new)
    summary = {
        "scope": __doc__,
        "results": {
            k: {n: v for n, v in r.items() if n not in ("state", "audit")}
            for k, r in results.items()
        },
        "end_time_s": new.time_s,
        "difference_in_original_error_norm": difference,
        "agree_within_one_percent_of_budget": difference < 0.01,
        "independent_audits_passed": True,
        "observed_speedup": results["backward_euler"]["wall_s"] / results["radau3"]["wall_s"],
        "source_sha256": hashes,
        "complete_operating_cycle_qualified": False,
    }
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]
        for p in paths
    ):
        raise RuntimeError("Sources changed during paired measurement")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2))
    return 0 if difference < 0.01 else 1


if __name__ == "__main__":
    raise SystemExit(main())
