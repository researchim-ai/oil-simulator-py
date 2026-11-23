"""Qualify changed adaptive paths against stricter local time-error budgets.

The original paired benchmark is retained verbatim, including a failed 0.01
same-path comparison. Changing adaptive paths requires temporal refinement,
not simply relabeling that comparison as passing. All norms below use the
original requested tolerances; physical stocks/geometry/commands are unchanged.
"""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

from check_pr_controlled_surface import audit
from run_pr_controlled_surface import run

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiled-cpu-chemical", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    EquilibriumFluid._compiled_chemical_cpu = args.compiled_cpu_chemical
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__),
        ROOT / "scripts/run_pr_controlled_surface.py",
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    results = {}
    for name in ("reference", "accelerated"):
        with gzip.open(args.pair / (name + ".json.gz"), "rt") as stream:
            results[name] = json.load(stream)
    original = results["reference"]["report"]["config"]
    model = PRControlledSurface.from_dict(original["model"])
    if original["model"].get("model") != "pr_subcycled_surface_v1":
        raise ValueError("Expected explicit local-subcycling model")
    local_budget = original["model"]["local_error_fraction"]
    for factor in (4, 16):
        config = json.loads(json.dumps(original))
        config["model"]["local_error_fraction"] = local_budget / factor
        refined_model = PRControlledSurface.from_dict(config["model"])
        # Only the local numerical error budget/identity changes. The original
        # transport/column model and every physical checkpoint field stay exact.
        if refined_model.plant.to_dict() != model.plant.to_dict():
            raise ValueError("Physical plant changed during numerical refinement")
        config["initial"]["state"]["model_id"] = refined_model.model_id
        refined_model.inspect(PRControlledSurfaceState.from_dict(config["initial"]["state"]))
        started = time.perf_counter()
        report = json.loads(json.dumps(run(config)))
        elapsed = time.perf_counter() - started
        checked = audit(config, report)
        if checked.get("accepted") is not True:
            raise ValueError("Refined trajectory failed independent audit")
        key = f"refined_{factor}"
        results[key] = {"wall_s": elapsed, "report": report, "audit": checked}
        with gzip.open(args.output_dir / (key + ".json.gz"), "wt") as stream:
            json.dump(results[key], stream, allow_nan=False)
        print(json.dumps({"completed": key, "wall_s": elapsed}), flush=True)
    states = {
        k: PRControlledSurfaceState.from_dict(v["report"]["intervals"][-1]["state"])
        for k, v in results.items()
    }
    if len({s.time_s for s in states.values()}) != 1:
        raise ValueError("End clocks differ")
    differences = {
        k: model._error(v, states["refined_16"]) for k, v in states.items() if k != "refined_16"
    }
    passed = (
        differences["refined_4"] <= local_budget / 4
        and differences["reference"] <= local_budget
        and differences["accelerated"] <= local_budget
    )
    unchanged = all(
        hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))]
        for p in paths
    )
    summary = {
        "scope": "Accepted interval; local time refinement only, not full-cycle/global validation",
        "difference_from_refined_16_in_original_norm": differences,
        "reference_convergence_limit": local_budget / 4,
        "original_local_error_budget": local_budget,
        "qualified": passed and unchanged,
        "sources_unchanged": unchanged,
        "source_sha256": hashes,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2))
    return 0 if summary["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
