"""Paired original/common controlled-plant timings with independent audits.

Both methods run the same complete request. This benchmark uses eager CPU
operators; do not compare its wall times with native/CUDA runs as a speedup.
"""

import argparse
import gzip
import hashlib
import json
import math
import statistics
import tarfile
import time
from pathlib import Path

from audit_pr_surface_collocation import audit as audit_common
from check_pr_controlled_surface import audit as audit_sampled
from prepared_surface_operators import operator_preparation
from run_pr_controlled_surface import run


def benchmark(request, output, repeats=3, *, experiment="integrators"):
    from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState

    if type(repeats) is not int or repeats < 1:
        raise ValueError("Positive repeat count required")
    if experiment not in ("integrators", "preparation"):
        raise ValueError("Unknown controlled benchmark experiment")
    request = json.loads(json.dumps(request))
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src/surface").glob("*.py")) + sorted((root / "scripts").glob("*.py"))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for name in hashes:
            archive.add(root / name, arcname=name)
    journal = []
    methods = (
        ("sampled", "sdirk2-common") if experiment == "integrators" else ("reference", "prepared")
    )
    auditors = {
        "sampled": audit_sampled,
        "sdirk2-common": audit_common,
        "reference": audit_common,
        "prepared": audit_common,
    }
    model = PRControlledSurface.from_dict(request["model"])

    def save(name, value):
        with gzip.open(output / f"{name}.json.gz", "wt") as stream:
            json.dump(value, stream)

    def take(method, name):
        (output / "status.json").write_text(json.dumps({"status": "running", "stage": name}) + "\n")
        start = time.perf_counter()
        if experiment == "preparation":
            with operator_preparation(method == "prepared"):
                result = run(request, integrator="sdirk2-common")
        else:
            result = run(request, integrator=method)
        elapsed = time.perf_counter() - start
        result = json.loads(json.dumps(result))
        checked = auditors[method](request, result)
        if not checked["accepted"]:
            raise ValueError("Independent common-plant audit rejected")
        save(name, result)
        row = {
            "method": method,
            "wall_s": elapsed,
            "audit": checked,
            "fallback_intervals": sum(i.get("fallback_intervals", 0) for i in result["intervals"]),
            "attempts": sum(i["attempts"] for i in result["intervals"]),
            "rejected_attempts": sum(
                i.get("rejected_attempts", i["attempts"] - len(i["steps"]) // 2)
                for i in result["intervals"]
            ),
            "collocation_nonlinear_solves": sum(
                i.get("collocation_nonlinear_solves", 0) for i in result["intervals"]
            ),
            "conservation_residuals": result["conservation_residuals"],
            "trajectory": f"{name}.json.gz",
        }
        journal.append(row)
        (output / "journal.json").write_text(json.dumps(journal, indent=2) + "\n")
        print(json.dumps({"stage": name, "wall_s": elapsed, "audit_accepted": True}), flush=True)
        return result, row

    for method in methods:
        take(method, f"prepare-{method}")
    samples, comparisons = [], []
    for repeat in range(repeats):
        states = {}
        for method in methods if repeat % 2 == 0 else methods[::-1]:
            result, row = take(method, f"repeat-{repeat}-{method}")
            row["repeat"] = repeat
            samples.append(row)
            states[method] = PRControlledSurfaceState.from_dict(result["intervals"][-1]["state"])
            print(json.dumps(row), flush=True)
        comparisons.append(
            {
                "repeat": repeat,
                "endpoint_difference_in_original_norm": model._error(
                    states[methods[0]], states[methods[1]]
                ),
            }
        )
        difference = comparisons[-1]["endpoint_difference_in_original_norm"]
        if not math.isfinite(difference) or difference > 1.0:
            raise ValueError("Controlled benchmark changed trajectory beyond original error norm")
    for name, digest in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during common-plant benchmark: {name}")
    medians = {
        method: statistics.median(s["wall_s"] for s in samples if s["method"] == method)
        for method in methods
    }
    summary = {
        "accepted": True,
        "scope": "same complete controlled PR request, eager CPU, warm paired runs; audits and serialization outside wall time",
        "experiment": experiment,
        "source_sha256": hashes,
        "samples": samples,
        "comparisons": comparisons,
        "median_wall_s": medians,
        "speedup": medians[methods[0]] / medians[methods[1]],
        "runtime": result["runtime"],
        "simulated_duration_s": sum(c["duration_s"] for c in request["schedule"]),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps({"status": "complete", "accepted": True}) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--experiment", choices=("integrators", "preparation"), default="integrators"
    )
    args = parser.parse_args()
    result = benchmark(
        json.loads(args.request.read_text()), args.output, args.repeats, experiment=args.experiment
    )
    print(json.dumps({k: result[k] for k in ("median_wall_s", "speedup", "comparisons")}, indent=2))
