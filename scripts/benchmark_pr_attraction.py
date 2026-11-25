"""Paired end-to-end timings of exact PR quadratic mixing, with raw audits."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
import tarfile
import time
from pathlib import Path

from surface.equilibrium_properties import EquilibriumFluid


def benchmark(config, output, *, kind, repeats=3):
    if kind == "linepack":
        from check_pr_linepack import audit
        from run_pr_linepack import run

        from surface.pr_linepack import PRLinepackNetwork
        from surface.stratified_pr_network import StratifiedNetworkState

        model = PRLinepackNetwork.from_dict(config["model"])

        def difference(a, b):
            return max(
                model.network._error(
                    dict(StratifiedNetworkState.from_dict(i["state"]).vessels),
                    dict(StratifiedNetworkState.from_dict(j["state"]).vessels),
                )
                for i, j in zip(a["intervals"], b["intervals"], strict=True)
            )

    elif kind == "facility":
        from audit_pr_surface_collocation import audit
        from run_pr_controlled_surface import run as run_controlled

        from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState

        model = PRControlledSurface.from_dict(config["model"])

        def run(request):
            return run_controlled(request, integrator="sdirk2-common")

        def difference(a, b):
            return max(
                model._error(
                    PRControlledSurfaceState.from_dict(i["state"]),
                    PRControlledSurfaceState.from_dict(j["state"]),
                )
                for i, j in zip(a["intervals"], b["intervals"], strict=True)
            )

    else:
        raise ValueError("Choose linepack or facility")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Positive integer repeats required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src/surface").glob("*.py")) + sorted((root / "scripts").glob("*.py"))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for p in paths:
            archive.add(p, arcname=str(p.relative_to(root)))
    (output / "request.json").write_text(json.dumps(config, indent=2) + "\n")
    samples, comparisons = [], []
    previous = EquilibriumFluid._factored_attraction

    def take(method, name):
        EquilibriumFluid._factored_attraction = method == "factored"
        (output / "status.json").write_text(json.dumps({"stage": name, "status": "running"}))
        start = time.perf_counter()
        result = run(config)
        elapsed = time.perf_counter() - start
        result = json.loads(json.dumps(result, allow_nan=False))
        with gzip.open(output / f"{name}.json.gz", "wt") as stream:
            json.dump(result, stream)
        certificate = audit(config, result)
        if not certificate["accepted"]:
            raise ValueError("Independent audit rejected trajectory")
        row = {
            "method": method,
            "wall_s": elapsed,
            "audit": certificate,
            "trajectory": f"{name}.json.gz",
        }
        print(json.dumps({"stage": name, "wall_s": elapsed, "audit": True}), flush=True)
        return result, row

    try:
        # Warm imports and runtime machinery, with complete model construction
        # and initial-state validation still included in every timed repetition.
        for method in ("reference", "factored"):
            take(method, f"warm-{method}")
        for repeat in range(repeats):
            trajectories = {}
            order = ("reference", "factored") if repeat % 2 == 0 else ("factored", "reference")
            for method in order:
                result, row = take(method, f"repeat-{repeat}-{method}")
                row["repeat"] = repeat
                samples.append(row)
                (output / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
                trajectories[method] = result
            error = difference(trajectories["reference"], trajectories["factored"])
            comparisons.append({"repeat": repeat, "maximum_interval_error_in_original_norm": error})
            if not math.isfinite(error) or error > 1:
                raise ValueError("Original physical error budget exceeded")
    finally:
        EquilibriumFluid._factored_attraction = previous
    for name, digest in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during benchmark: {name}")
    medians = {
        m: statistics.median(s["wall_s"] for s in samples if s["method"] == m)
        for m in ("reference", "factored")
    }
    summary = {
        "accepted": True,
        "kind": kind,
        "median_wall_s": medians,
        "speedup": medians["reference"] / medians["factored"],
        "samples": samples,
        "comparisons": comparisons,
        "source_sha256": hashes,
        "scope": "CPU eager, one process; same equations/tolerances; construction and solve included; audits/serialization excluded",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps({"status": "complete", "accepted": True}))
    print(
        json.dumps({k: summary[k] for k in ("median_wall_s", "speedup", "comparisons")}, indent=2)
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kind", choices=("linepack", "facility"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    benchmark(
        json.loads(args.config.read_text()), args.output, kind=args.kind, repeats=args.repeats
    )
