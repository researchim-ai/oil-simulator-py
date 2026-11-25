"""Paired whole-graph Radau step-doubling versus embedded control.

Accepts a normal run_pr_equilibrium_transport request. This measures the
supplied transport/machine graph, not the gravitational controlled facility.
Preparation and the capped-step numerical reference are outside timed runs.
"""

import argparse
import copy
import gzip
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from check_pr_equilibrium_transport import audit
from run_pr_equilibrium_transport import run


def benchmark(request, output, *, repeats=3, reference_step_s=1e-5):
    from surface.pr_equilibrium_transport import (
        PREquilibriumTransportNetwork,
        PREquilibriumTransportState,
    )

    request = json.loads(json.dumps(request))
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Positive repeat count required")
    if not math.isfinite(reference_step_s) or reference_step_s <= 0:
        raise ValueError("Positive finite reference cap required")
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "scripts/run_pr_equilibrium_transport.py"]
    paths += sorted((root / "scripts").glob("reference_*.py"))
    paths += sorted((root / "scripts").glob("audit_pr_collocation*.py"))
    paths += sorted((root / "scripts").glob("check_pr_*.py"))
    paths += sorted((root / "scripts").glob("check_stratified_pr*.py"))
    paths += sorted((root / "src/surface").glob("*.py"))
    fingerprint = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")

    def save(name, value):
        with gzip.open(output / f"{name}.json.gz", "wt") as stream:
            json.dump(value, stream)

    reference_request = copy.deepcopy(request)
    settings = reference_request["model"]
    cap = min(reference_step_s, settings["maximum_step_s"])
    if cap < settings["minimum_step_s"]:
        raise ValueError("Reference cap below model's minimum time step")
    settings["maximum_step_s"] = cap
    settings["initial_step_s"] = min(cap, settings["initial_step_s"])
    if "state" in reference_request["initial"]:
        # The model ID includes numerical policy. Use a model with the new
        # policy to validate the unchanged physical checkpoint explicitly.
        model = PREquilibriumTransportNetwork.from_dict(settings)
        checkpoint = reference_request["initial"]["state"]
        checkpoint["model_id"] = model.model_id
        checkpoint["next_step_s"] = cap
    reference = run(reference_request, integrator="radau3")
    reference_audit = audit(reference_request, json.loads(json.dumps(reference)))
    save("reference", reference)
    expected = PREquilibriumTransportState.from_dict(reference["intervals"][-1]["state"])
    model = PREquilibriumTransportNetwork.from_dict(request["model"])
    methods = ("radau3", "radau3-embedded")
    for method in methods:
        save(f"prepare-{method}", run(request, integrator=method))
    rows = []
    for repeat in range(repeats):
        for method in methods if repeat % 2 == 0 else methods[::-1]:
            start = time.perf_counter()
            result = run(request, integrator=method)
            elapsed = time.perf_counter() - start
            checked = audit(request, json.loads(json.dumps(result)))
            final = PREquilibriumTransportState.from_dict(result["intervals"][-1]["state"])
            error = model._error(final, expected)
            if not math.isfinite(error) or error > 1:
                raise ValueError(
                    f"{method}: reference error ratio {error} exceeds original tolerance"
                )
            name = f"repeat-{repeat}-{method}"
            save(name, result)
            rows.append(
                {
                    "method": method,
                    "repeat": repeat,
                    "wall_s": elapsed,
                    "reference_error_ratio": error,
                    "nonlinear_solves": sum(
                        r["collocation_nonlinear_solves"] for r in result["intervals"]
                    ),
                    "accepted_steps": sum(len(r["steps"]) for r in result["intervals"]),
                    "rejected_attempts": sum(r["rejected_attempts"] for r in result["intervals"]),
                    "fallback_intervals": sum(r["fallback_intervals"] for r in result["intervals"]),
                    "conservation_residuals": result["conservation_residuals"],
                    "trajectory": f"{name}.json.gz",
                    "independent_audit": checked["accepted"],
                }
            )
    for path in paths:
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != fingerprint[str(path.relative_to(root))]
        ):
            raise RuntimeError(f"Source changed during benchmark: {path}")
    medians = {
        method: statistics.median(row["wall_s"] for row in rows if row["method"] == method)
        for method in methods
    }
    summary = {
        "scope": "supplied PR transport/machine graph; no gravitational columns or feedback controllers",
        "timing": "whole run, including model construction and initialization; preparation excluded",
        "reference": "same Radau3 physics with step cap; numerical sensitivity, not external validation",
        "reference_step_s": cap,
        "runtime": reference["runtime"],
        "reference_audit": reference_audit["accepted"],
        "source_sha256": fingerprint,
        "samples": rows,
        "median_wall_s": medians,
        "speedup": medians[methods[0]] / medians[methods[1]],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-step-s", type=float, default=1e-5)
    args = parser.parse_args()
    result = benchmark(
        json.loads(args.request.read_text()),
        args.output,
        repeats=args.repeats,
        reference_step_s=args.reference_step_s,
    )
    print(json.dumps({k: result[k] for k in ("median_wall_s", "speedup", "samples")}, indent=2))


if __name__ == "__main__":
    main()
