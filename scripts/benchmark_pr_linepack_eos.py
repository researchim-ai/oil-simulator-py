"""Paired CPU timings of identical linepack equations with grouped endpoint EOS."""

import argparse
import gzip
import hashlib
import json
import statistics
import tarfile
import time
from pathlib import Path

from check_pr_linepack import audit
from prepared_surface_operators import operator_preparation
from run_pr_linepack import run


def benchmark(config, output, repeats=2, *, experiment="eos"):
    from surface.stratified_pr_network import StratifiedNetworkState, StratifiedPRNetwork
    from surface.torch_linear import TorchNewton as ReferenceNewton
    from surface.torch_nonlinear import TorchNewton

    with operator_preparation(True):
        pass  # Import both execution branches before starting any timer.

    if type(repeats) is not int or repeats < 1:
        raise ValueError("Positive integer repeat count required")
    if experiment not in ("eos", "integrators", "residual"):
        raise ValueError("Unknown benchmark experiment")
    methods = {
        "eos": ("original", "grouped"),
        "integrators": ("reference", "newton", "embedded"),
        "residual": ("reference", "prepared"),
    }[experiment]
    if experiment == "residual":
        config = json.loads(json.dumps(config))
        config["model"]["integration"]["temporal_error_control"] = "embedded_be"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src/surface").glob("*.py")) + sorted((root / "scripts").glob("*.py"))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output / "request.json").write_text(json.dumps(config, indent=2) + "\n")
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(root)))
    # Warm imports outside timings. No solver/flash warmup is removed from
    # measured runs: every sample constructs and initializes the entire model.
    from surface.pr_linepack import PRLinepackNetwork

    samples, comparisons = [], []
    previous = StratifiedPRNetwork._batch_connector_eos
    previous_newton = StratifiedPRNetwork._newton_type
    previous_warm = StratifiedPRNetwork._warm_darcy
    try:
        for repeat in range(repeats):
            results = {}
            for method in methods if repeat % 2 == 0 else methods[::-1]:
                request = json.loads(json.dumps(config))
                StratifiedPRNetwork._batch_connector_eos = method != "original"
                StratifiedPRNetwork._newton_type = previous_newton
                prepare = experiment == "residual" and method == "prepared"
                if experiment == "integrators":
                    StratifiedPRNetwork._warm_darcy = method != "reference"
                    StratifiedPRNetwork._newton_type = (
                        ReferenceNewton if method == "reference" else TorchNewton
                    )
                    request["model"]["integration"].pop("temporal_error_control", None)
                    if method == "embedded":
                        request["model"]["integration"]["temporal_error_control"] = "embedded_be"
                start = time.perf_counter()
                with operator_preparation(prepare):
                    result = run(request, device="cpu")
                wall = time.perf_counter() - start
                result = json.loads(json.dumps(result, allow_nan=False))
                certificate = audit(request, result)
                if certificate.get("accepted") is not True:
                    raise ValueError("Independent linepack audit rejected")
                name = f"repeat-{repeat}-{method}.json.gz"
                with gzip.open(output / name, "wt") as stream:
                    json.dump(result, stream, allow_nan=False)
                attempts = [
                    a for i in result["intervals"] for s in i["steps"] for a in s["solver_attempts"]
                ]
                counted = bool(attempts) and all(
                    a.get("jacobian_builds") is not None for a in attempts
                )
                row = {
                    "repeat": repeat,
                    "method": method,
                    "wall_s": wall,
                    "audit": certificate,
                    "trajectory": name,
                    "accepted_substeps": sum(len(i["steps"]) for i in result["intervals"]),
                    "nonlinear_solves": sum(
                        i.get("nonlinear_solves", 0) for i in result["intervals"]
                    ),
                    "embedded_steps": sum(i.get("embedded_steps", 0) for i in result["intervals"]),
                    "probe_failures": sum(
                        i.get("embedded_probe_failures", 0) for i in result["intervals"]
                    ),
                    "jacobian_builds": sum(a["jacobian_builds"] for a in attempts)
                    if counted
                    else None,
                    "jacobian_reuses": sum(a.get("jacobian_reuses", 0) for a in attempts)
                    if counted
                    else None,
                    "jacobian_counter_scope": "accepted physical steps only; null when unsupported",
                    "accepted_cross_solve_reuses": sum(
                        a.get("cross_solve_reuses", 0) for a in attempts
                    ),
                    "accepted_cross_solve_refreshes": sum(
                        a.get("cross_solve_refreshes", 0) for a in attempts
                    ),
                }
                print(json.dumps(row), flush=True)
                samples.append(row)
                (output / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
                results[method] = result
            model = PRLinepackNetwork.from_dict(config["model"], device="cpu")
            differences = [
                model.network._error(
                    dict(StratifiedNetworkState.from_dict(a["state"]).vessels),
                    dict(StratifiedNetworkState.from_dict(b["state"]).vessels),
                )
                for method in methods[1:]
                for a, b in zip(results[methods[0]]["intervals"], results[method]["intervals"])
            ]
            maximum = max(differences)
            if maximum > 1.0:
                raise ValueError("Implementation changed trajectory beyond original temporal norm")
            comparisons.append(
                {"repeat": repeat, "maximum_interval_difference_in_original_norm": maximum}
            )
        for path in paths:
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(root))]:
                raise RuntimeError(f"Source changed during benchmark: {path}")
        medians = {
            method: statistics.median(row["wall_s"] for row in samples if row["method"] == method)
            for method in methods
        }
        result = {
            "accepted": True,
            "device": "cpu",
            "repeats": repeats,
            "experiment": experiment,
            "timing_scope": "construction, initialization and solve; excludes independent audit/serialization",
            "simulation_duration_s": sum(c["duration_s"] for c in config["schedule"]),
            "median_wall_s": medians,
            "speedup": medians[methods[0]] / medians[methods[-1]],
            "speedups": {method: medians[methods[0]] / medians[method] for method in methods[1:]},
            "samples": samples,
            "comparisons": comparisons,
            "source_sha256": hashes,
            "scope": "same equations, quadrature and tolerances; synthetic linepack only; no GPU/field qualification",
        }
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        StratifiedPRNetwork._batch_connector_eos = previous
        StratifiedPRNetwork._newton_type = previous_newton
        StratifiedPRNetwork._warm_darcy = previous_warm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--experiment", choices=("eos", "integrators", "residual"), default="eos")
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                json.loads(args.config.read_text()),
                args.output,
                args.repeats,
                experiment=args.experiment,
            ),
            indent=2,
        )
    )
