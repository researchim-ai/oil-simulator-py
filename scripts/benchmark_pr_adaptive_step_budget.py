"""Measure user-supplied PR schedules at different maximum steps, with refinement gates.

Only the numerical maximum step changes. Every candidate retains its actual
embedded-error, EOS, stability and conservation checks. This is not a forced
large-step integration or a case-dependent physics shortcut.
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

from benchmark_pr_native_execution import reference_errors
from check_pr_equilibrium_transport import audit
from run_pr_equilibrium_transport import _run

from surface.equilibrium_properties import EquilibriumFluid
from surface.native_execution import native_kernel_execution
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
from surface.torch_backend import runtime_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--maximum-steps", nargs="+", type=float, default=[0.1, 1.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(not math.isfinite(x) or x <= 0 for x in args.maximum_steps):
        parser.error("Positive step limits and at least one warm repeat required")
    spec = json.loads(args.request.read_text())
    config = spec.get("request", spec)
    if set(config["initial"]) != {"state"}:
        parser.error("A saved initial physical state is required")
    with gzip.open(args.reference, "rt") as handle:
        reference = json.load(handle)["result"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "src/surface").glob("*.py")),
        Path(__file__),
        root / "scripts/benchmark_pr_native_execution.py",
        root / "scripts/run_pr_equilibrium_transport.py",
        root / "scripts/audit_pr_collocation.py",
        root / "scripts/check_pr_equilibrium_transport.py",
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output_dir / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    EquilibriumFluid._compiled_chemical_cpu = True
    cases = []
    with native_kernel_execution(progress=lambda e: print(json.dumps(e), flush=True)) as engine:
        for case, limit in enumerate(args.maximum_steps):
            request = copy.deepcopy(config)
            request["model"]["maximum_step_s"] = limit
            model = PREquilibriumTransportNetwork.from_dict(request["model"])
            # The only configuration change is a numerical cap. Physical
            # checkpoint fields remain identical and inspect still validates
            # them normally under the new configuration identifier.
            request["initial"]["state"]["model_id"] = model.model_id
            runs = []
            for repeat in range(args.repeats + 1):
                entered = time.perf_counter()
                result = _run(
                    request, device="cpu", integrator="radau3-embedded", compiled_properties=True
                )
                wall = time.perf_counter() - entered
                with gzip.open(args.output_dir / f"case_{case}_{repeat}.json.gz", "wt") as handle:
                    json.dump({"result": result, "wall_s": wall}, handle, allow_nan=False)
                checked = audit(json.loads(json.dumps(request)), json.loads(json.dumps(result)))
                errors = reference_errors(result, reference)
                if errors["maximum_requested_error_norm"] > 1:
                    raise RuntimeError("Refined trajectory error budget exceeded")
                stats = engine.statistics()
                if stats["disabled"] or stats["fallback_calls"]:
                    raise RuntimeError("Native execution fell back; timing is not qualified")
                steps = [s for interval in result["intervals"] for s in interval["steps"]]
                requested = math.fsum(s["duration_s"] for s in request["schedule"])
                actual = result["intervals"][-1]["state"]["time_s"] - result["initial"]["time_s"]
                if not math.isclose(actual, requested, rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError("Incomplete physical interval")
                row = {
                    "maximum_step_s": limit,
                    "repeat": repeat,
                    "wall_s": wall,
                    "simulated_s": actual,
                    "accepted_steps": len(steps),
                    "largest_accepted_step_s": max(s["dt_s"] for s in steps),
                    "rejected_attempts": sum(i["rejected_attempts"] for i in result["intervals"]),
                    "fallback_intervals": sum(i["fallback_intervals"] for i in result["intervals"]),
                    "audit": checked,
                    "refinement_errors": errors,
                    "execution": stats,
                }
                runs.append(row)
                print(json.dumps(row), flush=True)
            cases.append(
                {
                    "maximum_step_s": limit,
                    "first": runs[0],
                    "warm": runs[1:],
                    "warm_median_wall_s": statistics.median(r["wall_s"] for r in runs[1:]),
                }
            )
    if any(hashlib.sha256((root / p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError("Sources changed during measurement")
    summary = {
        "scope": "Input numerical step-cap study; unchanged equations and error tolerances",
        "source_sha256": hashes,
        "request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "runtime": runtime_fingerprint(),
        "cases": cases,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
