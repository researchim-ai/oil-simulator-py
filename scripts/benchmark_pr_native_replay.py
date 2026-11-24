"""Alternate private-buffer replay policies on one guarded native context.

Both policies use identical equations, compiled kernels, fresh physical states
and tolerances. Model initialization is inside timing; independent audits and
serialization are outside. Preparation is measured separately, never hidden.
"""

import argparse
import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path

from benchmark_pr_fast_integrator import measure
from check_pr_controlled_surface import audit

from surface.equilibrium_properties import EquilibriumFluid
from surface.native_execution import native_kernel_execution
from surface.torch_backend import runtime_fingerprint
from surface.torch_nonlinear import TorchNewton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maximum-native-kernels", type=int, default=16)
    args = parser.parse_args()
    if args.repeats < 1 or args.maximum_native_kernels < 1:
        parser.error("Positive repeat and kernel budgets required")
    request = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "src/surface").glob("*.py")),
        Path(__file__),
        root / "scripts/benchmark_pr_fast_integrator.py",
        root / "scripts/run_pr_controlled_surface.py",
        root / "scripts/check_pr_controlled_surface.py",
        root / "scripts/check_pr_subcycled_surface.py",
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"source_sha256": hashes}, indent=2) + "\n"
    )
    EquilibriumFluid._compiled_chemical_cpu = True
    EquilibriumFluid._compiled_full_eos_cpu = True
    TorchNewton._dense_jacobian_mode = "reverse"
    last_progress = [0.0]

    def progress(event):
        now = time.perf_counter()
        if event.get("event") == "native_kernel_compile" or now - last_progress[0] >= 20:
            print(json.dumps(event), flush=True)
            last_progress[0] = now

    results = {"clone_each_call": [], "private_buffers": []}
    states = None

    def execute(label, repeat):
        nonlocal states
        report, record = measure(request, native=True, progress=progress)
        with gzip.open(args.output_dir / f"{label}_{repeat}.json.gz", "wt") as handle:
            json.dump({"request": request, "report": report, **record}, handle, allow_nan=False)
        checked = audit(request, report)
        if checked.get("accepted") is not True:
            raise ValueError("Independent physical audit rejected replay")
        actual = [interval["state"] for interval in report["intervals"]]
        if states is None:
            states = actual
        if actual != states:
            raise ValueError("Input replay policy changed a physical reporting state")
        record.update(audit=checked, label=label, repeat=repeat)
        print(json.dumps({"event": "completed", **record}), flush=True)
        return record

    with native_kernel_execution(
        maximum_kernels=args.maximum_native_kernels, progress=progress
    ) as engine:
        preparation = execute("preparation", 0)
        # Stabilize fresh-model library preparation before either timed policy.
        settling = execute("settling", 0)
        for repeat in range(args.repeats):
            order = tuple(results) if repeat % 2 == 0 else tuple(reversed(results))
            for label in order:
                engine._reuse_input_buffers = label == "private_buffers"
                results[label].append(execute(label, repeat))
        execution = engine.statistics()
    unchanged = all(
        hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(root))]
        for p in paths
    )
    medians = {name: statistics.median(r["wall_s"] for r in runs) for name, runs in results.items()}
    summary = {
        "scope": __doc__,
        "runtime": runtime_fingerprint(),
        "preparation": preparation,
        "settling": settling,
        "results": results,
        "warm_median_wall_s": medians,
        "observed_speedup": medians["clone_each_call"] / medians["private_buffers"],
        "all_physical_reporting_states_identical": True,
        "independent_audits_passed": True,
        "sources_unchanged": unchanged,
        "native_execution": execution,
        "source_sha256": hashes,
        "accepted": unchanged and not execution["disabled"] and execution["fallback_calls"] == 0,
        "full_operating_cycle_qualified": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_sha256", "results")}, indent=2))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
