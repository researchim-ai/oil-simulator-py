"""Cold/warm whole-schedule CUDA execution comparison with independent audits."""

import argparse
import faulthandler
import gzip
import hashlib
import json
import time
from pathlib import Path

import torch
from check_pr_controlled_surface import audit
from run_pr_controlled_surface import run

from surface.cuda_execution import cuda_graph_execution
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState
from surface.torch_nonlinear import TorchNewton

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("An accessible CUDA device is required")
    torch.cuda.set_per_process_memory_fraction(0.30)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__),
        ROOT / "scripts/run_pr_controlled_surface.py",
        args.config,
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"source_sha256": hashes, "config": str(args.config)}, indent=2) + "\n"
    )
    config = json.loads(args.config.read_text())
    faulthandler.dump_traceback_later(90, repeat=True)
    TorchNewton._dense_jacobian_mode = "reverse"
    results = {}

    def measure(name, accelerated):
        print(json.dumps({"event": "begin", "run": name}), flush=True)
        last_progress = [0.0]

        def progress(event):
            now = time.perf_counter()
            if now - last_progress[0] > 15:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "run": name,
                            "solver_event": event.get("event"),
                            "time_s": event.get("time_s"),
                        }
                    ),
                    flush=True,
                )
                last_progress[0] = now

        torch.cuda.synchronize()
        start = time.perf_counter()
        report = run(config, device="cuda", cuda_graphs=accelerated, progress=progress)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        report = json.loads(json.dumps(report))
        # Preserve the completed runtime result even if the independent audit
        # fails; such a file remains explicitly unqualified.
        with gzip.open(args.output_dir / (name + ".json.gz"), "wt") as handle:
            json.dump({"report": report, "wall_s": elapsed, "audit": None}, handle, allow_nan=False)
        checked = audit(config, report)
        if not checked["accepted"]:
            raise ValueError("Independent physical audit failed: " + name)
        with gzip.open(args.output_dir / (name + ".json.gz"), "wt") as handle:
            json.dump(
                {"report": report, "wall_s": elapsed, "audit": checked}, handle, allow_nan=False
            )
        results[name] = {
            "wall_s": elapsed,
            "independent_audit": checked,
            "execution": report["numerical_policy"].get("cuda_execution"),
            "attempts": sum(v["attempts"] for v in report["intervals"]),
        }
        print(json.dumps({"event": "completed", "run": name, **results[name]}), flush=True)
        return report

    with cuda_graph_execution(
        progress=lambda event: print(json.dumps(event), flush=True)
    ) as execution:
        measure("accelerated_cold", True)
        if execution.disabled:
            raise RuntimeError("Accelerated backend disabled: " + str(execution.statistics()))
        accelerated = measure("accelerated_warm", True)
        execution_stats = execution.statistics()
    measure("reference_cold", False)
    reference = measure("reference_warm", False)
    faulthandler.cancel_dump_traceback_later()
    model = PRControlledSurface.from_dict(config["model"])
    before = PRControlledSurfaceState.from_dict(reference["intervals"][-1]["state"])
    after = PRControlledSurfaceState.from_dict(accelerated["intervals"][-1]["state"])
    difference = model._error(before, after)
    accepted = (
        before.time_s == after.time_s
        and difference < 0.01
        and execution_stats["captures"] > 0
        and execution_stats["replays"] > 0
    )
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in paths):
        raise RuntimeError("Sources changed during whole-network comparison")
    summary = {
        "accepted": accepted,
        "scope": "complete requested schedule including initialization; independent audits outside timing",
        "simulated_duration_s": sum(c["duration_s"] for c in config["schedule"]),
        "runs": results,
        "difference_in_original_error_norm": difference,
        "warm_observed_speedup": results["reference_warm"]["wall_s"]
        / results["accelerated_warm"]["wall_s"],
        "execution": execution_stats,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "source_sha256": hashes,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
