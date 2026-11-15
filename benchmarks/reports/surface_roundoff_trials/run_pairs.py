"""Bounded, alternating-order CPU qualification; run from the repository root."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--mode", choices=("roundoff", "shared_probe"), default="roundoff")
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("positive pair count required")
    root = Path.cwd()
    sys.path.insert(0, str(root / "scripts"))
    from check_pr_controlled_surface import audit

    template_path = Path(__file__).with_name("instrument_template.py")
    template = template_path.read_text()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join(str(root / p) for p in ("src", "scripts", "tests")),
    )
    results = []
    samples = {"reference": [], "optimized": []}
    source_hashes = None
    for pair in range(args.pairs):
        pair_reports = {}
        names = ("reference", "optimized") if pair % 2 == 0 else ("optimized", "reference")
        for name in names:
            stem = output / f"pair{pair}_{name}"
            report_path = Path(str(stem) + "_report.json")
            stats_path = Path(str(stem) + "_instrumentation.json")
            policy = (
                "TorchNewton._reuse_identical_trials=False\nTorchNewton._stagnation_patience=4"
                if name == "reference" else
                "TorchNewton._reuse_identical_trials=True\nTorchNewton._stagnation_patience=2"
            )
            if args.mode == "roundoff":
                # Preserve this historical workload after probe sharing
                # became the production default in a subsequent change.
                policy += "\nPRControlledSurface._share_probe_quadrature=False"
            else:
                policy = (
                    "TorchNewton._reuse_identical_trials=True\nTorchNewton._stagnation_patience=2\n"
                    f"PRControlledSurface._share_probe_quadrature={name == 'optimized'!r}"
                )
            code = template.replace("cfg=json.loads", policy + "\ncfg=json.loads")
            code = code.replace(
                repr("/tmp/surface_nonlinear_final_report.json"), repr(str(report_path))
            ).replace(
                repr("/tmp/surface_nonlinear_final_instrumentation.json"), repr(str(stats_path))
            )
            code = code.replace(
                "'unknowns':x.numel()",
                "'identical_trial_skips':self.identical_trial_skips,"
                "'solution_sha256':hashlib.sha256(ans[0].detach().cpu().numpy().tobytes()).hexdigest(),"
                "'unknowns':x.numel()",
            )
            driver = Path(str(stem) + ".py")
            driver.write_text(code)
            print(f"Running pair {pair}: {name}", flush=True)
            with Path(str(stem) + ".log").open("w") as log:
                subprocess.run(
                    [sys.executable, str(driver)], cwd=root, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, timeout=160, check=True,
                )
            report = json.loads(report_path.read_text())
            pair_reports[name] = report
            stats = json.loads(stats_path.read_text())
            checked = audit(report["config"], report)
            if source_hashes is None:
                source_hashes = stats["source_sha256"]
            if source_hashes != stats["source_sha256"]:
                raise RuntimeError("Source generations differ across benchmark cases")
            result = {
                "pair": pair, "policy": name, "elapsed_s": stats["elapsed_s"],
                "global_solves": len(stats["global_newton_solves"]),
                "residual_evaluations": sum(s["evaluations"] for s in stats["global_newton_solves"]),
                "jacobian_builds": sum(s["jacobian_builds"] for s in stats["global_newton_solves"]),
                "audit": checked,
            }
            results.append(result)
            samples[name].append(result["elapsed_s"])
            print(json.dumps(result), flush=True)
        if args.mode == "shared_probe":
            left = pair_reports["reference"]["intervals"]
            right = pair_reports["optimized"]["intervals"]
            if len(left) != len(right) or any(
                a["state"] != b["state"] or a["steps"] != b["steps"]
                for a, b in zip(left, right)
            ):
                raise RuntimeError("Shared probe changed the accepted state/history")
    result = {
        "scope": "One-microsecond CPU startup; not whole-cycle throughput",
        "whole_operating_cycle_qualified": False,
        "mode": args.mode,
        "cases": results,
        "median_s": {k: statistics.median(v) for k, v in samples.items()},
        "source_sha256": source_hashes,
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
    }
    (output / "timings.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
