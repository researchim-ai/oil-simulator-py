"""Measure complete source-bound surface transactions with exact caches on/off.

Both paths execute the requested schedule, receive independent physical audits,
and must produce identical physical checkpoints. A timeout fails the benchmark;
no completed microstep is substituted for the requested operating interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def fingerprints(config):
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__), config]
    paths += [
        ROOT / "scripts" / name
        for name in (
            "run_pr_controlled_surface.py",
            "run_pr_equilibrium_transport.py",
            "check_pr_controlled_surface.py",
            "check_pr_equilibrium_transport.py",
        )
    ]
    paths += sorted((ROOT / "scripts").glob("reference_*.py"))
    paths += sorted((ROOT / "scripts").glob("check_stratified_*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def worker(args):
    from contextlib import contextmanager

    from surface import equilibrium_cache as memo

    config = json.loads(args.config.read_text())
    controlled = "plant" in config["model"]
    if controlled:
        from run_pr_controlled_surface import run
    else:
        from run_pr_equilibrium_transport import run
    original_flash, original_stability = memo.cached_flashes, memo.cached_stability_checks
    statistics = []

    def instrument(original, kind):
        @contextmanager
        def context(fluid, **kwargs):
            if args.worker == "uncached":
                kwargs["enabled"] = False
            with original(fluid, **kwargs) as cache:
                try:
                    yield cache
                finally:
                    statistics.append({"kind": kind, **cache.statistics()})

        return context

    memo.cached_flashes = instrument(original_flash, "flash")
    memo.cached_stability_checks = instrument(original_stability, "stability")
    try:
        start = time.perf_counter()
        result = run(config, device=args.device)
        elapsed = time.perf_counter() - start
    finally:
        memo.cached_flashes, memo.cached_stability_checks = original_flash, original_stability
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    args.output.with_suffix(".timing.json").write_text(
        json.dumps(
            {
                "elapsed_s": elapsed,
                "caches": statistics,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("uncached", "cached"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.config = args.config.resolve()
    if args.worker:
        worker(args)
        return
    import math

    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("Finite positive timeout required per trajectory")
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Explicit CUDA_VISIBLE_DEVICES required")
    before = fingerprints(args.config)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = output.with_suffix(".reports")
    if output.exists() or directory.exists():
        raise FileExistsError("Use a fresh output path; completed evidence is immutable")
    directory.mkdir()
    result = {
        "format": "surface_transaction_cache_benchmark_v1",
        "accepted": False,
        "device": args.device,
        "source_sha256": before,
        "runs": {},
        "qualification": "Exact-cache performance and numerical parity on the requested synthetic schedule; not experimental or field-scale qualification",
    }
    try:
        config = json.loads(args.config.read_text())
        checker = (
            "check_pr_controlled_surface.py"
            if "plant" in config["model"]
            else "check_pr_equilibrium_transport.py"
        )
        for mode in ("uncached", "cached"):
            report = directory / (mode + ".json")
            log = directory / (mode + ".log")
            with log.open("x") as handle:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        str(args.config),
                        "--worker",
                        mode,
                        "--device",
                        args.device,
                        "--output",
                        str(report),
                    ],
                    stdout=handle,
                    stderr=handle,
                    timeout=args.timeout_s,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(f"{mode} worker failed: inspect {log.name}")
            checked = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / checker),
                    str(args.config),
                    str(report),
                ],
                capture_output=True,
                text=True,
                timeout=args.timeout_s,
                check=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            if checked.returncode:
                raise RuntimeError(f"{mode} independent audit failed: {checked.stderr[-2000:]}")
            audit = json.loads(checked.stdout)
            if audit.get("accepted") is not True:
                raise ValueError(f"{mode} physical audit rejected")
            result["runs"][mode] = {
                **json.loads(report.with_suffix(".timing.json").read_text()),
                "audit": audit,
                "report": f"{directory.name}/{report.name}",
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            }
        reports = [
            json.loads((directory / (mode + ".json")).read_text())
            for mode in ("uncached", "cached")
        ]
        # Compare the entire numerical trajectory, including diagnostics, not
        # only a small selected set of pressure/flow summaries.
        for key in (
            "config",
            "model",
            "initial",
            "intervals",
            "conservation_residuals",
            "total_heat_j",
            "total_work_j",
        ):
            if reports[0][key] != reports[1][key]:
                raise ValueError(f"Exact transaction parity failed: {key}")
        result["identical_complete_trajectory"] = True
        result["speedup"] = (
            result["runs"]["uncached"]["elapsed_s"] / result["runs"]["cached"]["elapsed_s"]
        )
        result["accepted"] = True
    except (subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError) as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
    result["sources_unchanged"] = fingerprints(args.config) == before
    if not result["sources_unchanged"]:
        result["accepted"] = False
        result["failure"] = "Sources/config changed during benchmark"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("accepted", "device", "sources_unchanged")}, indent=2))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
