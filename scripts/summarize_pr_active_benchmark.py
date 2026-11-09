"""Compact, source-bound CPU/CUDA scaling summary; no hidden large state blobs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(a, b, *, path="state", worst=None):
    worst = [0.0, ""] if worst is None else worst
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("Benchmark state registry differs")
        for key in a:
            if key != "model_id":
                compare(a[key], b[key], path=path + "/" + key, worst=worst)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError("Benchmark state shape differs")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path=path + f"/{i}", worst=worst)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        if not isinstance(b, (float, int)) or not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("Nonfinite or mismatched state scalar")
        scaled = abs(a - b) / (1e-8 + 1e-8 * max(abs(a), abs(b)))
        if scaled > worst[0]:
            worst[:] = [scaled, path]
        if scaled > 1:
            raise ValueError(f"CPU/CUDA final-state parity failed at {path}")
    elif a != b:
        raise ValueError("Benchmark nonnumeric state differs")
    return worst


def run(cpu_paths, gpu_path):
    cpu = {}
    gpu = {}
    provenance = []
    reference = None
    for path in [*map(Path, cpu_paths), Path(gpu_path)]:
        report = json.loads(path.read_text())
        if report["format"] != "pr_active_surface_scaling_v1":
            raise ValueError("Unknown scaling format")
        for name, digest in report["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Stale benchmark")
        if reference is None:
            reference = report["source_sha256"]
        elif report["source_sha256"] != reference:
            raise ValueError("Different CPU/CUDA physical sources")
        target = cpu if report["device"] == "cpu" else gpu
        for case in report["cases"]:
            if case["cells"] in target:
                raise ValueError("Duplicate scaling point")
            target[case["cells"]] = case
        provenance.append(
            {
                "original_report": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "device": report["device"],
                "hardware": report["hardware"],
                "torch": report["torch"],
            }
        )
    if cpu.keys() != gpu.keys():
        raise ValueError("Matching CPU/CUDA sizes required")
    cases = []
    for size, a in sorted(cpu.items()):
        b = gpu[size]
        for key in [
            "branches",
            "columns",
            "equipment_vessels",
            "pump_rotors",
            "active_phases",
            "physical_step_s",
            "simulated_duration_s",
        ]:
            if a[key] != b[key]:
                raise ValueError("Different physical workload")
        worst = compare(a["final_state"], b["final_state"])
        aa = {k: v for k, v in a.items() if k != "final_state"}
        bb = {k: v for k, v in b.items() if k != "final_state"}
        cases.append(
            {
                "cells": size,
                "cpu": aa,
                "cuda": bb,
                "cuda_time_divided_by_cpu_time": b["mean_step_seconds"] / a["mean_step_seconds"],
                "checkpoint_parity_accepted": True,
                "maximum_parity_tolerance_fraction": worst[0],
                "worst_parity_path": worst[1],
            }
        )
    return {
        "format": "pr_active_surface_scaling_summary_v1",
        "qualification": "Real coupled short-step scaling, not long-horizon physical validation; no claim of GPU speedup",
        "source_sha256": reference,
        "original_reports": provenance,
        "cpu_threads": 1,
        "gpu_uuid": "GPU-2f9ce359-f3f4-accc-bf9c-31bc88bb1cb7",
        "cpu_hardware_observed": "AMD Ryzen 9 9900X 12-Core Processor",
        "conditions": "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1; other RTX3090 ran a separate job, RTX3090Ti had only this benchmark compute process",
        "timing_scope": "inclusive host wall times; CUDA synchronized at global step boundaries; parents and children must not be summed",
        "parity_atol": 1e-8,
        "parity_rtol": 1e-8,
        "cases": cases,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu", type=Path, nargs="+", required=True)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.cpu, args.cuda)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "cases": len(result["cases"]),
                "checkpoint_parity_accepted": True,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
