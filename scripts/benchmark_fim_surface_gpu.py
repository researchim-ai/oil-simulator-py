"""Matched CPU/hybrid/CUDA-quadrature benchmark on a synthetic closed reservoir.

No automatic device selection or CPU fallback: mask the intended GPU before
launch. Field arrays, integrals, graph status and cold/steady times are saved.
"""

import argparse
import json
import platform
import statistics
from pathlib import Path

import numpy as np
import torch
from run_fim_blackoil_surface import run_case

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fim_blackoil_surface.json")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 2 or args.days < 2 or args.repeats < 1:
        parser.error("size>=2, days>=2, repeats>=1 required")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; no implicit CPU fallback")
    cfg = json.loads(args.config.read_text())
    cfg["fluid"]["pvt_path"] = str(ROOT / cfg["fluid"]["pvt_path"])
    cfg["reservoir"]["dimensions"] = [args.size, args.size, 1]
    if len(cfg["wells"]) != 2:
        parser.error("This benchmark requires the two-producer example")
    cfg["wells"][0].update(i=0, j=0, k=0)
    cfg["wells"][1].update(i=args.size - 1, j=args.size - 1, k=0)
    rows = []
    for device, backend in [
        ("cpu", "cpu_reference"),
        ("cuda", "cpu_reference"),
        ("cuda", "torch_device"),
    ]:
        c = json.loads(json.dumps(cfg))
        c["simulation"]["physical_surface_network"]["network_backend"] = backend
        print(f"{device}/{backend} unmeasured one-interval warmup", flush=True)
        run_case(c, 1e6, 1.0, 1.0, device)
        runs = []
        for i in range(args.repeats):
            print(f"{device}/{backend} repeat {i + 1}/{args.repeats}", flush=True)
            r = run_case(c, 1e6, args.days, 1.0, device)
            if r["elapsed_days"] != args.days:
                raise RuntimeError("Report interval not completed")
            if device == "cuda" and not r["field_graphs_active"]:
                raise RuntimeError("Reservoir CUDA graphs were disabled")
            runs.append(r)
        steady = [t["wall_s"] for r in runs for t in r["interval_timings"][1:]]
        rows.append(
            {
                "device": device,
                "network_backend": backend,
                "runs": runs,
                "median_total_wall_s": statistics.median(r["solve_wall_s"] for r in runs),
                "median_cold_interval_s": statistics.median(
                    r["interval_timings"][0]["wall_s"] for r in runs
                ),
                "median_steady_interval_s": statistics.median(steady),
            }
        )
    reference = rows[0]["runs"][0]
    checks = []
    for row in rows[1:]:
        for run in row["runs"]:
            errors = {
                key: float(np.max(np.abs(np.array(run[key]) - np.array(reference[key]))))
                for key in ("final_pressure_pa", "final_s_w", "final_s_g", "final_rs")
            }
            if (
                errors["final_pressure_pa"] > 1
                or max(errors["final_s_w"], errors["final_s_g"]) > 1e-7
                or errors["final_rs"] > 1e-5
            ):
                raise RuntimeError(f"CPU/GPU field discrepancy: {errors}")
            a = np.array(run["withdrawn_component_mass_kg"])
            b = np.array(reference["withdrawn_component_mass_kg"])
            mass_relative = float(np.max(np.abs(a - b) / np.maximum(1, np.abs(b))))
            if mass_relative > 1e-6:
                raise RuntimeError("CPU/GPU component withdrawal discrepancy")
            checks.append(
                {
                    "device": row["device"],
                    "network_backend": row["network_backend"],
                    "field_absolute_errors": errors,
                    "component_withdrawal_relative_error": mass_relative,
                }
            )
    hardware = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = {
        "model": "synthetic coupled black-oil reservoir/gathering",
        "cells": args.size**2,
        "grid": cfg["reservoir"]["dimensions"],
        "days": args.days,
        "repeats": args.repeats,
        "hardware": {
            "name": hardware.name,
            "visible_device_count": torch.cuda.device_count(),
            "uuid": str(getattr(hardware, "uuid", "unknown")),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "configuration": cfg,
        "results": rows,
        "parity_checks": checks,
        "scope": "matched synthetic case; one unmeasured warmup per backend; cold interval includes new-model graph capture, steady excludes first interval; no Norne extrapolation; tree/root orchestration remains host-side",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "runs"} for row in rows], indent=2))


if __name__ == "__main__":
    main()
