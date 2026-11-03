"""Paired actual hydraulic API traffic, physical waves and full coupled station.

Every accepted record, guard/refinement and stop event is compared exactly.
Reported payload counters cover the adapter, with separate CUDA Newton status
read counts; they are not an Nsight measurement of all driver/library traffic.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_liquid_transient import wave_model, wave_reference

from surface import (
    LiquidTransientNetwork,
    LiquidTransientState,
    ThermalLiquidStation,
    run_thermal_liquid_schedule,
)
from surface.torch_backend import TORCH_SOURCES, liquid_backend_config, runtime_fingerprint
from surface.torch_liquid_bridge import TorchLiquidBridge


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def delta(before, after):
    return {k: after[k] - before[k] for k in before}


def waves(cells, device, modes, intervals, macro_steps):
    start = time.perf_counter()
    data = wave_model(cells).to_dict()
    data.update(device=device, temporal_error_control=True)
    model = LiquidTransientNetwork.from_dict(data)
    initial = model.initialize(dict(zip(model.free, wave_reference(cells, 0))))
    ceiling = min(
        [model.maximum_dt_s]
        + [
            model.acoustic_courant
            * p.length_m
            / math.sqrt(model.bulk_modulus_pa / model.network.fluid.density_kg_m3)
            for p in model.network.pipes
        ]
    )
    duration = ceiling * macro_steps
    model.advance(initial, ceiling, hydraulic_state_transfer="solve")
    sync(device)
    setup = time.perf_counter() - start
    rows = []
    reference = None
    for mode in modes:
        before = model._torch_bridge.transfer_stats()
        sync(device)
        start = time.perf_counter()
        state = initial
        results = []
        for _ in range(intervals):
            result = model.advance(state, duration, hydraulic_state_transfer=mode)
            results.append(result)
            state = LiquidTransientState(**result["state"])
        sync(device)
        elapsed = time.perf_counter() - start
        key = digest(results)
        if reference is None:
            reference = key
        pressures = dict(state.pressures_pa)
        actual = np.asarray([pressures[n] for n in model.free])
        error = float(np.max(np.abs(actual - wave_reference(cells, state.time_s))))
        row = {
            "kind": "physical_acoustic_wave",
            "cells": cells,
            "unknowns": 2 * cells + 1,
            "device": device,
            "mode": mode,
            "wall_seconds": elapsed,
            "setup_and_warmup_seconds": setup,
            "intervals": intervals,
            "duration_per_interval_s": duration,
            "final_time_s": state.time_s,
            "accepted_steps": sum(len(r["steps"]) for r in results),
            "retries": sum(r["failed_steps"] for r in results),
            "pressure_error_vs_mode_pa": error,
            "sparse_matrix_nnz": model._torch_bridge.equations.newton.matrix.values.numel()
            if model._torch_bridge.equations.newton.matrix is not None
            else None,
            "jacobian_colors": model._torch_bridge.equations.newton.coloring.seeds.shape[0]
            if model._torch_bridge.equations.newton.coloring is not None
            else None,
            "physical_sha256": key,
            "passed": key == reference and error < 0.1,
            "transfer_stats": delta(before, model._torch_bridge.transfer_stats()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def coupled(config, device, modes, save):
    data = json.loads(config.read_text())
    rows = []
    reference = None
    for mode in modes:
        start = time.perf_counter()
        model = ThermalLiquidStation.from_dict(
            liquid_backend_config(data["system"], "torch", device)
        )
        initial = model.initialize(**data["initial"])
        physics = model.recycling.physics
        physics._torch_bridge = TorchLiquidBridge(physics, device)
        sync(device)
        setup = time.perf_counter() - start
        before = physics._torch_bridge.transfer_stats()
        periods = [
            {**p, "hydraulic_state_transfer": mode, "thermal_state_transfer": "report"}
            for p in data["periods"]
        ]
        sync(device)
        start = time.perf_counter()
        result = run_thermal_liquid_schedule(model, initial, periods)
        sync(device)
        elapsed = time.perf_counter() - start
        key = digest({"state": result["state"], "records": result["records"]})
        if reference is None:
            reference = key
        if save:
            save.mkdir(parents=True, exist_ok=True)
            (save / (mode + ".json")).write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
        row = {
            "kind": "actual_coupled_station_schedule",
            "mode": mode,
            "device": device,
            "setup_seconds": setup,
            "wall_seconds": elapsed,
            "physical_sha256": key,
            "passed": key == reference,
            "accepted_steps": sum(len(r["steps"]) for r in result["records"]),
            "retries": sum(r["failed_steps"] for r in result["records"]),
            "max_combined_energy_residual_j": max(
                abs(x["combined_energy_residual_j"])
                for r in result["records"]
                for x in r["thermal_audits"]
            ),
            "transfer_stats": delta(before, physics._torch_bridge.transfer_stats()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cells", type=int, nargs="+", default=[32, 80, 256])
    p.add_argument("--intervals", type=int, default=3)
    p.add_argument("--macro-steps", type=int, default=4)
    p.add_argument(
        "--modes", nargs="+", choices=("solve", "interval"), default=["solve", "interval"]
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--save-runs", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.intervals < 1 or a.macro_steps < 1 or any(n < 1 for n in a.cells):
        p.error("Positive work sizes required")
    if len(set(a.modes)) != len(a.modes):
        p.error("Duplicate modes")
    rows = (
        coupled(a.config, a.device, a.modes, a.save_runs)
        if a.config
        else [
            row for n in a.cells for row in waves(n, a.device, a.modes, a.intervals, a.macro_steps)
        ]
    )
    files = (
        *TORCH_SOURCES,
        "liquid_transient.py",
        "controlled_liquid.py",
        "liquid_protection.py",
        "thermal_liquid.py",
    )
    output = {
        "passed": all(r["passed"] for r in rows),
        "runtime": runtime_fingerprint(),
        "gpu": torch.cuda.get_device_name() if a.device.startswith("cuda") else None,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {
            n: hashlib.sha256((ROOT / "src/surface" / n).read_bytes()).hexdigest() for n in files
        },
        "thread_environment": {
            n: os.environ.get(n) for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "results": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
