"""Paired real hydraulic or full-station compact journal qualification.

Compares all retained records, complete states, controller/protection/thermal
logs and original numerical acceptance. Packet counters are logical API
traffic, not a measurement of every CUDA driver transfer.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from benchmark_liquid_state_transfer import delta, digest, sync
from check_liquid_transient import wave_model, wave_reference

from surface import (
    LiquidTransientNetwork,
    LiquidTransientState,
    ThermalLiquidStation,
    run_thermal_liquid_schedule,
)
from surface.liquid_journal import compact_record
from surface.torch_backend import TORCH_SOURCES, liquid_backend_config, runtime_fingerprint
from surface.torch_heat_bridge import TorchHeatBridge
from surface.torch_liquid_bridge import TorchLiquidBridge


def normalize(result, branches):
    return {**result, "steps": [compact_record(r, branches) for r in result["steps"]]}


def run_station(config, device, modes, save):
    data = json.loads(config.read_text())
    rows, reference = [], None
    for mode in modes:
        model = ThermalLiquidStation.from_dict(
            liquid_backend_config(data["system"], "torch", device)
        )
        initial = model.initialize(**data["initial"])
        physics = model.recycling.physics
        physics._torch_bridge = TorchLiquidBridge(physics, device)
        model.heat._torch_bridge = TorchHeatBridge(model.heat, device)
        overrides = {
            "hydraulic_journal": mode,
            "hydraulic_heat_transfer": "device",
            "thermal_execution": "auto",
        }
        command = {k: v for k, v in data["periods"][0].items() if k != "duration_s"}
        model.advance(initial, 0.001, **{**command, **overrides})
        bridges = {"mechanical": physics._torch_bridge, "thermal": model.heat._torch_bridge}
        before = {k: b.transfer_stats() for k, b in bridges.items()}
        periods = [{**p, **overrides} for p in data["periods"]]
        sync(device)
        load_start = os.getloadavg()
        start = time.perf_counter()
        result = run_thermal_liquid_schedule(model, initial, periods)
        sync(device)
        wall = time.perf_counter() - start
        loops = tuple(x.valve for x in model.recycling.loops)
        projected = {
            "state": result["state"],
            "records": (
                [
                    normalize(r, (*loops, *(periods[i].get("hydraulic_journal_branches") or ())))
                    for i, r in enumerate(result["records"])
                ]
                if mode == "full"
                else result["records"]
            ),
        }
        key = digest(projected)
        if reference is None:
            reference = key
        if save:
            save.mkdir(parents=True, exist_ok=True)
            (save / (mode + ".json")).write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
        row = {
            "kind": "complete_station",
            "mode": mode,
            "device": device,
            "wall_seconds": wall,
            "loadavg_start": load_start,
            "loadavg_end": os.getloadavg(),
            "common_projection_sha256": key,
            "passed": key == reference,
            "raw_physical_sha256": digest({"state": result["state"], "records": result["records"]}),
            "accepted_steps": sum(len(r["steps"]) for r in result["records"]),
            "retries": sum(r["failed_steps"] for r in result["records"]),
            "max_combined_energy_residual_j": max(
                abs(a["combined_energy_residual_j"])
                for r in result["records"]
                for a in r["thermal_audits"]
            ),
            "accepted_journal_json_bytes": sum(
                len(json.dumps(r["steps"], separators=(",", ":"), allow_nan=False).encode())
                for r in result["records"]
            ),
            "transfer_stats": {k: delta(before[k], b.transfer_stats()) for k, b in bridges.items()},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def run_waves(cells, device, modes, intervals):
    rows, reference = [], None
    for mode in modes:
        model = LiquidTransientNetwork.from_dict(
            {**wave_model(cells).to_dict(), "device": device, "temporal_error_control": True}
        )
        initial = model.initialize(dict(zip(model.free, wave_reference(cells, 0))))
        ceiling = min(
            model.maximum_dt_s, model.acoustic_courant * 5 / cells / math.sqrt(1e8 / 1000)
        )
        model.advance(initial, ceiling, hydraulic_journal=mode)
        before = model._torch_bridge.transfer_stats()
        sync(device)
        start = time.perf_counter()
        state = initial
        records = []
        for _ in range(intervals):
            result = model.advance(
                state, 4 * ceiling, hydraulic_journal=mode, hydraulic_journal_branches=["p0"]
            )
            records.append(result)
            state = LiquidTransientState(**result["state"])
        sync(device)
        wall = time.perf_counter() - start
        key = digest([normalize(r, ("p0",)) for r in records] if mode == "full" else records)
        if reference is None:
            reference = key
        pressures = dict(state.pressures_pa)
        row = {
            "kind": "acoustic_wave",
            "cells": cells,
            "hydraulic_unknowns": 2 * cells + 1,
            "mode": mode,
            "device": device,
            "wall_seconds": wall,
            "common_projection_sha256": key,
            "passed": key == reference,
            "accepted_steps": sum(len(r["steps"]) for r in records),
            "final_time_s": state.time_s,
            "max_pressure_error_pa": float(
                max(
                    abs(pressures[n] - v)
                    for n, v in zip(model.free, wave_reference(cells, state.time_s))
                )
            ),
            "accepted_journal_json_bytes": sum(
                len(json.dumps(r["steps"], separators=(",", ":"), allow_nan=False).encode())
                for r in records
            ),
            "transfer_stats": {"mechanical": delta(before, model._torch_bridge.transfer_stats())},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cells", type=int, nargs="+", default=[32, 80, 256])
    parser.add_argument("--intervals", type=int, default=3)
    parser.add_argument(
        "--modes", nargs="+", choices=("full", "compact"), default=["full", "compact"]
    )
    parser.add_argument("--save-runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.intervals < 1
        or any(n < 1 for n in args.cells)
        or len(set(args.modes)) != len(args.modes)
    ):
        parser.error("Positive sizes and unique modes required")
    source_files = (
        *TORCH_SOURCES,
        "liquid_transient.py",
        "controlled_liquid.py",
        "liquid_protection.py",
        "liquid_recycle.py",
        "liquid_heat.py",
        "thermal_liquid.py",
        "thermal_liquid_schedule.py",
    )
    source = {
        n: hashlib.sha256((ROOT / "src/surface" / n).read_bytes()).hexdigest() for n in source_files
    }
    rows = (
        run_station(args.config, args.device, args.modes, args.save_runs)
        if args.config
        else [
            row for n in args.cells for row in run_waves(n, args.device, args.modes, args.intervals)
        ]
    )
    if source != {
        n: hashlib.sha256((ROOT / "src/surface" / n).read_bytes()).hexdigest() for n in source_files
    }:
        raise RuntimeError("Numerical source changed during benchmark")
    output = {
        "passed": all(r["passed"] for r in rows),
        "runtime": runtime_fingerprint(),
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "source": source,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()
        if args.config
        else None,
        "thread_environment": {
            n: os.environ.get(n) for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if not output["passed"]:
        raise SystemExit("Compact projection mismatch")


if __name__ == "__main__":
    main()
