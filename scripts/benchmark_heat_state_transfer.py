"""Paired measurement of thermal field traffic through the actual station API.

Transport probes prescribe conservative hydraulic fluxes: they do not solve a
large hydraulic model. --config instead runs every real coupled station period,
including controllers/trips; --save-runs preserves its source-bound checkpoints.
GPU timings synchronize. Histories and accepted audits are compared exactly;
only execution counters and the schedule fingerprint differ between modes.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_thermal_liquid import pipe_probe

from surface import (
    LiquidHeatNetwork,
    LiquidHeatState,
    ThermalLiquidStation,
    run_thermal_liquid_schedule,
)
from surface.torch_backend import TORCH_SOURCES, liquid_backend_config, runtime_fingerprint
from surface.torch_heat_bridge import TorchHeatBridge


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def difference(before, after):
    return {k: after[k] - before[k] for k in before}


def transport_case(cells, device, modes, record_count, window_size):
    setup = time.perf_counter()
    model, _, base = pipe_probe(cells=cells, dt=0.25 / cells, duration=0.25 / cells, compact=True)
    data = model.to_dict()
    data["physics"]["device"] = data["device"] = device
    m = LiquidHeatNetwork.from_dict(data)
    initial = m.initialize(m.physics.initialize())
    records = [dict(copy.deepcopy(base), time_s=i * base["dt_s"]) for i in range(record_count)]
    m.advance_record(initial, records[0])  # equal kernel warmup, no input mutation
    sync(device)
    setup = time.perf_counter() - setup
    results = []
    reference = None
    for mode in modes:
        before = m._torch_bridge.transfer_stats()
        sync(device)
        start = time.perf_counter()
        cursor = initial
        steps, audits = [], []
        failed = factors = 0
        session = (
            m.open_session(initial, fluid_sensors=("a", "line::0")) if mode == "report" else None
        )
        for index in range(0, len(records), window_size):
            window = records[index : index + window_size]
            if session:
                calc = session.advance_records(window)
            else:
                calc = m.advance_records(cursor, window, resident=mode == "window")
                cursor = LiquidHeatState(**calc["state"])
            steps.extend(calc["steps"])
            audits.extend(calc["audits"])
            failed += calc["failed_steps"]
            factors += calc["linear_factorizations"]
        if session:
            cursor = session.close()
        sync(device)
        wall = time.perf_counter() - start
        result = {
            "state": cursor.to_dict(),
            "steps": steps,
            "audits": audits,
            "failed_steps": failed,
            "linear_factorizations": factors,
        }
        fingerprint = digest(result)
        if reference is None:
            reference = fingerprint
        stats = difference(before, m._torch_bridge.transfer_stats())
        row = {
            "kind": "prescribed_flux_thermal_state_traffic",
            "pipe_cells": cells,
            "device": device,
            "mode": mode,
            "wall_seconds": wall,
            "setup_and_warmup_seconds": setup,
            "hydraulic_records": len(records),
            "window_records": window_size,
            "thermal_steps": len(steps),
            "simulated_seconds": record_count * base["dt_s"],
            "physical_result_sha256": fingerprint,
            "transfer_stats": stats,
            "passed": fingerprint == reference,
            "max_combined_energy_residual_j": max(
                abs(a["combined_energy_residual_j"]) for a in audits
            ),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


def coupled_case(config_path, device, modes, save_runs):
    config = json.loads(config_path.read_text())
    rows = []
    reference = None
    for mode in modes:
        start = time.perf_counter()
        m = ThermalLiquidStation.from_dict(liquid_backend_config(config["system"], "torch", device))
        initial = m.initialize(**config["initial"])
        m.heat._torch_bridge = TorchHeatBridge(m.heat, m.heat.device)
        sync(device)
        setup = time.perf_counter() - start
        before = m.heat._torch_bridge.transfer_stats()
        periods = [{**p, "thermal_state_transfer": mode} for p in config["periods"]]
        sync(device)
        start = time.perf_counter()
        result = run_thermal_liquid_schedule(m, initial, periods)
        sync(device)
        wall = time.perf_counter() - start
        fingerprint = digest({"state": result["state"], "records": result["records"]})
        if reference is None:
            reference = fingerprint
        if save_runs:
            save_runs.mkdir(parents=True, exist_ok=True)
            (save_runs / (mode + ".json")).write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
        stats = difference(before, m.heat._torch_bridge.transfer_stats())
        row = {
            "kind": "actual_coupled_station_schedule",
            "device": device,
            "mode": mode,
            "wall_seconds": wall,
            "setup_seconds": setup,
            "physical_result_sha256": fingerprint,
            "passed": fingerprint == reference,
            "hydraulic_steps": sum(len(r["steps"]) for r in result["records"]),
            "thermal_steps": sum(len(r["thermal_steps"]) for r in result["records"]),
            "retries": sum(r["failed_steps"] for r in result["records"]),
            "transfer_stats": stats,
            "max_combined_energy_residual_j": max(
                abs(x["combined_energy_residual_j"])
                for r in result["records"]
                for x in r["thermal_audits"]
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cells", type=int, nargs="+", default=[4096, 32768])
    p.add_argument("--records", type=int, default=16)
    p.add_argument("--window-records", type=int, default=4)
    p.add_argument(
        "--modes", nargs="+", choices=("record", "window", "report"), default=["record", "report"]
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--save-runs", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.records < 1 or a.window_records < 1 or any(n < 1 for n in a.cells):
        p.error("Positive records, window-records and cells required")
    if len(set(a.modes)) != len(a.modes):
        p.error("Duplicate measurement modes")
    rows = (
        coupled_case(a.config, a.device, a.modes, a.save_runs)
        if a.config
        else [
            row
            for n in a.cells
            for row in transport_case(n, a.device, a.modes, a.records, a.window_records)
        ]
    )
    files = (*TORCH_SOURCES, "liquid_heat.py", "thermal_liquid.py", "thermal_liquid_schedule.py")
    result = {
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
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
