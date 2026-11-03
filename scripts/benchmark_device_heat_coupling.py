"""Paired complete station schedules with both adapter traffic counters.

Compares all accepted journals and final states, excluding source fingerprints
and execution-mode period fields. Logical counters are not PCIe profiling.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from benchmark_liquid_state_transfer import delta, digest, sync

from surface import ThermalLiquidStation, run_thermal_liquid_schedule
from surface.torch_backend import TORCH_SOURCES, liquid_backend_config, runtime_fingerprint
from surface.torch_heat_bridge import TorchHeatBridge
from surface.torch_liquid_bridge import TorchLiquidBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surface_thermal_liquid.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-runs", type=Path)
    parser.add_argument(
        "--modes", nargs="+", choices=("host", "device"), default=["host", "device"]
    )
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("Duplicate modes")
    config = json.loads(args.config.read_text())
    source_files = (
        *TORCH_SOURCES,
        "liquid_transient.py",
        "controlled_liquid.py",
        "liquid_protection.py",
        "thermal_liquid.py",
        "thermal_liquid_schedule.py",
    )
    source = {
        n: hashlib.sha256((ROOT / "src/surface" / n).read_bytes()).hexdigest() for n in source_files
    }
    rows = []
    reference = None
    for mode in args.modes:
        model = ThermalLiquidStation.from_dict(
            liquid_backend_config(config["system"], "torch", args.device)
        )
        initial = model.initialize(**config["initial"])
        physics = model.recycling.physics
        physics._torch_bridge = TorchLiquidBridge(physics, args.device)
        model.heat._torch_bridge = TorchHeatBridge(model.heat, args.device)
        # Same warmup in both modes, outside the measured schedule.
        commands = {k: v for k, v in config["periods"][0].items() if k != "duration_s"}
        model.advance(initial, 0.001, hydraulic_heat_transfer=mode, **commands)
        bridges = {"mechanical": physics._torch_bridge, "thermal": model.heat._torch_bridge}
        before = {n: b.transfer_stats() for n, b in bridges.items()}
        periods = [{**p, "hydraulic_heat_transfer": mode} for p in config["periods"]]
        sync(args.device)
        start = time.perf_counter()
        result = run_thermal_liquid_schedule(model, initial, periods)
        sync(args.device)
        wall = time.perf_counter() - start
        key = digest({"state": result["state"], "records": result["records"]})
        if reference is None:
            reference = key
        if args.save_runs:
            args.save_runs.mkdir(parents=True, exist_ok=True)
            (args.save_runs / (mode + ".json")).write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
        row = {
            "mode": mode,
            "device": args.device,
            "wall_seconds": wall,
            "physical_sha256": key,
            "passed": key == reference,
            "accepted_steps": sum(len(r["steps"]) for r in result["records"]),
            "retries": sum(r["failed_steps"] for r in result["records"]),
            "max_combined_energy_residual_j": max(
                abs(a["combined_energy_residual_j"])
                for r in result["records"]
                for a in r["thermal_audits"]
            ),
            "transfer_stats": {n: delta(before[n], b.transfer_stats()) for n, b in bridges.items()},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    if source != {
        n: hashlib.sha256((ROOT / "src/surface" / n).read_bytes()).hexdigest() for n in source_files
    }:
        raise RuntimeError("Numerical source changed during benchmark")
    output = {
        "passed": all(r["passed"] for r in rows),
        "runtime": runtime_fingerprint(),
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": source,
        "thread_environment": {
            n: os.environ.get(n) for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if not output["passed"]:
        raise SystemExit("Physical journal mismatch")


if __name__ == "__main__":
    main()
