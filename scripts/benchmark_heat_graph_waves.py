"""Actual coupled acoustic/heat waves with eager/graph caloric kernels.

The pressure error uses an independent linear acoustic mode. It is not an
external equipment calibration or a full field/surface performance estimate.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from benchmark_liquid_state_transfer import delta, digest, sync
from check_liquid_transient import wave_model, wave_reference

from surface import LiquidHeatNetwork, LiquidTransientNetwork, LiquidTransientState
from surface.torch_backend import TORCH_SOURCES, runtime_fingerprint
from surface.torch_heat_session import TorchHeatWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cells", type=int, nargs="+", default=[32, 80, 256])
    parser.add_argument("--intervals", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-runs", type=Path)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("Eager/graph comparison requires --device cuda:N")
    if args.intervals < 1 or any(n < 1 for n in args.cells):
        parser.error("Positive sizes required")
    sources = (*TORCH_SOURCES, "liquid_transient.py", "liquid_heat.py")
    hashes = {
        n: hashlib.sha256((ROOT / "src/surface" / n).read_bytes()).hexdigest() for n in sources
    }
    rows = []
    for cells in args.cells:
        expected = None
        for mode in ("eager", "graph"):
            physics = LiquidTransientNetwork.from_dict(
                {
                    **wave_model(cells).to_dict(),
                    "device": args.device,
                    "temporal_error_control": True,
                }
            )
            heat = LiquidHeatNetwork(
                physics,
                4200,
                {"left": 310, "right": 300},
                reference_temperature_k=300,
                maximum_dt_s=physics.maximum_dt_s,
                temporal_error_control=False,
                store_step_temperatures=False,
            )
            initial = physics.initialize(dict(zip(physics.free, wave_reference(cells, 0))))
            thermal = heat.initialize(initial)
            ceiling = min(
                physics.maximum_dt_s, physics.acoustic_courant * 5 / cells / (1e8 / 1000) ** 0.5
            )
            duration = 4 * ceiling
            # Equal warmup of the actual coupled kernels, outside counters/time.
            warm = heat.open_session(thermal, execution=mode)
            window = TorchHeatWindow(warm, None, None, {}, 100000)
            physics.advance(initial, ceiling, _on_accepted_step=window.push)
            window.finish()
            warm.close()
            bridges = {"mechanical": physics._torch_bridge, "thermal": heat._torch_bridge}
            before = {n: b.transfer_stats() for n, b in bridges.items()}
            sync(args.device)
            start = time.perf_counter()
            session = heat.open_session(thermal, execution=mode)
            state = initial
            records, thermal_steps, audits = [], [], []
            for _ in range(args.intervals):
                window = TorchHeatWindow(session, None, None, {}, 100000)
                result = physics.advance(state, duration, _on_accepted_step=window.push)
                calc = window.finish()
                records.append(result)
                thermal_steps.extend(calc["steps"])
                audits.extend(calc["audits"])
                state = LiquidTransientState(**result["state"])
            final_heat = session.close().to_dict()
            sync(args.device)
            elapsed = time.perf_counter() - start
            key = digest(
                {"physical": records, "heat": final_heat, "steps": thermal_steps, "audits": audits}
            )
            if args.save_runs:
                args.save_runs.mkdir(parents=True, exist_ok=True)
                (args.save_runs / f"{cells}_{mode}.json").write_text(
                    json.dumps(
                        {
                            "physical": records,
                            "heat": final_heat,
                            "steps": thermal_steps,
                            "audits": audits,
                        },
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                )
            if expected is None:
                expected = key
            pressures = dict(state.pressures_pa)
            error = max(
                abs(pressures[n] - ref)
                for n, ref in zip(physics.free, wave_reference(cells, state.time_s))
            )
            row = {
                "cells": cells,
                "hydraulic_unknowns": 2 * cells + 1,
                "thermal_cells": len(heat.names),
                "mode": mode,
                "wall_seconds": elapsed,
                "accepted_steps": sum(len(r["steps"]) for r in records),
                "physical_sha256": key,
                "passed": key == expected,
                "final_time_s": state.time_s,
                "max_pressure_error_pa": float(error),
                "max_combined_energy_residual_j": max(
                    abs(a["combined_energy_residual_j"]) for a in audits
                ),
                "transfer_stats": {
                    n: delta(before[n], b.transfer_stats()) for n, b in bridges.items()
                },
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    if hashes != {
        n: hashlib.sha256((ROOT / "src/surface" / n).read_bytes()).hexdigest() for n in sources
    }:
        raise RuntimeError("Numerical source changed during benchmark")
    output = {
        "passed": all(r["passed"] for r in rows),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "runtime": runtime_fingerprint(),
        "source": hashes,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if not output["passed"]:
        raise SystemExit("Physical journal mismatch")


if __name__ == "__main__":
    main()
