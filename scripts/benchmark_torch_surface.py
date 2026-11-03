"""Reproducible float64 Torch liquid roots and resident caloric intervals.

Transport uses prescribed manufactured hydraulic fluxes, NOT a large coupled
hydraulic model or Norne. Timings include interval graph capture; repeated
hydraulic roots report capture separately. External SciPy oracles only verify
results and are never part of the measured device integrator.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_liquid_transient import coupled_model
from check_thermal_liquid import pipe_probe

from surface.torch_backend import TORCH_SOURCES, runtime_fingerprint
from surface.torch_heat import TorchHeatEquations, TorchHeatState
from surface.torch_heat_interval import TorchHeatInterval
from surface.torch_liquid import TorchLiquidEquations


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def hydraulic(device, repeats):
    model = coupled_model()
    e = TorchLiquidEquations(model, device)
    args = tuple(
        e.tensor(v)
        for v in (
            [550000, 500000],
            [0.002],
            [100, 110],
            0.001,
            [5, 7],
            [400000, 500000, 500000, 480000],
            [1],
        )
    )
    synchronize(device)
    start = time.perf_counter()
    out = e.solve(*args)
    synchronize(device)
    cold = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(repeats):
        out = e.solve(*args)
    synchronize(device)
    elapsed = time.perf_counter() - start
    ledger = e.mechanical_ledger(*args[:4], out[2], out[1])
    return {
        "kind": "coupled_pressure_pipe_pumps_shafts_root",
        "device": device,
        "unknowns": out[0].numel(),
        "cold_seconds": cold,
        "mean_repeat_seconds": elapsed / repeats,
        "repeats": repeats,
        "max_residual": float(out[1].abs().max()),
        "energy_residual_j": float(ledger["energy_residual_j"]),
        "passed": bool(out[3]) and bool(ledger["conservation_valid"]),
    }


def transport(device, cells):
    setup_start = time.perf_counter()
    steps = 64
    h = 0.25 / cells
    duration = steps * h
    model, state, rec = pipe_probe(cells=cells, dt=h, duration=duration, compact=True)
    e = TorchHeatEquations(model, device)
    temperatures, masses = dict(state.temperatures_k), dict(state.masses_kg)
    s = TorchHeatState(
        e.tensor([temperatures[n] for n in model.names]),
        e.tensor([masses[n] for n in model.names]),
    )
    flow = e.tensor([rec["midpoint_branch_volume_rates_m3_s"][n] for n in e.branch_names])
    heat = e.tensor([0.0] * len(e.branch_names))
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    start = time.perf_counter()
    result = TorchHeatInterval(e).advance(s, s.mass, flow, heat, duration, steps)
    synchronize(device)
    elapsed = time.perf_counter() - start
    host = result.state.temperature.cpu().numpy()
    actual = np.asarray([host[model._index[f"line::{i}"]] for i in range(cells)])
    expected = 300 + 10 * gammainc(np.arange(1, cells + 1), cells * duration)
    error = float(np.max(np.abs(actual - expected)))
    energy_error = float(result.audit[5])
    return {
        "kind": "manufactured_flux_resident_caloric_interval",
        "device": device,
        "pipe_cells": cells,
        "unknowns": e.size,
        "duration_s": duration,
        "setup_seconds_including_initial_upload": setup_seconds,
        "wall_seconds_including_capture": elapsed,
        "substeps": steps,
        "status_reads": result.status_reads,
        "graph_replays": result.graph_replays,
        "field_transfers_within_interval": 0,
        "max_temperature_error_vs_chain_ode_k": error,
        "energy_residual_j": energy_error,
        "max_local_energy_residual_j": float(result.audit[6]),
        "dense_matrix_allocated": e.identity is not None,
        "passed": error < 0.01 and abs(energy_error) < 1e-5,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cells", type=int, nargs="+", default=[16, 512, 4096, 32768])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.repeats < 1 or any(n < 1 for n in a.cells):
        p.error("Positive cells and repeats required")
    rows = [hydraulic(a.device, a.repeats)]
    print(json.dumps(rows[0]), flush=True)
    for cells in a.cells:
        row = transport(a.device, cells)
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": {
            name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
            for name in ("check_thermal_liquid.py", "check_liquid_transient.py")
        },
        "runtime": runtime_fingerprint(),
        "device": a.device,
        "gpu": torch.cuda.get_device_name() if a.device.startswith("cuda") else None,
        "source": {
            name: hashlib.sha256((ROOT / "src/surface" / name).read_bytes()).hexdigest()
            for name in TORCH_SOURCES
        },
        "passed": all(row["passed"] for row in rows),
        "results": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
