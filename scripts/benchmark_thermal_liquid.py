"""CPU thermal scaling and long coupled closed-recycle verification.

Scalability probes use manufactured conservative fluxes, not a large hydraulic
solve. The long run uses actual hydraulic/shaft/controller steps throughout.
"""

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
from check_thermal_liquid import pipe_probe
from scipy.linalg import expm
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surface import LiquidHeatNetwork, ThermalLiquidState, ThermalLiquidStation


def scaling(cells):
    start = time.monotonic()
    m, s, rec = pipe_probe(cells=cells, dt=0.01, duration=0.25, compact=True)
    data = m.to_dict()
    data["temporal_error_control"] = True
    m = LiquidHeatNetwork.from_dict(data)
    s = m.initialize(m.physics.initialize())
    r = m.advance_record(s, rec, max_substeps=1000000)
    elapsed = time.monotonic() - start
    expected = np.asarray([300 + 10 * gammainc(i + 1, cells * 0.25) for i in range(cells)])
    actual = np.asarray([r["state"]["temperatures_k"][f"line::{i}"] for i in range(cells)])
    error = float(max(abs(actual - expected)))
    return {
        "passed": error < 0.01 and abs(r["combined_energy_residual_j"]) < 1e-5,
        "kind": "manufactured_flux_caloric_scaling",
        "pipe_cells": cells,
        "fluid_unknowns": len(m.names),
        "simulated_seconds": 0.25,
        "wall_seconds": elapsed,
        "accepted_steps": len(r["steps"]),
        "linear_factorizations": r["linear_factorizations"],
        "max_temperature_error_k": error,
        "combined_energy_residual_j": r["combined_energy_residual_j"],
        "peak_process_resident_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def long_run(seconds, report_interval=5.0):
    config = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    # Deliberately separate a heating-only benchmark from the original trip/
    # cooling scenario, which remains independently checked without changes.
    config["system"]["temperature_trips"] = []
    config["system"]["thermostats"] = []
    config["system"]["heat"]["store_step_temperatures"] = False
    model = ThermalLiquidStation.from_dict(config["system"])
    state = model.initialize(**config["initial"])
    base = model.heat.inspect(state.heat)
    commands = {
        k: v
        for k, v in config["periods"][0].items()
        if k not in ("duration_s", "temperature_control_modes")
    }
    cp = 2000.0
    q = 0.0013
    dp = 250000.0
    omega = 100 * math.sqrt((dp / (1000 * 9.80665) + 1e6 * q * q) / 20)
    pump_heat = 2 * (dp * q * (1 / 0.8 - 1) + 10 * (omega / 100) ** 3)
    valve_heat = dp * 2 * q
    rate = 2 * 1000 * q * cp
    UA = 20.0
    Ch = 1000 * 0.01 * (1 + (650000 - 500000) / 5e7) * cp
    Ci = 5 * cp
    Cw = 1000.0
    matrix = np.array(
        [
            [-(rate + UA) / Ch, rate / Ch, UA / Ch, 0],
            [rate / Ci, -(rate + UA) / Ci, 0, UA / Ci],
            [UA / Cw, 0, -UA / Cw, 0],
            [0, UA / Cw, 0, -UA / Cw],
        ]
    )
    affine = np.zeros((5, 5))
    affine[:4, :4] = matrix
    affine[:4, 4] = [pump_heat / Ch, valve_heat / Ci, 0, 0]
    rows = []
    accepted = thermal = retry = thermal_retry = 0
    max_audit = 0.0
    start = time.monotonic()
    while state.heat.time_s < seconds:
        duration = min(report_interval, seconds - state.heat.time_s)
        result = model.advance(
            state, duration, max_substeps=1000000, max_thermal_substeps=1000000, **commands
        )
        state = ThermalLiquidState.from_dict(result["state"])
        props = model.heat.inspect(state.heat)
        now = state.heat.time_s
        exact = 300 + (expm(affine * now) @ np.array([0, 0, 0, 0, 1]))[:4]
        actual = np.array(
            [
                props["temperatures_k"]["h"],
                props["temperatures_k"]["in"],
                props["wall_temperatures_k"]["h"],
                props["wall_temperatures_k"]["in"],
            ]
        )
        expected_heat = (pump_heat + valve_heat) * now
        actual_heat = (
            props["fluid_energy_j"]
            + props["wall_energy_j"]
            - base["fluid_energy_j"]
            - base["wall_energy_j"]
        )
        accepted += len(result["steps"])
        thermal += len(result["thermal_steps"])
        retry += result["failed_steps"]
        thermal_retry += result["failed_thermal_steps"]
        max_audit = max(
            max_audit, max(abs(x["combined_energy_residual_j"]) for x in result["thermal_audits"])
        )
        row = {
            "time_s": now,
            "wall_seconds": time.monotonic() - start,
            "max_temperature_error_k": float(max(abs(actual - exact))),
            "analytic_energy_error_j": actual_heat - expected_heat,
            "temperatures_k": actual.tolist(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row["max_temperature_error_k"] > 1e-4 or abs(row["analytic_energy_error_j"]) > 0.01:
            raise RuntimeError("Long coupled run diverged from independent affine thermal ODE")
    return {
        "passed": max_audit < 1e-5,
        "kind": "actual_coupled_closed_recycle",
        "duration_s": seconds,
        "wall_seconds": time.monotonic() - start,
        "hydraulic_steps": accepted,
        "thermal_steps": thermal,
        "hydraulic_retries": retry,
        "thermal_retries": thermal_retry,
        "max_combined_energy_residual_j": max_audit,
        "max_temperature_error_k": max(r["max_temperature_error_k"] for r in rows),
        "max_analytic_energy_error_j": max(abs(r["analytic_energy_error_j"]) for r in rows),
        "expected_heat_rate_w": pump_heat + valve_heat,
        "reports": rows,
        "peak_process_resident_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "benchmark_system": config["system"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int)
    parser.add_argument("--long-run-seconds", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.cells is None) == (args.long_run_seconds is None):
        parser.error("Choose exactly one benchmark kind")
    if args.cells is not None and args.cells < 1:
        parser.error("cells must be positive")
    if args.long_run_seconds is not None and (
        not math.isfinite(args.long_run_seconds) or not 0 < args.long_run_seconds <= 900
    ):
        parser.error("long run must be in (0,900] s inside declared temperature range")
    result = scaling(args.cells) if args.cells is not None else long_run(args.long_run_seconds)
    paths = [
        ROOT / "src/surface" / n
        for n in (
            "liquid_heat.py",
            "thermal_liquid.py",
            "liquid_transient.py",
            "liquid_recycle.py",
            "controlled_liquid.py",
            "liquid_protection.py",
            "actuator.py",
            "pid.py",
            "liquid.py",
            "vessel_wall.py",
        )
    ]
    paths.extend(
        [
            Path(__file__).resolve(),
            ROOT / "scripts/check_thermal_liquid.py",
            ROOT / "configs/surface_thermal_liquid.json",
        ]
    )
    result["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("reports", "benchmark_system", "source_sha256")
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
