"""Measure CPU distributed-pipe solves; no claim about coupled-field GPU speed."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.thermal_pipe import ThermalHomogeneousPipe
from surface.transient_pipe import DriftFluxPipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", type=int, default=[4, 16, 64])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thermal", action="store_true")
    parser.add_argument("--thermal-flux", choices=("entropy_stable", "rusanov"))
    args = parser.parse_args()
    if args.thermal_flux is not None and not args.thermal:
        parser.error("--thermal-flux requires --thermal")
    records = []
    for cells in args.cells:
        cls = ThermalHomogeneousPipe if args.thermal else DriftFluxPipe
        pipe = cls(
            10,
            0.1,
            cells,
            850,
            500,
            300,
            1 if args.thermal else 1.1,
            0 if args.thermal else 0.1,
            0.02,
            **({"numerical_flux": args.thermal_flux} if args.thermal_flux is not None else {}),
        )
        initial = pipe.initialize(1e6, 0.2)
        started = time.perf_counter()
        result = pipe.advance(
            initial,
            0.02,
            left={
                "pressure_pa": 1010000,
                "gas_void_fraction": 0.2,
                "mixture_velocity_m_s": 0,
                **({"temperature_k": 300} if args.thermal else {}),
            },
            right={
                "pressure_pa": 1000000,
                "gas_void_fraction": 0.2,
                "mixture_velocity_m_s": 0,
                **({"temperature_k": 300} if args.thermal else {}),
            },
            initial_dt_s=0.01,
        )
        wall = time.perf_counter() - started
        records.append(
            {
                "cells": cells,
                "unknowns": (4 if args.thermal else 3) * cells,
                "simulated_s": 0.02,
                "wall_s": wall,
                "accepted_steps": len(result["steps"]),
                "failed_steps": result["failed_steps"],
                "max_mass_residual_kg": max(
                    max(abs(x) for x in step["mass_residuals_kg"]) for step in result["steps"]
                ),
                "max_momentum_residual_kg_m_s": max(
                    abs(step["momentum_residual_kg_m_s"]) for step in result["steps"]
                ),
            }
        )
        if args.thermal:
            records[-1]["max_energy_residual_j"] = max(
                abs(s["energy_residual_j"]) for s in result["steps"]
            )
    source = Path(__file__).resolve().parents[1] / (
        "src/surface/thermal_pipe.py" if args.thermal else "src/surface/transient_pipe.py"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "CPU reference, distributed two-phase pipe only; initialization excluded; single timing per mesh, not a speedup study",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "model": "thermal_homogeneous" if args.thermal else "isothermal_drift_flux",
                "numerical_flux": pipe.numerical_flux if args.thermal else "rusanov",
                "thermal_flux_sha256": hashlib.sha256(
                    source.with_name("thermal_flux.py").read_bytes()
                ).hexdigest()
                if args.thermal
                else None,
                "cases": records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
