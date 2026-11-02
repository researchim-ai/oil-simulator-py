"""Grid/time refinement against a small-amplitude closed standing acoustic wave."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.thermal_pipe import ThermalHomogeneousPipe
from surface.transient_pipe import DriftFluxPipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thermal", action="store_true")
    args = parser.parse_args()
    records = []
    for n in (8, 16, 32):
        cls = ThermalHomogeneousPipe if args.thermal else DriftFluxPipe
        pipe = cls(10, 0.1, n, 850, 500, 300, 1, 0, 0)
        x = (np.arange(n) + 0.5) * 10 / n
        shape = np.cos(np.pi * x / 10) * np.sinc(0.5 / n)
        pressure = 1e6 + 100 * shape
        void = 0.98 if args.thermal else 0.2
        ml = (1 - void) * 850
        mg = void * 1e6 / (500 * 300)
        gamma = (
            1 + mg * 500 / (ml * pipe.liquid_cv_j_kg_k + mg * pipe.gas_cv_j_kg_k)
            if args.thermal
            else 1.0
        )
        temperature = 300 * (pressure / 1e6) ** ((gamma - 1) / gamma)
        ratio = mg / ml
        alpha = ratio * 850 / (pressure / (500 * temperature) + ratio * 850)
        initial = (
            pipe.initialize(pressure, alpha, temperature_k=temperature)
            if args.thermal
            else pipe.initialize(pressure, alpha)
        )
        started = time.perf_counter()
        result = pipe.advance(initial, 0.005, initial_dt_s=0.004 / n)
        rho = ml + mg
        sound = math.sqrt(gamma * 1e6 / (void * rho))
        expected = 1e6 + 100 * shape * math.cos(sound * math.pi / 10 * 0.005)
        error = float(np.mean(abs(np.array(result["properties"]["pressure_pa"]) - expected)))
        records.append(
            {
                "cells": n,
                "dt_s": 0.004 / n,
                "pressure_L1_error_pa": error,
                "wall_s": time.perf_counter() - started,
            }
        )
        if args.thermal:
            expected_t = 300 + 300 * (gamma - 1) / gamma * (expected - 1e6) / 1e6
            records[-1]["temperature_L1_error_k"] = float(
                np.mean(abs(np.array(result["properties"]["temperature_k"]) - expected_t))
            )
    if not all(
        b["pressure_L1_error_pa"] < 0.75 * a["pressure_L1_error_pa"]
        for a, b in zip(records, records[1:])
    ):
        raise RuntimeError("Pipe standing-wave refinement failed")
    if args.thermal and not all(
        b["temperature_L1_error_k"] < 0.75 * a["temperature_L1_error_k"]
        for a, b in zip(records, records[1:])
    ):
        raise RuntimeError("Pipe thermal-wave refinement failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "reference": "Linear standing acoustic wave, homogeneous thermal adiabatic limit; 100 Pa perturbation about 1 MPa; no slip, no friction"
                if args.thermal
                else "Linear standing acoustic wave, homogeneous isothermal Wood limit; 100 Pa perturbation about 1 MPa; no slip, no friction",
                "cases": records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
