"""Independent Rankine-Hugoniot gate for symmetric thermal-pipe compression.

The oracle solves jump conservation, without calling the simulator EOS/flux.
Both phases have fixed mass composition and a common velocity/temperature.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import ThermalHomogeneousPipe


def symmetric_shock_reference(
    pressure_pa=1e6,
    temperature_k=300,
    void_fraction=0.98,
    inward_velocity_m_s=50,
    liquid_density_kg_m3=850,
    gas_constant_j_kg_k=500,
    liquid_cv_j_kg_k=2000,
    gas_cv_j_kg_k=1500,
):
    values = (
        pressure_pa,
        temperature_k,
        inward_velocity_m_s,
        liquid_density_kg_m3,
        gas_constant_j_kg_k,
        liquid_cv_j_kg_k,
        gas_cv_j_kg_k,
    )
    if not all(math.isfinite(x) and x > 0 for x in values) or not 0 < void_fraction < 1:
        raise ValueError("Shock oracle requires two phases and positive physical parameters")
    ml = (1 - void_fraction) * liquid_density_kg_m3
    mg = pressure_pa * void_fraction / (gas_constant_j_kg_k * temperature_k)
    rho = ml + mg
    zl, zg = ml / rho, mg / rho
    cv = zl * liquid_cv_j_kg_k + zg * gas_cv_j_kg_k
    rg = zg * gas_constant_j_kg_k
    covolume = zl / liquid_density_kg_m3
    volume = 1 / rho
    velocity = inward_velocity_m_s
    gamma = 1 + rg / cv
    sound = math.sqrt(gamma * pressure_pa / (rho * void_fraction))

    def states(speed):
        star_volume = volume * speed / (speed + velocity)
        star_pressure = pressure_pa + rho * velocity * (speed + velocity)
        star_temperature = star_pressure * (star_volume - covolume) / rg
        return star_volume, star_pressure, star_temperature

    initial_h = cv * temperature_k + pressure_pa * volume

    def energy_jump(speed):
        v, p, t = states(speed)
        return cv * t + p * v + 0.5 * speed**2 - initial_h - 0.5 * (speed + velocity) ** 2

    minimum = velocity * covolume / (volume - covolume) * (1 + 1e-8)
    speed = brentq(energy_jump, minimum, 10 * sound + velocity, xtol=1e-11)
    v, p, t = states(speed)
    mass_flux = rho * (speed + velocity)
    return {
        "pressure_pa": p,
        "temperature_k": t,
        "density_kg_m3": 1 / v,
        "gas_void_fraction": 1 - zl / (v * liquid_density_kg_m3),
        "velocity_m_s": 0.0,
        "shock_speed_m_s": speed,
        "energy_jump_j_kg": energy_jump(speed),
        "mass_jump_kg_m2_s": speed / v - mass_flux,
        "momentum_jump_pa": p + speed**2 / v - pressure_pa - rho * (speed + velocity) ** 2,
        "initial_density_kg_m3": rho,
        "mixture_cv_j_kg_k": cv,
        "initial_specific_enthalpy_j_kg": initial_h,
    }


def run_case(cells, time_s=0.005, void_fraction=0.98, inward_velocity_m_s=50.0):
    pipe = ThermalHomogeneousPipe(10, 0.1, cells, 850, 500, 300, 1, 0, 0)
    x = (np.arange(cells) + 0.5) * 10 / cells - 5
    initial = pipe.initialize(
        1e6, void_fraction, np.where(x < 0, inward_velocity_m_s, -inward_velocity_m_s)
    )
    state = {"pressure_pa": 1e6, "gas_void_fraction": void_fraction, "temperature_k": 300}
    result = pipe.advance(
        initial,
        time_s,
        left={**state, "mixture_velocity_m_s": inward_velocity_m_s},
        right={**state, "mixture_velocity_m_s": -inward_velocity_m_s},
        initial_dt_s=0.004 / cells,
    )
    oracle = symmetric_shock_reference(
        void_fraction=void_fraction, inward_velocity_m_s=inward_velocity_m_s
    )
    center = slice(cells // 2 - 1, cells // 2 + 1)
    pressure = float(np.mean(result["properties"]["pressure_pa"][center]))
    temperature = float(np.mean(result["properties"]["temperature_k"][center]))
    # Compare finite-volume averages of CONSERVATIVE exact states, not
    # point samples at moving discontinuities. Keep the stagnation-cell
    # temperature error separately; L1 convergence does not certify it.
    dx = 10 / cells
    distance = oracle["shock_speed_m_s"] * time_s
    fraction = (
        np.maximum(0.0, np.minimum(x + dx / 2, distance) - np.maximum(x - dx / 2, -distance)) / dx
    )
    rho0 = oracle["initial_density_kg_m3"]
    rhos = oracle["density_kg_m3"]
    cv = oracle["mixture_cv_j_kg_k"]
    zl = (1 - void_fraction) * 850 / rho0
    zg = 1 - zl
    rho = fraction * rhos + (1 - fraction) * rho0
    momentum = (1 - fraction) * rho0 * np.where(x < 0, inward_velocity_m_s, -inward_velocity_m_s)
    energy = fraction * rhos * cv * oracle["temperature_k"] + (1 - fraction) * (
        rho0 * cv * 300 + 0.5 * rho0 * inward_velocity_m_s**2
    )
    reference_t = (energy - 0.5 * momentum**2 / rho) / (rho * cv)
    reference_p = rho * zg * 500 * reference_t / (1 - rho * zl / 850)
    return {
        "cells": cells,
        "void_fraction": void_fraction,
        "inward_velocity_m_s": inward_velocity_m_s,
        "numerical_flux": pipe.numerical_flux,
        "thermal_entropy_dissipation": pipe.thermal_entropy_dissipation,
        "adaptive_entropy_dissipation": pipe.adaptive_entropy_dissipation,
        "dt_s": 0.004 / cells,
        "time_s": time_s,
        "center_pressure_pa": pressure,
        "center_temperature_k": temperature,
        "pressure_error_pa": abs(pressure - oracle["pressure_pa"]),
        "temperature_error_k": abs(temperature - oracle["temperature_k"]),
        "pressure_L1_error_pa": float(
            np.mean(abs(np.array(result["properties"]["pressure_pa"]) - reference_p))
        ),
        "temperature_L1_error_k": float(
            np.mean(abs(np.array(result["properties"]["temperature_k"]) - reference_t))
        ),
        "temperature_min_k": float(np.min(result["properties"]["temperature_k"])),
        "temperature_max_k": float(np.max(result["properties"]["temperature_k"])),
        "accepted_steps": len(result["steps"]),
        "failed_steps": result["failed_steps"],
        "max_energy_residual_j": max(abs(s["energy_residual_j"]) for s in result["steps"]),
        "minimum_entropy_production_j_k": min(s["entropy_production_j_k"] for s in result["steps"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--require-local-temperature", action="store_true")
    args = parser.parse_args()
    oracle = symmetric_shock_reference()
    if (
        len(args.cells) < 2
        or any(n % 2 or n < 2 or n > 1000 for n in args.cells)
        or any(b <= a for a, b in zip(args.cells, args.cells[1:]))
    ):
        parser.error("Use increasing even cell counts in [2,1000]")
    rows = [run_case(n) for n in args.cells]
    l1_passed = (
        all(
            b["pressure_L1_error_pa"] < a["pressure_L1_error_pa"]
            and b["temperature_L1_error_k"] < a["temperature_L1_error_k"]
            for a, b in zip(rows, rows[1:])
        )
        and rows[-1]["pressure_L1_error_pa"] < 0.005 * oracle["pressure_pa"]
        and rows[-1]["temperature_L1_error_k"] < 0.1
    )
    local_passed = rows[-1]["temperature_error_k"] < 0.02
    passed = l1_passed and (not args.require_local_temperature or local_passed)
    result = {
        "scope": "Independent jump conservation for frozen homogeneous horizontal two-phase caloric model; not an experimental multiphase correlation validation",
        "oracle": oracle,
        "cases": rows,
        "passed": passed,
        "l1_passed": l1_passed,
        "local_temperature_required": args.require_local_temperature,
        "stagnation_temperature_gate_passed": local_passed,
        "scope_of_passed": "Selected gate: finite-volume L1 refinement, plus local stagnation temperature if explicitly required. A passing L1 gate does not certify local temperature accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not passed:
        parser.exit(1, "Thermal pipe shock refinement failed\n")
    print("Thermal pipe shock refinement passed")


if __name__ == "__main__":
    main()
