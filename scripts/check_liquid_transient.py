"""Independent damped acoustic modes for the weakly compressible liquid model."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    Liquid,
    LiquidNetwork,
    LiquidTransientNetwork,
    Node,
    Pipe,
    Pump,
    RotatingDrive,
    ThrottleValve,
)


def oscillator(dt=0.0005):
    network = LiquidNetwork(
        Liquid(1000, 0.001, 2000), [Node("a"), Node("b")], [Pipe("line", "a", "b", 1, 0.05)]
    )
    return LiquidTransientNetwork(
        network,
        {"a": 0.01, "b": 0.01},
        1e8,
        reference_pressure_pa=600000,
        maximum_dt_s=dt,
        temporal_error_control=False,
    )


def oscillator_reference(time_s):
    area = math.pi * 0.05**2 / 4
    inertance = 1000 / area
    compliance = 0.01 / 1e8
    resistance = 128 * 0.001 / (math.pi * 0.05**4)
    damping = resistance / (2 * inertance)
    frequency = math.sqrt(2 / (inertance * compliance) - damping**2)
    d = (
        2000
        * math.exp(-damping * time_s)
        * (math.cos(frequency * time_s) + damping / frequency * math.sin(frequency * time_s))
    )
    q = 2000 / (inertance * frequency) * math.exp(-damping * time_s) * math.sin(frequency * time_s)
    return np.asarray([600000 + d / 2, 600000 - d / 2]), q


def wave_model(cells, dt=0.00005):
    length, diameter = 10.0, 0.1
    dx = length / cells
    names = ["left"] + [f"c{i}" for i in range(cells)] + ["right"]
    nodes = [Node(n, pressure_pa=600000 if n in ("left", "right") else None) for n in names]
    network = LiquidNetwork(
        Liquid(1000, 0.001, 2000),
        nodes,
        [
            Pipe(f"p{i}", a, b, dx / 2 if i in (0, cells) else dx, diameter)
            for i, (a, b) in enumerate(zip(names[:-1], names[1:]))
        ],
    )
    return LiquidTransientNetwork(
        network,
        {f"c{i}": math.pi * diameter**2 / 4 * dx for i in range(cells)},
        1e8,
        reference_pressure_pa=600000,
        maximum_dt_s=dt,
        temporal_error_control=False,
    )


def wave_reference(cells, time_s):
    # Constant-density linear acoustic mode with laminar Poiseuille friction.
    # This continuous formula does not use the code's matrix or pipe method.
    c, k = math.sqrt(1e8 / 1000), math.pi / 10
    damping = 16 * 0.001 / (1000 * 0.1**2)
    frequency = math.sqrt(c * c * k * k - damping * damping)
    amplitude = (
        1000
        * math.exp(-damping * time_s)
        * (math.cos(frequency * time_s) + damping / frequency * math.sin(frequency * time_s))
    )
    x = (np.arange(cells) + 0.5) * 10 / cells
    return 600000 + amplitude * np.sin(k * x)


def coupled_model(dt=0.002):
    network = LiquidNetwork(
        Liquid(1000, 0.1, 2000),
        [Node("in", pressure_pa=400000), Node("h"), Node("t"), Node("out", pressure_pa=480000)],
        [Pipe("line", "h", "t", 10, 0.05)],
        [
            Pump("a", "in", "h", 20, 1e6, 0.004, efficiency=0.8, check_valve=True),
            Pump("b", "in", "h", 22, 1.2e6, 0.004, efficiency=0.75, check_valve=True),
        ],
        [ThrottleValve("export", "t", "out", 0.02, discharge_coefficient=0.7)],
    )
    return LiquidTransientNetwork(
        network,
        {"h": 0.1, "t": 0.1},
        1e7,
        reference_pressure_pa=500000,
        drives={
            n: RotatingDrive(
                j,
                150,
                20,
                10,
                viscous_drag_nm_s=b,
                motor_efficiency=0.9,
                regeneration_efficiency=0.7,
            )
            for n, j, b in zip(("a", "b"), (0.5, 0.7), (0.01, 0.015))
        },
        nominal_speeds_rad_s={"a": 100, "b": 110},
        no_flow_power_w_at_nominal={"a": 5, "b": 8},
        maximum_dt_s=dt,
        temporal_error_control=False,
    )


def coupled_rhs(t, y):
    ph, pt, q, wa, wb = y
    w = np.asarray([wa, wb])
    qp = np.sqrt(
        np.maximum(
            0.0, 1000 * 9.80665 * np.asarray([20, 22]) * (w / [100, 110]) ** 2 - (ph - 400000)
        )
        / (1000 * 9.80665 * np.asarray([1e6, 1.2e6]))
    )
    load = (ph - 400000) * qp / np.asarray([0.8, 0.75]) / w + np.asarray([5, 8]) * (
        w / [100, 110]
    ) ** 3 / w
    valve = (
        0.7
        * math.pi
        * 0.02**2
        / 4
        * math.copysign(math.sqrt(2 * abs(pt - 480000) / 1000), pt - 480000)
    )
    resistance = 128 * 0.1 * 10 / (math.pi * 0.05**4)
    inertance = 1000 * 10 / (math.pi * 0.05**2 / 4)
    return np.r_[
        1e8 * (sum(qp) - q),
        1e8 * (q - valve),
        (ph - pt - resistance * q) / inertance,
        (np.asarray([5, 7]) - load - np.asarray([0.01, 0.015]) * w) / [0.5, 0.7],
    ]


def check_reference():
    expected, expected_q = oscillator_reference(0.02)
    temporal = []
    for dt in (0.0005, 0.00025, 0.000125):
        model = oscillator(dt)
        result = model.advance(model.initialize({"a": 601000, "b": 599000}), 0.02)
        error = max(
            abs(np.asarray([result["state"]["pressures_pa"][n] for n in ("a", "b")]) - expected)
        )
        temporal.append(
            {
                "dt_s": dt,
                "pressure_error_pa": float(error),
                "flow_error_m3_s": abs(
                    result["state"]["pipe_volume_rates_m3_s"]["line"] - expected_q
                ),
                "max_energy_residual_j": max(abs(s["energy_residual_j"]) for s in result["steps"]),
                "max_mass_residual_kg": max(
                    s["max_node_mass_residual_kg"] for s in result["steps"]
                ),
            }
        )
    ratios = [
        temporal[i]["pressure_error_pa"] / temporal[i + 1]["pressure_error_pa"] for i in range(2)
    ]
    spatial = []
    for cells in (8, 16, 32):
        model = wave_model(cells)
        initial_p = wave_reference(cells, 0)
        initial = model.initialize(dict(zip(model.free, initial_p)))
        result = model.advance(initial, 0.01)
        p = np.asarray([result["state"]["pressures_pa"][n] for n in model.free])
        error = max(abs(p - wave_reference(cells, 0.01)))
        spatial.append(
            {
                "cells": cells,
                "pressure_error_pa": float(error),
                "max_energy_residual_j": max(abs(s["energy_residual_j"]) for s in result["steps"]),
            }
        )
    space_ratios = [
        spatial[i]["pressure_error_pa"] / spatial[i + 1]["pressure_error_pa"] for i in range(2)
    ]
    initial_y = np.asarray([550000.0, 500000.0, 0.002, 100.0, 110.0])
    reference = solve_ivp(coupled_rhs, (0, 0.2), initial_y, method="DOP853", rtol=1e-12, atol=1e-12)
    coupled = []
    for dt in (0.004, 0.002, 0.001):
        model = coupled_model(dt)
        initial = model.initialize(
            {"h": 550000, "t": 500000}, {"line": 0.002}, {"a": 100, "b": 110}
        )
        result = model.advance(initial, 0.2, motor_torques_nm={"a": 5, "b": 7})
        state = result["state"]
        y = np.asarray(
            [
                state["pressures_pa"]["h"],
                state["pressures_pa"]["t"],
                state["pipe_volume_rates_m3_s"]["line"],
                state["speeds_rad_s"]["a"],
                state["speeds_rad_s"]["b"],
            ]
        )
        coupled.append(
            {
                "dt_s": dt,
                "pressure_error_pa": float(max(abs(y[:2] - reference.y[:2, -1]))),
                "speed_error_rad_s": float(max(abs(y[3:] - reference.y[3:, -1]))),
                "max_energy_residual_j": max(abs(s["energy_residual_j"]) for s in result["steps"]),
            }
        )
    coupled_ratios = [
        coupled[i]["pressure_error_pa"] / coupled[i + 1]["pressure_error_pa"] for i in range(2)
    ]
    passed = (
        min(ratios) > 3.8
        and min(space_ratios) > 3.5
        and temporal[-1]["pressure_error_pa"] < 0.2
        and spatial[-1]["pressure_error_pa"] < 0.5
        and all(
            r["max_mass_residual_kg"] < 1e-10 and r["max_energy_residual_j"] < 1e-8
            for r in temporal
        )
        and all(r["max_energy_residual_j"] < 1e-8 for r in spatial)
        and reference.success
        and min(coupled_ratios) > 3.5
        and coupled[-1]["pressure_error_pa"] < 0.5
        and coupled[-1]["speed_error_rad_s"] < 1e-5
    )
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__)] + [
        root / "src/surface" / name
        for name in (
            "liquid_transient.py",
            "liquid_relief.py",
            "liquid.py",
            "dynamic_station.py",
            "rotating_drive.py",
            "curves.py",
        )
    ]
    return {
        "passed": bool(passed),
        "temporal": temporal,
        "temporal_ratios": ratios,
        "spatial": spatial,
        "spatial_ratios": space_ratios,
        "coupled": coupled,
        "coupled_pressure_ratios": coupled_ratios,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_reference()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit("Liquid transient reference failed")


if __name__ == "__main__":
    main()
