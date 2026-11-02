"""Independent parallel-pump Poiseuille/ODE reference; synthetic liquid only."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import DynamicPumpStation, Liquid, LiquidNetwork, Node, Pipe, Pump, RotatingDrive


def reference_model(dt=0.1):
    names = ("a", "b")
    network = LiquidNetwork(
        Liquid(1000, 0.1, 2000),
        [Node("in", pressure_pa=200000), Node("header"), Node("out", pressure_pa=250000)],
        [Pipe("line", "header", "out", 100, 0.05)],
        [
            Pump("a", "in", "header", 20, 1e6, 0.004, efficiency=0.8, check_valve=True),
            Pump("b", "in", "header", 24, 1.3e6, 0.004, efficiency=0.75, check_valve=True),
        ],
    )
    drives = {
        n: RotatingDrive(
            j,
            150,
            20,
            10,
            viscous_drag_nm_s=drag,
            motor_efficiency=0.9,
            regeneration_efficiency=0.7,
        )
        for n, j, drag in zip(names, (1.0, 1.4), (0.02, 0.015))
    }
    return DynamicPumpStation(
        network,
        drives,
        dict(zip(names, (100.0, 110.0))),
        pressure_node="header",
        maximum_dt_s=dt,
        control_interval_s=0.2,
        max_speed_change_rad_s=10,
        no_flow_power_w_at_nominal={"a": 5, "b": 8},
    )


def independent_hydraulics(speeds):
    # No surface/network/pump solver calls: common header from an independently
    # written scalar mass balance and Hagen-Poiseuille pipe resistance.
    resistance = 128 * 0.1 * 100 / (math.pi * 0.05**4)
    heads = 1000 * 9.80665 * np.asarray((20.0, 24.0)) * (speeds / (100.0, 110.0)) ** 2
    curves = 1000 * 9.80665 * np.asarray((1e6, 1.3e6))

    def flows(rise):
        return np.sqrt(np.maximum(0, heads - rise) / curves)

    if max(heads) <= 50000:
        rise = 50000.0
    else:
        rise = brentq(
            lambda p: sum(flows(p)) - (p - 50000) / resistance, 50000.0, max(heads), xtol=1e-8
        )
    q = flows(rise)
    load = np.divide(rise * q / (0.8, 0.75), speeds, out=np.zeros(2), where=speeds > 0)
    load += np.asarray((5.0, 8.0)) * (speeds / (100.0, 110.0)) ** 3 / speeds
    return 200000 + rise, q, load


def check_reference():
    initial = np.asarray((90.0, 96.0))
    torque = np.asarray((6.0, 8.0))

    def rhs(t, omega):
        return (torque - independent_hydraulics(omega)[2] - (0.02, 0.015) * omega) / (1.0, 1.4)

    solution = solve_ivp(rhs, (0, 4), initial, method="DOP853", rtol=1e-12, atol=1e-12)
    if not solution.success:
        raise RuntimeError("Independent reference integration failed")
    rows = []
    for dt in (0.2, 0.1, 0.05):
        model = reference_model(dt)
        result = model.advance(
            model.initialize(dict(zip(model.names, initial))),
            4,
            mode="torque",
            motor_torques_nm=dict(zip(model.names, torque)),
        )
        speeds = np.asarray([result["state"]["speeds_rad_s"][n] for n in model.names])
        p, q, _ = independent_hydraulics(speeds)
        rows.append(
            {
                "dt_s": dt,
                "speed_error_rad_s": float(max(abs(speeds - solution.y[:, -1]))),
                "pressure_error_pa": abs(result["endpoint"]["pressures_pa"]["header"] - p),
                "flow_error_m3_s": max(
                    abs(result["endpoint"]["pump_volume_rates_m3_s"][n] - q[i])
                    for i, n in enumerate(model.names)
                ),
                "max_energy_residual_j": max(abs(s["energy_residual_j"]) for s in result["steps"]),
                "max_node_mass_residual_kg_s": max(
                    abs(v)
                    for s in result["steps"]
                    for v in s["midpoint"]["node_mass_residuals_kg_s"].values()
                ),
                "accepted_steps": len(result["steps"]),
                "retries": result["failed_steps"],
            }
        )
    ratios = [rows[i]["speed_error_rad_s"] / rows[i + 1]["speed_error_rad_s"] for i in range(2)]
    passed = (
        all(r > 3.5 for r in ratios)
        and rows[-1]["speed_error_rad_s"] < 1e-4
        and all(
            r["pressure_error_pa"] < 1e-3
            and r["flow_error_m3_s"] < 1e-9
            and r["max_energy_residual_j"] < 1e-4
            and r["max_node_mass_residual_kg_s"] < 1e-8
            for r in rows
        )
    )
    # A separate nonsmooth case crosses a pump check-valve opening. Do not
    # claim a uniform refinement ratio across this switching surface.
    switch_initial = np.asarray((90.0, 105.0))
    switch_reference = solve_ivp(
        rhs, (0, 4), switch_initial, method="DOP853", rtol=1e-12, atol=1e-12
    )
    switch_model = reference_model(0.025)
    switched = switch_model.advance(
        switch_model.initialize(dict(zip(switch_model.names, switch_initial))),
        4,
        mode="torque",
        motor_torques_nm=dict(zip(switch_model.names, torque)),
    )
    switch_error = max(
        abs(switched["state"]["speeds_rad_s"][n] - switch_reference.y[i, -1])
        for i, n in enumerate(switch_model.names)
    )
    passed = passed and switch_reference.success and switch_error < 2e-4
    root = Path(__file__).resolve().parents[1]
    return {
        "passed": bool(passed),
        "case": "two heterogeneous coupled shafts, laminar liquid line",
        "reference_final_speeds_rad_s": solution.y[:, -1].tolist(),
        "refinements": rows,
        "refinement_ratios": ratios,
        "check_valve_crossing": {
            "dt_s": 0.025,
            "speed_error_rad_s": float(switch_error),
            "tolerance_rad_s": 2e-4,
            "uniform_order_claimed": False,
        },
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                root / "src/surface/dynamic_station.py",
                root / "src/surface/liquid.py",
                root / "src/surface/curves.py",
            ]
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
        raise SystemExit("Dynamic station independent reference failed")


if __name__ == "__main__":
    main()
