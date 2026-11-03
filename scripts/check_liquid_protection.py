"""Independent compliance/orifice blowdown and sampled constant-signal trip."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_controlled_liquid import reference_station

from surface import (
    ControlledLiquidStation,
    Liquid,
    LiquidNetwork,
    LiquidRelief,
    LiquidTransientNetwork,
    LiquidTrip,
    Node,
    ProtectedLiquidStation,
    ThrottleValve,
)


def relief_model(dt=0.01, adaptive=False):
    network = LiquidNetwork(
        Liquid(1000, 0.01, 2000),
        [Node("v"), Node("out", pressure_pa=500000)],
        [],
        valves=[ThrottleValve("relief", "v", "out", 0.002, opening=0, check_valve=True)],
    )
    return LiquidTransientNetwork(
        network,
        {"v": 0.1},
        1e8,
        reference_pressure_pa=500000,
        maximum_dt_s=dt,
        temporal_error_control=adaptive,
        relief_valves={"relief": LiquidRelief(10000, 20000)},
    )


def timer_model():
    data = reference_station().to_dict()
    data["physics"]["bulk_modulus_pa"] = 5e7
    controller = ControlledLiquidStation.from_dict(data)
    return ProtectedLiquidStation(
        controller,
        [
            LiquidTrip(
                "high-inlet",
                "in",
                "pressure_pa",
                "above",
                350000,
                trip_station=True,
                delay_s=0.03,
                reset_margin=10000,
            )
        ],
        0.01,
    )


def check_reference():
    coefficient = 0.62 * math.pi * 0.002**2 / 4 * math.sqrt(2 / 1000)
    compliance = 0.1 / 1e8
    exact = 500000 + (math.sqrt(100000) - coefficient / (2 * compliance)) ** 2
    rows = []
    for dt in (0.04, 0.02, 0.01):
        model = relief_model(dt)
        result = model.advance(model.initialize({"v": 600000}), 1.0)
        rows.append(
            {
                "dt_s": dt,
                "pressure_error_pa": abs(result["state"]["pressures_pa"]["v"] - exact),
                "max_mass_residual_kg": max(
                    s["max_node_mass_residual_kg"] for s in result["steps"]
                ),
                "max_energy_residual_j": max(abs(s["energy_residual_j"]) for s in result["steps"]),
            }
        )
    ratios = [rows[i]["pressure_error_pa"] / rows[i + 1]["pressure_error_pa"] for i in range(2)]

    # Independent partial-lift formula, including the valve's finite area.
    def partial_rhs(t, y):
        dp = y[0] - 500000
        fraction = min(1.0, max(0.0, (dp - 10000) / 10000))
        return [-coefficient * fraction * math.sqrt(max(0.0, dp)) / compliance]

    ref = solve_ivp(partial_rhs, (0, 1.0), [515000.0], method="DOP853", rtol=1e-12, atol=1e-10)
    m = relief_model(0.005)
    partial = m.advance(m.initialize({"v": 515000}), 1.0)
    partial_error = abs(partial["state"]["pressures_pa"]["v"] - ref.y[0, -1])
    protected = timer_model()
    start = protected.initialize()
    result = protected.advance(start, 0.04, pressure_setpoint_pa=570000)
    trips = [e for e in result["events"] if e["type"] == "trip"]
    trip_time = trips[0]["time_s"] if trips else None
    passed = bool(
        min(ratios) > 3.8
        and rows[-1]["pressure_error_pa"] < 0.1
        and ref.success
        and partial_error < 0.01
        and trip_time is not None
        and abs(trip_time - 0.03) < 1e-12
        and all(
            r["max_energy_residual_j"] < 1e-7 and r["max_mass_residual_kg"] < 1e-9 for r in rows
        )
    )
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        root / "scripts/check_controlled_liquid.py",
        root / "scripts/check_liquid_transient.py",
    ] + [
        root / "src/surface" / n
        for n in (
            "liquid_relief.py",
            "liquid_protection.py",
            "liquid_transient.py",
            "controlled_liquid.py",
            "pid.py",
            "liquid.py",
            "curves.py",
            "dynamic_station.py",
        )
    ]
    return {
        "passed": passed,
        "full_lift_reference_pressure_pa": exact,
        "refinements": rows,
        "refinement_ratios": ratios,
        "partial_lift_pressure_error_pa": partial_error,
        "trip_time_s": trip_time,
        "expected_trip_time_s": 0.03,
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
        raise SystemExit("Liquid protection independent reference failed")


if __name__ == "__main__":
    main()
