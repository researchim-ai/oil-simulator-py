"""Independent laminar hydraulic/ODE convergence gate for the coupled shaft."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import DynamicPumpNetwork, Liquid, LiquidNetwork, Node, Pipe, Pump, RotatingDrive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # All reference equations below are independent of surface flow/torque helpers.
    density, viscosity, diameter, length = 1000.0, 0.1, 0.1, 100.0
    head, curve, efficiency, nominal = 20.0, 1e6, 0.8, 100.0
    inertia, drag, idle, torque, horizon, initial = 1.0, 0.02, 20.0, 6.0, 4.0, 60.0
    resistance = 128 * viscosity * length / (math.pi * diameter**4)
    quadratic = density * 9.80665 * curve

    def reference_flow(speed):
        rise = density * 9.80665 * head * (speed / nominal) ** 2 - 50000
        if rise <= 0:
            return 0.0
        return 2 * rise / (resistance + math.sqrt(resistance**2 + 4 * quadratic * rise))

    def reference_load(speed):
        flow = reference_flow(speed)
        base = (50000 + resistance * flow) * flow / efficiency
        return (base + idle * (speed / nominal) ** 3) / speed if speed else 0.0

    reference = solve_ivp(
        lambda t, y: [(torque - drag * y[0] - reference_load(y[0])) / inertia],
        (0, horizon),
        [initial],
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    if not reference.success:
        raise RuntimeError("Independent ODE integration failed")
    target = float(reference.y[0, -1])
    network = LiquidNetwork(
        Liquid(density, viscosity, 2000),
        [Node("in", pressure_pa=2e5), Node("delivery"), Node("out", pressure_pa=2.5e5)],
        [Pipe("line", "delivery", "out", length, diameter)],
        [
            Pump(
                "pump",
                "in",
                "delivery",
                head,
                curve,
                0.004,
                efficiency=efficiency,
                check_valve=True,
            )
        ],
    )
    drive = RotatingDrive(inertia, 150, 20, 10, viscous_drag_nm_s=drag)
    rows = []
    for dt in (0.2, 0.1, 0.05):
        model = DynamicPumpNetwork(
            network,
            drive,
            nominal,
            measurement_node="delivery",
            control_interval_s=0.2,
            maximum_dt_s=dt,
            max_speed_change_rad_s=10,
            no_flow_power_w_at_nominal=idle,
        )
        result = model.advance(
            model.initialize(initial), horizon, mode="manual", motor_torque_nm=torque
        )
        max_flow_error = max(
            abs(
                s["midpoint_flow_m3_s"]
                - reference_flow(0.5 * (s["initial_speed_rad_s"] + s["final_speed_rad_s"]))
            )
            for s in result["steps"]
        )
        max_reynolds = max(
            4 * density * abs(s["midpoint_flow_m3_s"]) / (math.pi * viscosity * diameter)
            for s in result["steps"]
        )
        if max_reynolds >= 2000:
            raise RuntimeError("Poiseuille oracle left its laminar domain")
        rows.append(
            {
                "maximum_dt_s": dt,
                "speed_error_rad_s": abs(result["state"]["angular_velocity_rad_s"] - target),
                "flow_error_m3_s": max_flow_error,
                "max_reynolds": max_reynolds,
                "max_energy_residual_j": max(abs(s["energy_residual_j"]) for s in result["steps"]),
                "accepted_steps": len(result["steps"]),
                "failed_steps": result["failed_steps"],
            }
        )
    ratios = [a["speed_error_rad_s"] / b["speed_error_rad_s"] for a, b in zip(rows, rows[1:])]
    passed = (
        all(x > 3.5 for x in ratios)
        and rows[-1]["speed_error_rad_s"] < 2e-5
        and all(r["flow_error_m3_s"] < 1e-9 for r in rows)
    )
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        *(
            root / "src/surface" / n
            for n in ["pump_drive.py", "pid.py", "rotating_drive.py", "liquid.py", "curves.py"]
        ),
    ]
    report = {
        "scope": "One pressure-driven liquid pump, laminar pipe, explicit quadratic map and no-flow loss; independent Poiseuille/quadratic oracle + high-accuracy DOP853. Not equipment calibration or a field/GPU benchmark.",
        "passed": passed,
        "reference_speed_rad_s": target,
        "reference_evaluations": reference.nfev,
        "cases": rows,
        "error_reduction_ratios": ratios,
        "sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("passed" if passed else "failed")
    if not passed:
        parser.exit(1, "Coupled pump convergence gate failed\n")


if __name__ == "__main__":
    main()
