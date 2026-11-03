"""Independent sampled PI cascade + DOP853 liquid/shaft reference."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_liquid_transient import coupled_model, coupled_rhs

from surface import ControlledLiquidStation, PIDController


def reference_station(dt=0.002):
    physics = coupled_model(dt)
    return ControlledLiquidStation(
        physics,
        "h",
        PIDController(1e-6, 2e-6, 0, 0, 1.3, tracking_time_s=2),
        {n: PIDController(0.7, 3, 0, -10, 20, tracking_time_s=0.2) for n in physics.names},
        control_interval_s=0.02,
    )


def independent_reference():
    # This recurrence does not call the project's PID or controller class.
    # The independently written hydraulic RHS is shared with the previous
    # pipe/shaft oracle; only its prescribed motor torques are replaced here.
    y = np.asarray([550000.0, 500000.0, 0.002, 100.0, 110.0])
    outer_i = 0.0
    inner_i = np.zeros(2)
    previous_ratio = 0.0
    samples = []
    dt = 0.02
    for tick in range(10):
        actual_ratio = float(np.mean(y[3:] / [100, 110]))
        error = 570000 - y[0]
        if tick == 0:
            ratio = actual_ratio
            outer_i = ratio - 1e-6 * error
        else:
            outer_i += dt / (2 + dt) * (actual_ratio - previous_ratio) + 2e-6 * error * dt
            raw = 1e-6 * error + outer_i
            ratio = float(np.clip(raw, 0, 1.3))
            outer_i += dt / (2 + dt) * (ratio - raw)
        previous_ratio = ratio
        speed_error = ratio * np.asarray([100.0, 110.0]) - y[3:]
        if tick == 0:
            torques = np.zeros(2)
            inner_i = -0.7 * speed_error
        else:
            inner_i += 3 * speed_error * dt
            raw = 0.7 * speed_error + inner_i
            torques = np.clip(raw, -10, 20)
            inner_i += dt / (0.2 + dt) * (torques - raw)
        samples.append(
            {"time_s": tick * dt, "desired_speed_ratio": ratio, "torques_nm": torques.tolist()}
        )

        def rhs(t, state, held_torques=torques):
            out = coupled_rhs(t, state)
            out[3:] += (held_torques - np.asarray([5.0, 7.0])) / [0.5, 0.7]
            return out

        integration = solve_ivp(
            rhs, (tick * dt, (tick + 1) * dt), y, method="DOP853", rtol=1e-12, atol=1e-12
        )
        if not integration.success:
            raise RuntimeError("Independent sampled reference integration failed")
        y = integration.y[:, -1]
    return y, samples


def check_reference():
    expected, oracle_samples = independent_reference()
    rows = []
    for dt in (0.004, 0.002, 0.001):
        model = reference_station(dt)
        initial = model.initialize(
            physical_initial={
                "pressures_pa": {"h": 550000, "t": 500000},
                "pipe_volume_rates_m3_s": {"line": 0.002},
                "speeds_rad_s": {"a": 100, "b": 110},
            }
        )
        result = model.advance(initial, 0.2, pressure_setpoint_pa=570000)
        end = result["state"]["physical"]
        actual = np.asarray(
            [
                end["pressures_pa"]["h"],
                end["pressures_pa"]["t"],
                end["pipe_volume_rates_m3_s"]["line"],
                end["speeds_rad_s"]["a"],
                end["speeds_rad_s"]["b"],
            ]
        )
        rows.append(
            {
                "dt_s": dt,
                "pressure_error_pa": float(max(abs(actual[:2] - expected[:2]))),
                "speed_error_rad_s": float(max(abs(actual[3:] - expected[3:]))),
                "samples": len(result["control_samples"]),
                "max_command_error_nm": max(
                    abs(sample["motor_commands_nm"][n] - ref["torques_nm"][i])
                    for sample, ref in zip(result["control_samples"], oracle_samples)
                    for i, n in enumerate(model.names)
                ),
                "max_energy_residual_j": max(abs(s["energy_residual_j"]) for s in result["steps"]),
            }
        )
    ratios = [rows[i]["pressure_error_pa"] / rows[i + 1]["pressure_error_pa"] for i in range(2)]
    passed = bool(
        min(ratios) > 3.5
        and rows[-1]["pressure_error_pa"] < 0.5
        and rows[-1]["speed_error_rad_s"] < 1e-5
        and rows[-1]["max_command_error_nm"] < 1e-4
        and all(r["samples"] == 10 and r["max_energy_residual_j"] < 1e-5 for r in rows)
    )
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "scripts/check_liquid_transient.py"] + [
        root / "src/surface" / n
        for n in (
            "controlled_liquid.py",
            "liquid_transient.py",
            "liquid_relief.py",
            "pid.py",
            "dynamic_station.py",
            "liquid.py",
            "curves.py",
        )
    ]
    return {
        "passed": passed,
        "reference_final_state": expected.tolist(),
        "refinements": rows,
        "pressure_refinement_ratios": ratios,
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
        raise SystemExit("Controlled liquid independent reference failed")


if __name__ == "__main__":
    main()
