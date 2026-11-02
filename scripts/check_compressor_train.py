"""Independent choked blowdown and manufactured recycle equilibria."""

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import DynamicCompressorTrain


def nozzle_flux(p, t, pb, rg, gamma):
    ratio = pb / p
    critical = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    factor = (
        gamma * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
        if ratio <= critical
        else 2 * gamma / (gamma - 1) * (ratio ** (2 / gamma) - ratio ** ((gamma + 1) / gamma))
    )
    return p / math.sqrt(rg * t) * math.sqrt(factor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "configs/surface_compressor_train.json").read_text())["system"]
    gas = base["compressor"]["gas"]
    rg = gas["specific_gas_constant_j_kg_k"]
    gamma = gas["heat_capacity_ratio"]
    cp = gamma * rg / (gamma - 1)
    blowdown = []
    for dt in (0.04, 0.02, 0.01):
        data = copy.deepcopy(base)
        data.update(
            maximum_dt_s=dt,
            control_interval_s=0.2,
            heat_transfer_w_k=0.0,
            max_recycle_cooling_w=None,
            max_recycle_heating_w=None,
        )
        p = DynamicCompressorTrain.from_dict(data)
        initial = p.initialize(pressure_pa=4e5, temperature_k=300, recycle_position=0)
        result = p.advance(
            initial, 0.2, mode="off", export_opening=1, recycle_actuator_mode="stuck"
        )
        mass0 = 4e5 * data["plenum_volume_m3"] / (rg * 300)
        flow0 = data["export_area_m2"] * nozzle_flux(
            4e5, 300, data["export_back_pressure_pa"], rg, gamma
        )
        power = (gamma + 1) / 2
        fraction = (1 + (power - 1) * flow0 / mass0 * 0.2) ** (-1 / (power - 1))
        predicted_p = 4e5 * fraction**gamma
        predicted_t = 300 * fraction ** (gamma - 1)
        if data["export_back_pressure_pa"] / predicted_p > (2 / (gamma + 1)) ** (
            gamma / (gamma - 1)
        ):
            raise RuntimeError("Blowdown oracle left choked domain")
        blowdown.append(
            {
                "dt_s": dt,
                "mass_error_kg": abs(result["state"]["mass_kg"] - mass0 * fraction),
                "temperature_error_k": abs(result["temperature_k"] - predicted_t),
                "reference_mass_kg": mass0 * fraction,
                "reference_temperature_k": predicted_t,
            }
        )
    ratios = [a["mass_error_kg"] / b["mass_error_kg"] for a, b in zip(blowdown, blowdown[1:])]
    equilibria = []
    for recycle_fraction in (0.6, 1.0):
        data = copy.deepcopy(base)
        data.update(heat_transfer_w_k=0.0, max_recycle_cooling_w=None, max_recycle_heating_w=None)
        table = data["compressor"]["map"]
        pin = data["suction_pressure_pa"]
        tin = data["suction_temperature_k"]
        corrected = table["corrected_speed_rad_s"][1]
        pr = table["pressure_ratio"][1][1]
        eta = table["isentropic_efficiency"][1][1]
        speed = corrected * math.sqrt(tin / table["reference_temperature_k"])
        mass = (
            table["corrected_mass_flow_kg_s"][1][1]
            * (pin / table["reference_pressure_pa"])
            / math.sqrt(tin / table["reference_temperature_k"])
        )
        temperature = tin * (1 + (pr ** ((gamma - 1) / gamma) - 1) / eta)
        pressure = pin * pr
        flux_recycle = nozzle_flux(pressure, temperature, pin, rg, gamma)
        flux_export = nozzle_flux(pressure, temperature, data["export_back_pressure_pa"], rg, gamma)
        data["recycle_area_m2"] = mass * recycle_fraction / flux_recycle
        data["export_area_m2"] = mass * max(1e-6, 1 - recycle_fraction) / flux_export
        gas_power = mass * cp * (temperature - tin)
        shaft = gas_power / data["compressor"]["mechanical_efficiency"]
        torque = shaft / speed + data["drive"]["viscous_drag_nm_s"] * speed
        p = DynamicCompressorTrain.from_dict(data)
        initial = p.initialize(
            pressure_pa=pressure,
            temperature_k=temperature,
            angular_velocity_rad_s=speed,
            recycle_position=1.0,
        )
        result = p.advance(
            initial,
            0.5,
            mode="run",
            speed_control_mode="manual",
            motor_torque_nm=torque,
            anti_surge_mode="manual",
            manual_recycle_opening=1.0,
            recycle_actuator_mode="stuck",
            export_opening=0.0 if recycle_fraction == 1 else 1.0,
        )
        row = {
            "recycle_fraction": recycle_fraction,
            "mass_drift_kg": abs(result["state"]["mass_kg"] - initial.mass_kg),
            "energy_drift_j": abs(result["state"]["internal_energy_j"] - initial.internal_energy_j),
            "speed_drift_rad_s": abs(result["state"]["angular_velocity_rad_s"] - speed),
            "net_suction_mass_kg": sum(x["net_suction_mass_kg"] for x in result["steps"]),
            "export_mass_kg": sum(x["export_mass_kg"] for x in result["steps"]),
            "recycle_mass_kg": sum(x["recycle_mass_kg"] for x in result["steps"]),
            "conditioning_heat_j": sum(x["conditioning_heat_j"] for x in result["steps"]),
            "reference_conditioning_heat_j": mass
            * recycle_fraction
            * cp
            * (temperature - tin)
            * 0.5,
        }
        row["passed"] = (
            row["mass_drift_kg"] < 1e-8
            and row["energy_drift_j"] < 1e-4
            and row["speed_drift_rad_s"] < 1e-7
            and abs(row["conditioning_heat_j"] - row["reference_conditioning_heat_j"]) < 1e-5
        )
        row["passed"] = bool(row["passed"])
        equilibria.append(row)
    passed = (
        all(x > 1.8 for x in ratios)
        and blowdown[-1]["mass_error_kg"] < 0.001
        and all(r["passed"] for r in equilibria)
    )
    files = [
        Path(__file__),
        root / "configs/surface_compressor_train.json",
        *(
            root / "src/surface" / name
            for name in [
                "compressor_map.py",
                "compressor_train.py",
                "compressor_schedule.py",
                "gas.py",
                "gas_valve.py",
                "pid.py",
                "actuator.py",
                "rotating_drive.py",
            ]
        ),
    ]
    report = {
        "scope": "Analytical choked ideal-gas blowdown + manufactured partial/full recycle equilibria. Independent nozzle/isentropic formulas, no equipment calibration or post-surge validation.",
        "passed": passed,
        "blowdown": blowdown,
        "mass_error_reduction_ratios": ratios,
        "equilibria": equilibria,
        "sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("passed" if passed else "failed")
    if not passed:
        parser.exit(1, "Compressor reference gate failed\n")


if __name__ == "__main__":
    main()
