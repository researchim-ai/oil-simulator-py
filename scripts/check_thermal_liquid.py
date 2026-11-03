"""Independent caloric oracles, actual thermal trip and optional full-run gate."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surface import LiquidHeatNetwork, LiquidTransientNetwork, ThermalLiquidStation


def manufactured_record(physics, duration=1.0, time_s=0.0):
    """Momentum-independent transport probes; NOT a hydraulic validation."""
    return {
        "time_s": time_s,
        "dt_s": duration,
        "final_pressures_pa": {n: physics.reference_pressure_pa for n in physics.free},
        "midpoint_branch_volume_rates_m3_s": {b.name: 0.0 for b in physics._branches},
        "boundary_mass_transfers_in_kg": {
            n.name: 0.0 for n in physics.network.nodes if n.pressure_pa is not None
        },
        "dissipation_heat_j": {
            "friction": {b.name: 0.0 for b in physics.network.pipes + physics.network.valves},
            **{k: {n: 0.0 for n in physics.names} for k in ("pump", "shaft_drag", "motor")},
        },
        **{
            k: 0.0
            for k in (
                "electrical_energy_j",
                "boundary_work_in_j",
                "shaft_kinetic_energy_change_j",
                "pipe_kinetic_energy_change_j",
                "acoustic_energy_change_j",
                "friction_heat_j",
                "pump_heat_j",
                "shaft_drag_heat_j",
                "motor_heat_j",
            )
        },
    }


def reference_pipe(cells=4, dt=0.01, compact=False):
    p = LiquidTransientNetwork.from_dict(
        {
            "network": {
                "model": "incompressible_liquid",
                "fluid": {"density_kg_m3": 1000, "viscosity_pa_s": 0.001},
                "nodes": [
                    {"name": "in", "pressure_pa": 500000},
                    {"name": "a"},
                    {"name": "out", "pressure_pa": 500000},
                ],
                "pipes": [
                    {
                        "name": "line",
                        "start": "in",
                        "end": "a",
                        "length_m": 1,
                        "diameter_m": math.sqrt(4 / (1000 * math.pi)),
                    }
                ],
                "pumps": [],
                "valves": [{"name": "v", "start": "a", "end": "out", "diameter_m": 0.01}],
            },
            "storage_volumes_m3": {"a": 0.001},
            "bulk_modulus_pa": 1e8,
            "reference_pressure_pa": 500000,
        }
    )
    return LiquidHeatNetwork(
        p,
        1000,
        {"in": 310, "out": 300},
        reference_temperature_k=300,
        pipe_cells={"line": cells},
        maximum_dt_s=dt,
        temporal_error_control=False,
        maximum_temperature_change_k=100,
        store_step_temperatures=not compact,
    )


def pipe_probe(cells=4, dt=0.01, duration=2.0, compact=False):
    m = reference_pipe(cells, dt, compact)
    rec = manufactured_record(m.physics, duration)
    rec["midpoint_branch_volume_rates_m3_s"] = {"line": 0.001, "v": 0.001}
    rec["boundary_mass_transfers_in_kg"] = {"in": duration, "out": -duration}
    return m, m.initialize(m.physics.initialize()), rec


def check_reference():
    rows = []
    for dt in (0.02, 0.01, 0.005):
        m, s, rec = pipe_probe(dt=dt)
        r = m.advance_record(s, rec)
        # The N-cell pipe is an Erlang residence chain. Reference temperatures
        # use incomplete gamma and an independent quadrature for the outlet
        # tank, NOT the project's transport matrix or integration scheme.
        expected = {f"line::{i}": 300 + 10 * gammainc(i + 1, 4 * 2) for i in range(4)}
        expected["a"] = (
            300
            + 10
            * quad(
                lambda u: math.exp(-(2 - u)) * gammainc(4, 4 * u), 0, 2, epsabs=1e-12, epsrel=1e-12
            )[0]
        )
        error = max(abs(r["state"]["temperatures_k"][n] - v) for n, v in expected.items())
        rows.append(
            {
                "dt_s": dt,
                "max_temperature_error_k": error,
                "accepted_steps": len(r["steps"]),
                "energy_residual_j": r["combined_energy_residual_j"],
            }
        )
    ratios = [
        a["max_temperature_error_k"] / b["max_temperature_error_k"] for a, b in zip(rows, rows[1:])
    ]
    # Independent two-capacity cooling: analytical matrix exponential.
    pipe = reference_pipe()
    data = pipe.to_dict()
    data.update(
        walls={
            "a": {
                "heat_capacity_j_k": 2000,
                "fluid_ua_w_k": 100,
                "ambient_ua_w_k": 50,
                "ambient_temperature_k": 290,
            }
        },
        maximum_dt_s=0.01,
    )
    m = LiquidHeatNetwork.from_dict(data)
    initial = {n: 300 for n in m.names}
    initial["a"] = 320
    s = m.initialize(m.physics.initialize(), initial, {"a": 300})
    r = m.advance_record(s, manufactured_record(m.physics, 10))
    exact = 290 + expm(np.array([[-0.1, 0.1], [0.05, -0.075]]) * 10) @ np.array([30, 10])
    wall_error = max(
        abs(r["state"]["temperatures_k"]["a"] - exact[0]),
        abs(r["state"]["wall_temperatures_k"]["a"] - exact[1]),
    )
    entropy = min(x["passive_exchange_entropy_j_k"] for x in r["steps"])
    # A real coupled calculation with zero flow and a known heat input.
    cfg = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())["system"]
    cfg["temperature_trips"] = [
        {
            "name": "linear-temperature",
            "location": "t",
            "threshold_k": 300.0048,
            "delay_s": 0.1,
            "trip_station": True,
        }
    ]
    system = ThermalLiquidStation.from_dict(cfg)
    initial = {
        "protected_initial": {
            "controller_initial": {"physical_initial": {"pressures_pa": {"h": 500000, "t": 500000}}}
        }
    }
    state = system.initialize(recycling_initial=initial)
    coupled = system.advance(
        state,
        1.25,
        mode="off",
        valve_openings={"export": 0},
        heat_duties_w={"t": 100},
        temperature_control_modes={"cool-header": "off", "cool-suction": "off"},
    )
    trip = next(
        x["time_s"] for x in coupled["temperature_events"] if x["type"] == "temperature_trip"
    )
    final_error = abs(
        coupled["state"]["heat"]["temperatures_k"]["t"] - (300 + 100 / (10 * 2000) * 1.25)
    )
    return {
        "passed": min(ratios) > 3.9
        and rows[-1]["max_temperature_error_k"] < 1e-4
        and wall_error < 2e-6
        and entropy >= 0
        and abs(trip - 1.1) < 1e-10
        and final_error < 1e-8,
        "pipe_refinements": rows,
        "temporal_refinement_ratios": ratios,
        "wall_cooling_error_k": wall_error,
        "minimum_passive_exchange_entropy_j_k": entropy,
        "actual_linear_trip_time_s": trip,
        "expected_linear_trip_time_s": 1.1,
        "actual_linear_temperature_error_k": final_error,
    }


def check_trajectory(path):
    raw = Path(path).read_bytes()
    d = json.loads(raw)
    records = d["records"]
    thermal = [s for r in records for s in r["thermal_steps"]]
    audits = [s for r in records for s in r["thermal_audits"]]
    max_heat = max(abs(x["energy_residual_j"]) for x in thermal)
    max_combined = max(abs(x["combined_energy_residual_j"]) for x in audits)
    hot = records[0]["state"]["heat"]["temperatures_k"]["in"] > 300
    trip = any(x["type"] == "temperature_trip" for x in records[1]["temperature_events"])
    denied = any(x["type"] == "reset_denied" for x in records[2]["temperature_events"])
    cooling = records[3]["state"]["heat"]["temperatures_k"]["h"] < 300.13
    reset = (
        any(x["type"] == "reset" for x in records[4]["temperature_events"])
        and not records[4]["latched_temperature_trips"]
    )
    utility = max(
        -math.fsum(v["applied_duty_w"] for v in s["controllers"].values())
        for r in records
        for s in r["temperature_samples"]
    )
    return {
        "passed": hot
        and trip
        and denied
        and cooling
        and reset
        and utility <= 1000 + 1e-9
        and max_heat < 1e-5
        and max_combined < 1e-5,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "duration_s": math.fsum(r["duration_s"] for r in records),
        "finite_suction_heats": hot,
        "hot_trip": trip,
        "unsafe_reset_denied": denied,
        "active_cooling": cooling,
        "safe_temperature_reset": reset,
        "peak_shared_cooling_w": utility,
        "max_thermal_energy_residual_j": max_heat,
        "max_combined_energy_residual_j": max_combined,
        "hydraulic_steps": sum(len(r["steps"]) for r in records),
        "thermal_steps": len(thermal),
        "hydraulic_retries": sum(r["failed_steps"] for r in records),
        "thermal_retries": sum(r["failed_thermal_steps"] for r in records),
        "external_cooling_duty_j": math.fsum(x["external_heat_duty_j"] for x in thermal),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path)
    args = parser.parse_args()
    result = {"reference": check_reference()}
    if args.trajectory:
        result["trajectory"] = check_trajectory(args.trajectory)
    result["passed"] = all(v["passed"] for v in result.values())
    files = [
        "liquid_heat.py",
        "thermal_liquid.py",
        "thermal_liquid_schedule.py",
        "vessel_wall.py",
        "liquid_transient.py",
        "liquid_recycle.py",
        "liquid_protection.py",
        "controlled_liquid.py",
        "pid.py",
        "actuator.py",
        "liquid.py",
        "rotating_drive.py",
        "dynamic_station.py",
        "station.py",
        "curves.py",
        "operation.py",
    ]
    paths = [ROOT / "src/surface" / n for n in files] + [
        Path(__file__).resolve(),
        ROOT / "configs/surface_thermal_liquid.json",
    ]
    result["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
