"""Independent analytic moving-orifice oracle and optional station trajectory gate."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import LiquidTransientNetwork


def check_reference():
    # y=sqrt(p-p_out) obeys y'=-Cd*A*x(t)*sqrt(2/rho)/(2*C).
    # Exact actuator x(t)=1-.8*exp(-t/.05). No project hydraulic/PID
    # implementation is used to construct this analytic trajectory.
    exact = (
        500000
        + (
            math.sqrt(100000)
            - 0.7
            * math.pi
            * 0.01**2
            / 4
            * math.sqrt(2 / 1000)
            / (2e-7)
            * (0.2 - 0.8 * 0.05 * (-math.expm1(-4)))
        )
        ** 2
    )
    records = []
    for dt in (0.02, 0.01, 0.005):
        m = LiquidTransientNetwork.from_dict(
            {
                "network": {
                    "model": "incompressible_liquid",
                    "fluid": {"density_kg_m3": 1000, "viscosity_pa_s": 0.001},
                    "nodes": [{"name": "a"}, {"name": "out", "pressure_pa": 500000}],
                    "pipes": [],
                    "pumps": [],
                    "valves": [
                        {
                            "name": "v",
                            "start": "a",
                            "end": "out",
                            "diameter_m": 0.01,
                            "discharge_coefficient": 0.7,
                            "opening": 0,
                            "check_valve": True,
                        }
                    ],
                },
                "storage_volumes_m3": {"a": 1},
                "bulk_modulus_pa": 1e7,
                "reference_pressure_pa": 500000,
                "maximum_dt_s": dt,
                "temporal_error_control": False,
                "actuated_valves": {
                    "v": {
                        "opening_rate_per_s": 100,
                        "closing_rate_per_s": 100,
                        "fail_position": 1,
                        "time_constant_s": 0.05,
                        "max_position_change": 1,
                    }
                },
            }
        )
        r = m.advance(
            m.initialize({"a": 600000}, actuator_positions={"v": 0.2}), 0.2, valve_openings={"v": 1}
        )
        records.append(
            {
                "dt_s": dt,
                "pressure_error_pa": abs(r["state"]["pressures_pa"]["a"] - exact),
                "position_error": abs(
                    r["state"]["actuator_positions"]["v"] - (1 - 0.8 * math.exp(-4))
                ),
                "max_mass_residual_kg": max(abs(s["mass_residual_kg"]) for s in r["steps"]),
                "max_energy_residual_j": max(abs(s["energy_residual_j"]) for s in r["steps"]),
                "accepted_steps": len(r["steps"]),
                "rejected_steps": r["failed_steps"],
            }
        )
    ratios = [a["pressure_error_pa"] / b["pressure_error_pa"] for a, b in zip(records, records[1:])]
    return {
        "passed": min(ratios) > 3.8
        and records[-1]["pressure_error_pa"] < 0.5
        and max(r["position_error"] for r in records) < 1e-13,
        "exact_pressure_pa": exact,
        "refinement_ratios": ratios,
        "runs": records,
    }


def check_trajectory(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    records = data["records"]
    steps = [s for r in records for s in r["steps"]]
    closed = records[1]
    late = [s for s in closed["steps"] if s["time_s"] >= 9.5]
    healthy = (
        not records[0]["latched_trips"]
        and not closed["latched_trips"]
        and not records[2]["latched_trips"]
    )
    zero_export = max(abs(s["midpoint_branch_volume_rates_m3_s"]["export"]) for s in late) < 1e-10
    min_flow = min(
        min(s["midpoint_branch_volume_rates_m3_s"][n] for n in ("p1", "p2")) for s in late
    )
    fault = "minflow-p2" in records[3]["latched_trips"]
    opened = (
        records[4]["state"]["protected"]["controlled"]["physical"]["actuator_positions"]["recycle"]
        > 0.999
    )
    mass = max(s["max_node_mass_residual_kg"] for s in steps)
    energy = max(abs(s["energy_residual_j"]) for s in steps)
    return {
        "passed": healthy
        and zero_export
        and min_flow > 0.00125
        and fault
        and opened
        and mass < 1e-8
        and energy < 1e-4,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "duration_s": math.fsum(r["duration_s"] for r in records),
        "healthy_periods_no_trip": healthy,
        "closed_export_zero_flow": zero_export,
        "closed_late_min_pump_flow_m3_s": min_flow,
        "stuck_closed_guard_trip": fault,
        "fail_safe_actual_open": opened,
        "accepted_steps": len(steps),
        "rejected_steps": sum(r["failed_steps"] for r in records),
        "max_node_mass_residual_kg": mass,
        "max_energy_residual_j": energy,
        "recycled_volume_m3": math.fsum(r["recycled_volume_m3"]["recycle"] for r in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"analytic": check_reference()}
    if args.trajectory:
        result["trajectory"] = check_trajectory(args.trajectory)
    root = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__).resolve(),
            root / "src/surface/liquid_transient.py",
            root / "src/surface/liquid_recycle.py",
            *[
                root / "src/surface" / name
                for name in (
                    "actuator.py",
                    "pid.py",
                    "controlled_liquid.py",
                    "liquid_protection.py",
                    "liquid_relief.py",
                    "liquid_recycle_schedule.py",
                    "liquid.py",
                    "curves.py",
                    "dynamic_station.py",
                    "station.py",
                    "rotating_drive.py",
                    "operation.py",
                )
            ],
            root / "configs/surface_recycling_liquid.json",
        ]
    }
    result["passed"] = all(v["passed"] for k, v in result.items() if k != "source_sha256")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
