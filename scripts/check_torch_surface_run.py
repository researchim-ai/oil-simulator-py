"""Compare reporting endpoints/events after a Torch migration, not internal meshes.

The validation gates are explicit tolerances for the provided constant-property
thermal station scenario. They do not establish calibration or general model
accuracy. Independent mechanical/thermal oracles remain in the check scripts.
"""

import argparse
import json
from pathlib import Path


def compare(reference, candidate):
    rows = []
    if len(reference["records"]) != len(candidate["records"]):
        raise ValueError("Different number of reporting periods")
    for old, new in zip(reference["records"], candidate["records"]):

        def physical(record):
            return record["state"]["recycling"]["protected"]["controlled"]["physical"]

        a, b = physical(old), physical(new)

        def difference(x, y):
            if set(x) != set(y):
                raise ValueError("Unknowns changed")
            return max((abs(x[n] - y[n]) for n in x), default=0.0)

        pressures = difference(a["pressures_pa"], b["pressures_pa"])
        speeds = difference(a["speeds_rad_s"], b["speeds_rad_s"])
        temperatures = difference(
            old["state"]["heat"]["temperatures_k"], new["state"]["heat"]["temperatures_k"]
        )
        walls = difference(
            old["state"]["heat"]["wall_temperatures_k"], new["state"]["heat"]["wall_temperatures_k"]
        )
        events_a = [(e["type"], e["name"], e["time_s"]) for e in old["temperature_events"]]
        events_b = [(e["type"], e["name"], e["time_s"]) for e in new["temperature_events"]]
        latches_equal = all(
            old[key] == new[key] for key in ("latched_temperature_trips", "latched_trips")
        )
        max_energy = max(
            (abs(s["combined_energy_residual_j"]) for s in new["thermal_audits"]), default=0.0
        )
        max_mass = max((s["max_node_mass_residual_kg"] for s in new["steps"]), default=0.0)
        rows.append(
            {
                "period": new["period"],
                "pressure_difference_pa": pressures,
                "speed_difference_rad_s": speeds,
                "temperature_difference_k": temperatures,
                "wall_temperature_difference_k": walls,
                "temperature_events_equal": events_a == events_b,
                "latches_equal": latches_equal,
                "max_combined_energy_residual_j": max_energy,
                "max_node_mass_residual_kg": max_mass,
                "hydraulic_steps": len(new["steps"]),
                "retries": new["failed_steps"],
                "passed": pressures <= 10
                and speeds <= 1e-4
                and temperatures <= 1e-5
                and walls <= 1e-5
                and events_a == events_b
                and latches_equal
                and max_energy <= 1e-5
                and max_mass <= 1e-8,
            }
        )
    return {
        "passed": all(r["passed"] for r in rows),
        "report_periods": rows,
        "note": "Compare report endpoints/events, not adaptive internal step counts",
        "reference_fingerprint": reference["fingerprint"],
        "candidate_fingerprint": candidate["fingerprint"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = compare(json.loads(a.reference.read_text()), json.loads(a.candidate.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "periods": len(result["report_periods"])}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
