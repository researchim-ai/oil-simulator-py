"""Reproduce the synthetic export-capacity what-if after numerical qualification."""

import argparse
import copy
import gzip
import json
from pathlib import Path

from check_facility_refinement_pair import completed
from check_fim_facility_forecast import audit
from export_field_forecast import production_series

ROOT = Path(__file__).resolve().parent


def compare(root):
    long = json.loads((root / "decade_refinement.json").read_text())
    upgrade = json.loads((root / "upgrade_qualified_refinement.json").read_text())
    if not long["accepted"] or not upgrade["accepted"]:
        raise ValueError("Both declared refinement gates must pass")
    summaries, bundles = [], []
    for name in ("annual_current", "upgrade_5day"):
        summary, bundle, _ = completed(root / name, 365)
        audit(
            bundle["config"],
            bundle["result"],
            expected_days=365,
            expected_schedule=bundle["schedule"],
        )
        summaries.append(summary)
        bundles.append(bundle)
    python = lambda s: {k: v for k, v in s["source_sha256"].items() if k.endswith(".py")}
    if python(summaries[0]) != python(summaries[1]):
        raise ValueError("Different numerical source versions")
    a, b = bundles
    if a["schedule"] != b["schedule"]:
        raise ValueError("Different operating schedules")
    x, y = [copy.deepcopy(bundle["config"]) for bundle in bundles]
    valve = lambda c: c["simulation"]["coupled_facility"]["facility"]["gas_export"]["export_valve"]
    areas = [valve(c)["throat_area_m2"] for c in (x, y)]
    valve(y)["throat_area_m2"] = valve(x)["throat_area_m2"]
    for config in (x, y):
        for key in ("max_time_step_days", "initial_time_step_days"):
            config["simulation"].pop(key, None)
    if x != y:
        raise ValueError("Physical differences beyond the declared valve area")
    with gzip.open(root / "decade_10day/trajectory.json.gz", "rt") as stream:
        decade = json.load(stream)
    prefix = [s for s in decade["result"]["accepted_substeps"] if s["time_days"] <= 365]
    annual = a["result"]["accepted_substeps"]
    if len(prefix) != len(annual):
        raise ValueError("Different annual and decade prefix clocks")
    for left, right in zip(prefix, annual, strict=True):
        for key in ("time_days", "dt_s", "withdrawn_component_mass_kg", "surface"):
            if left[key] != right[key]:
                raise ValueError("Annual accepted observations differ from qualified decade prefix")
    results = []
    for bundle in bundles:
        row = production_series(bundle["config"], bundle["result"])[-1]
        plant = bundle["result"]["accepted_substeps"][-1]["facility_coupling"]["facility"]
        results.append(
            {
                "final_kpis": row,
                "final_gas_valve_utilization": plant["measurements"]["gas_valve_utilization"][
                    "value"
                ],
                "final_maximum_limit_utilizations": [
                    {"name": limit["name"], "utilization": limit["value"] / limit["maximum"]}
                    for limit in plant["limits"]
                    if limit["maximum"] is not None and limit["maximum"] > 0
                ],
            }
        )
    return {
        "accepted": True,
        "scope": "Numerically qualified synthetic production-only quasi-steady capacity what-if; no experimental equipment calibration or injection-network claim",
        "same_python": True,
        "same_schedule": True,
        "annual_accepted_observations_equal_qualified_decade_prefix": True,
        "only_changed_physical_parameter": "gas_export.export_valve.throat_area_m2",
        "valve_area_m2": areas,
        "time_refinement_evidence": ["decade_refinement.json", "upgrade_qualified_refinement.json"],
        "baseline": results[0],
        "upgrade": results[1],
        "oil_cumulative_ratio": results[1]["final_kpis"]["oil_cumulative_stock_m3"]
        / results[0]["final_kpis"]["oil_cumulative_stock_m3"],
        "energy_ratio": results[1]["final_kpis"]["cumulative_equipment_drive_energy_kwh"]
        / results[0]["final_kpis"]["cumulative_equipment_drive_energy_kwh"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
