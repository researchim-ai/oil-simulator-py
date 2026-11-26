"""Completion-level refinement in addition to the whole-loop acceptance gate."""

import argparse
import gzip
import json
from pathlib import Path

from compare_fim_waterflood import qualify


def compare(candidate, reference, whole_budgets, budgets):
    whole = qualify(candidate, reference, whole_budgets)
    bundles = []
    for study in [candidate, reference]:
        with gzip.open(study / "trajectory.json.gz", "rt") as stream:
            bundles.append(json.load(stream)["result"])
    dates = {round(s["time_days"], 10) for s in bundles[0]["reservoir_snapshots"]}
    histories = [
        {
            round(step["time_days"], 10): step["coupling"]["injection_network"]
            for step in result["accepted_substeps"]
            if round(step["time_days"], 10) in dates
        }
        for result in bundles
    ]
    errors = {
        "connected_pressure_pa": 0.0,
        "component_rate_scaled": 0.0,
        "valve_state_mismatches": 0,
    }
    for date in dates:
        left, right = (h[date] for h in histories)
        for a, b in zip(left["perforations"], right["perforations"], strict=True):
            if (a["well"], a["depth_m"]) != (b["well"], b["depth_m"]):
                raise ValueError("Different completion geometry")
            name = a["well"]
            if left["wells"][name]["bhp_status"] != right["wells"][name]["bhp_status"]:
                raise ValueError("Different physical pressure status")
            if left["wells"][name]["bhp_status"] != "isolated_gauge":
                errors["connected_pressure_pa"] = max(
                    errors["connected_pressure_pa"],
                    abs(a["well_pressure_pa"] - b["well_pressure_pa"]),
                )
            errors["valve_state_mismatches"] += a["valve_open"] != b["valve_open"]
            for x, y in zip(a["component_mass_rates_kg_s"], b["component_mass_rates_kg_s"], strict=True):
                errors["component_rate_scaled"] = max(
                    errors["component_rate_scaled"], abs(x - y) / max(abs(y), 1e-8)
                )
    if set(budgets) != set(errors):
        raise ValueError("Completion budgets required for each metric")
    return {
        "accepted": whole["accepted"] and all(errors[k] <= budgets[k] for k in errors),
        "errors": errors,
        "budgets": budgets,
        "report_dates": len(dates),
        "scope": "Actual individual completion pressures/fluxes/valves at common report dates; isolated pressure gauges excluded.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--whole-budgets", type=Path, required=True)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.candidate,
        args.reference,
        json.loads(args.whole_budgets.read_text()),
        json.loads(args.budgets.read_text()),
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)
