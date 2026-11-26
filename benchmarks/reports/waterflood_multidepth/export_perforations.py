"""Export accepted completion histories from a guarded waterflood study."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from check_fim_waterflood import audit


def export(study, output):
    summary = json.loads((study / "summary.json").read_text())
    path = study / "trajectory.json.gz"
    if (
        summary["status"] != "complete"
        or not summary["sources_unchanged"]
        or (study / "failure.json").exists()
        or hashlib.sha256(path.read_bytes()).hexdigest() != summary["trajectory_sha256"]
    ):
        raise ValueError("Completed intact guarded study required")
    with gzip.open(path, "rt") as stream:
        bundle = json.load(stream)
    audit(bundle["config"], bundle["result"])
    rows = []
    for step in bundle["result"]["accepted_substeps"]:
        network = step["coupling"]["injection_network"]
        for index, perf in enumerate(network["perforations"]):
            well = network["wells"][perf["well"]]
            physical = well["bhp_status"] != "isolated_gauge"
            row = {
                "time_days": step["time_days"],
                "dt_s": step["coupling"]["dt_s"],
                "connection_index": index,
                "well": perf["well"],
                "mode": well["mode"],
                "bhp_status": well["bhp_status"],
                "reference_depth_m": well["reference_depth_m"],
                "depth_m": perf["depth_m"],
                "hydrostatic_head_pa": perf["hydrostatic_head_pa"],
                "bhp_pa": well["bhp_pa"] if physical else None,
                "well_pressure_pa": perf["well_pressure_pa"] if physical else None,
                "reservoir_pressure_pa": perf["reservoir_pressure_pa"],
                "valve_open": perf["valve_open"],
            }
            for phase, mass in zip(
                ["oil", "water", "gas"], perf["component_mass_rates_kg_s"], strict=True
            ):
                row[phase + "_withdrawal_mass_rate_kg_s"] = mass
            rows.append(row)
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "sign": "negative withdrawal is injection", "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.study, args.output), indent=2))
