"""Export every accepted waterflood stage; no thinned-frame integration."""

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
        or hashlib.sha256(path.read_bytes()).hexdigest() != summary["trajectory_sha256"]
    ):
        raise ValueError("Completed intact guarded study required")
    with gzip.open(path, "rt") as stream:
        bundle = json.load(stream)
    config, result = bundle["config"], bundle["result"]
    audit(config, result)
    rho = config["simulation"]["coupled_waterflood"]["facility"]["stock_densities"]
    energy = 0.0
    amounts = {name: 0.0 for name in ("produced", "injected", "recycled", "makeup", "disposed")}
    rows = []
    for stage in result["accepted_substeps"]:
        report = stage["coupling"]
        dt = report["dt_s"]
        energy += report["total_drive_power_w"] * dt / 3_600_000
        row = {
            "time_days": stage["time_days"],
            "dt_days": dt / 86400,
            "production_fraction": report["production_fraction"],
            "injection_fraction": report["injection_fraction"],
            "injection_enabled": report["operator_controls"]["injection_enabled"],
            "makeup_limit_kg_s": report["operator_controls"]["makeup_water_max_kg_s"],
            "total_shaft_power_w": report["total_drive_power_w"],
            "injection_shaft_power_w": report["injection_network"]["shaft_power_w"],
            "cumulative_shaft_energy_kwh": energy,
        }
        for name, rate in report["water_rates_kg_s"].items():
            amounts[name] += rate * dt
            row["water_" + name + "_kg_s"] = rate
            row["water_" + name + "_cumulative_stock_m3"] = amounts[name] / rho["water_kg_m3"]
        for name, well in stage["surface"]["wells"].items():
            row[name + "_mode"] = well["mode"]
            row[name + "_bhp_pa"] = (
                None if well.get("bhp_status") == "isolated_gauge" else well["bhp_pa"]
            )
        rows.append(row)
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "rows": len(rows),
        "water_cumulative_stock_m3": {
            key: value / rho["water_kg_m3"] for key, value in amounts.items()
        },
        "shaft_energy_kwh": energy,
        "volume_basis": "Configured stock water density; not a thermal energy balance or an experimental reference",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export(args.study, args.output), indent=2))
