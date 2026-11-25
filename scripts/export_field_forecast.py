"""Audited stock-condition production and equipment KPIs from a completed forecast.

Uses every accepted implicit stage. Never reconstructs histories from thinned
viewer frames, or invents intermediate reservoir fields/injection rates.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from check_facility_refinement_pair import completed
from check_fim_facility_forecast import audit

COMPONENTS = ("oil", "water", "gas")


def production_series(config, result):
    densities = config["simulation"]["coupled_facility"]["facility"]["stock_densities"]
    rho = [densities[name + "_kg_m3"] for name in COMPONENTS]
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in rho):
        raise ValueError("Positive finite stock densities required")
    if result["component_order"] != list(COMPONENTS):
        raise ValueError("Unexpected stock-component ordering")
    amounts, energy = [0.0] * 3, 0.0
    rows = []
    for step in result["accepted_substeps"]:
        dt = step["dt_s"]
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite accepted duration required")
        rates = []
        for j, name in enumerate(COMPONENTS):
            mass = step["withdrawn_component_mass_kg"][j]
            if not math.isfinite(mass) or mass < 0:
                raise ValueError("Production-only exporter requires nonnegative finite withdrawals")
            amounts[j] = math.fsum((amounts[j], mass))
            rates.append(mass / rho[j] / dt * 86400)
        coupling = step["facility_coupling"]
        measurement = coupling["facility"]["measurements"]["facility_drive_power_total_w"]
        if measurement["unit"] != "W" or not math.isfinite(measurement["value"]):
            raise ValueError("Finite equipment drive power in watts required")
        power = measurement["value"]
        energy = math.fsum((energy, power * dt / 3_600_000))
        bhp = [well["bhp_pa"] for well in step["surface"]["wells"].values()]
        row = {
            "time_days": step["time_days"],
            "dt_days": dt / 86400,
            "water_cut_stock": rates[1] / (rates[0] + rates[1]) if rates[0] + rates[1] else None,
            "gas_oil_ratio_stock_m3_m3": rates[2] / rates[0] if rates[0] else None,
            "bhp_min_pa": min(bhp),
            "bhp_max_pa": max(bhp),
            "production_cap_fraction": coupling["fraction"],
            "equipment_drive_power_w": power,
            "cumulative_equipment_drive_energy_kwh": energy,
        }
        row.update({name + "_rate_stock_m3_day": rates[j] for j, name in enumerate(COMPONENTS)})
        row.update(
            {
                name + "_cumulative_stock_m3": amounts[j] / rho[j]
                for j, name in enumerate(COMPONENTS)
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError("No accepted production stages")
    return rows


def export(directory, output):
    directory, output = Path(directory), Path(output)
    metadata = json.loads((directory / "summary.json").read_text())
    summary, bundle, trajectory = completed(directory, metadata["audit"]["elapsed_days"])
    config, result = bundle["config"], bundle["result"]
    certificate = audit(
        config, result, expected_schedule=bundle["schedule"], expected_days=bundle["requested_days"]
    )
    rows = production_series(config, result)
    report = {
        "schema": "field_production_kpis_v1",
        "scope": "Production-only black-oil and quasi-steady facility; no injection network, dynamic vessel inventories, thermal reservoir or experimental calibration",
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "solve_wall_s": summary["solve_wall_s"],
        "elapsed_days": result["elapsed_days"],
        "accepted_substeps": len(rows),
        "operating_events": result["facility_control_events"],
        "volume_basis": {
            "stock_densities": config["simulation"]["coupled_facility"]["facility"][
                "stock_densities"
            ],
            "definition": "Stock-component mass divided by configured stock density, not in-situ phase volume",
        },
        "rate_basis": "Accepted-interval average from actual component withdrawal integrals",
        "energy_basis": "Quasi-steady pump/compressor shaft power integrated over accepted implicit stages; not a whole-field thermal energy balance",
        "undefined_ratios": "null in JSON, empty in CSV when denominator is zero",
        "reservoir_fields": "Final snapshot only; no intermediate pressure/saturation fields inferred",
        "final_reservoir_pressure_range_pa": [
            min(result["final_pressure_pa"]),
            max(result["final_pressure_pa"]),
        ],
        "final": rows[-1],
        "audit": certificate,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "production.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.forecast, args.output), indent=2, allow_nan=False))
