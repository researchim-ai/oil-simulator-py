"""Audit accepted forecast material and controls without running either solver.

This is an accounting/constraint audit, not independent validation of the PVT
or homogeneous gathering closure. The production FIM CNV/MB remain required.
"""

import argparse
import json
import math
from pathlib import Path

COMPONENTS = ("oil", "water", "gas")


def audit(
    config, report, *, expected_schedule=None, expected_days=None, mass_atol_kg=0.01, mass_rtol=1e-8
):
    def close(a, b, *, atol=1e-9, rtol=1e-10):
        if (
            not math.isfinite(a)
            or not math.isfinite(b)
            or not math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
        ):
            raise ValueError(f"Forecast audit mismatch: {a} != {b}")

    density = config["simulation"]["coupled_facility"]["facility"]["stock_densities"]
    rho = [density[c + "_kg_m3"] for c in COMPONENTS]
    steps = report["accepted_substeps"]
    if not steps:
        raise ValueError("No accepted coupled steps")
    clock = 0.0
    withdrawn = [0.0] * 3
    max_surface_residual = 0.0
    evaluations = 0
    events = report.get("facility_control_events", [])
    if expected_days is not None:
        close(report["elapsed_days"], expected_days, atol=1e-12, rtol=0)
    if (
        expected_schedule is not None
        and [{"time_days": e["time_days"], "controls": e["controls"]} for e in events]
        != expected_schedule
    ):
        raise ValueError("Requested operating schedule was not completed")
    previous_event = -1.0
    for event in events:
        if not previous_event < event["time_days"] < report["elapsed_days"]:
            raise ValueError("Invalid forecast event order")
        close(event["applied_time_days"], event["time_days"], atol=1e-12, rtol=0)
        previous_event = event["time_days"]
    for step in steps:
        dt = step["dt_s"]
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid accepted duration")
        end = step["time_days"] * 86400
        close(end, clock + dt, atol=1e-7, rtol=1e-12)
        controls = {}
        for event in events:
            when = event["time_days"] * 86400
            if clock + 8 * math.ulp(clock) < when < end - 8 * math.ulp(end):
                raise ValueError("Accepted step crossed an operating event")
            if when <= clock + 8 * math.ulp(clock):
                controls = event["controls"]
        if step["facility_controls"] != controls:
            raise ValueError("Wrong controls on accepted reservoir interval")
        coupling = step["facility_coupling"]
        plant = coupling["facility"]
        if (
            not plant["feasible"]
            or plant["violations"]
            or not all(x["passed"] for x in plant["limits"])
        ):
            raise ValueError("Infeasible plant was accepted")
        close(coupling["dt_s"], dt, atol=1e-7)
        if not 0 < coupling["fraction"] <= 1:
            raise ValueError("Invalid rate-cap fraction")
        for applied, well in zip(
            coupling["applied_rate_caps_m3_day"], config["wells"], strict=True
        ):
            close(applied, well["control_value"] * coupling["fraction"])
        for limit in config["simulation"]["coupled_facility"]["facility"].get("limits", []):
            measurement = plant["measurements"][limit["metric"]]
            value = measurement["value"]
            if (
                measurement["unit"] != limit["unit"]
                or not math.isfinite(value)
                or value < limit.get("minimum", -math.inf)
                or value > limit.get("maximum", math.inf)
            ):
                raise ValueError("Configured equipment constraint violated")
        evaluations += coupling["evaluations"]
        wells = list(step["surface"]["wells"].values())
        for i, component in enumerate(COMPONENTS):
            well_rate = math.fsum(w["component_mass_rates_kg_s"][i] for w in wells)
            feed_rate = plant["feed"][component + "_m3_s"] * rho[i]
            export_rate = math.fsum(
                s[component + "_m3_s"] * rho[i] for s in plant["products"].values()
            )
            close(feed_rate, well_rate, atol=1e-12)
            close(export_rate, well_rate, atol=1e-12)
            max_surface_residual = max(max_surface_residual, abs(export_rate - well_rate))
            amount = well_rate * dt
            close(step["withdrawn_component_mass_kg"][i], amount)
            withdrawn[i] = math.fsum((withdrawn[i], amount))
            account = coupling["material_account"]
            close(account["produced_kg"][component], withdrawn[i])
            close(account["elapsed_s"], end, atol=1e-7)
        clock = end
    close(clock, report["elapsed_days"] * 86400, atol=1e-7)
    residuals = []
    for i in range(3):
        close(report["withdrawn_component_mass_kg"][i], withdrawn[i])
        residual = (
            report["initial_component_mass_kg"][i]
            - report["final_component_mass_kg"][i]
            - withdrawn[i]
        )
        close(residual, report["mass_balance_error_kg"][i], atol=1e-8)
        budget = mass_atol_kg + mass_rtol * max(abs(withdrawn[i]), 1.0)
        if abs(residual) > budget:
            raise ValueError("Reservoir-to-export mass budget exceeded")
        residuals.append(residual)
    return {
        "accepted": True,
        "checked_substeps": len(steps),
        "elapsed_days": clock / 86400,
        "minimum_step_s": min(s["dt_s"] for s in steps),
        "maximum_step_s": max(s["dt_s"] for s in steps),
        "capacity_evaluations": evaluations,
        "operating_events": len(events),
        "mass_balance_error_kg": residuals,
        "maximum_surface_component_residual_kg_s": max_surface_residual,
        "mass_atol_kg": mass_atol_kg,
        "mass_rtol": mass_rtol,
        "scope": __doc__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.bundle.suffix == ".gz":
        import gzip

        with gzip.open(args.bundle, "rt") as handle:
            bundle = json.load(handle)
    else:
        bundle = json.loads(args.bundle.read_text())
    print(
        json.dumps(
            audit(
                bundle["config"],
                bundle["result"],
                expected_schedule=bundle["schedule"],
                expected_days=bundle["requested_days"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
