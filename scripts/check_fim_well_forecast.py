"""Accepted producer-only FIM -> gathering material/operating audit.

No solver invocation or rescaling of requested production. This is accounting
and constraint checking, not independent PVT/correlation validation.
"""

import argparse
import gzip
import json
import math
from pathlib import Path


def audit(config, result, *, expected_days, mass_atol_kg=0.01, mass_rtol=1e-8):
    def close(a, b, atol=1e-9, rtol=1e-10):
        if (
            not math.isfinite(a)
            or not math.isfinite(b)
            or not math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
        ):
            raise ValueError(f"Well forecast mismatch {a} != {b}")

    wells = {w["name"]: w for w in config["wells"]}
    if not wells or any(
        w["type"] != "producer" or w["control_type"] != "rate" or w["control_value"] <= 0
        for w in wells.values()
    ):
        raise ValueError("This audit requires positive producer rate caps")
    spec = config["simulation"]["physical_surface_network"]
    densities = spec["network"]["stock_densities"]
    backend = spec.get("well_block_backend", "dense")
    clock = 0.0
    total = [0.0] * 3
    modes = {}
    steps = result["accepted_substeps"]
    if not steps:
        raise ValueError("No accepted steps")
    for step in steps:
        dt = step["dt_s"]
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Positive accepted duration required")
        close(step["time_days"] * 86400, clock + dt, atol=1e-7)
        physical = step["surface"]
        if set(physical["wells"]) != set(wells):
            raise ValueError("Exact accepted well registry required")
        execution = physical["execution"]
        if execution.get("well_block_backend", "dense") != backend:
            raise ValueError("Wrong well block backend")
        if backend == "tree_structured" and (
            execution["dense_B_stored"] or execution["dense_D_stored"]
        ):
            raise ValueError("Structured assembly stored dense well blocks")
        for name, row in physical["wells"].items():
            mode = row["mode"]
            modes[mode] = modes.get(mode, 0) + 1
            if mode not in ("rate", "surface", "bhp_limit"):
                raise ValueError("Unexpected producer active set")
            cap = wells[name]["control_value"] / 86400
            tolerance = config["simulation"].get("newton_tol_well_rate", 1e-6)
            q = row["standard_control_rate_m3_s"]
            if not math.isfinite(q) or q < 0 or q > cap * (1 + tolerance) + 1e-12:
                raise ValueError("Accepted producer exceeds its rate cap")
            if mode == "surface":
                close(
                    row["bhp_pa"],
                    row["network_reference_pressure_pa"],
                    atol=config["simulation"].get("newton_tol_well_bhp", 1e3),
                    rtol=0,
                )
        network = physical["network"]
        for i, component in enumerate(("oil", "water", "gas")):
            rates = [r["component_mass_rates_kg_s"][i] for r in physical["wells"].values()]
            if any(not math.isfinite(q) or q < 0 for q in rates):
                raise ValueError("Invalid producing component rate")
            rate = math.fsum(rates)
            close(rate, network["receiving_component_mass_rates_kg_s"][component], atol=1e-12)
            close(
                rate,
                network["receiving_stream"][component + "_m3_s"] * densities[component + "_kg_m3"],
                atol=1e-12,
            )
            amount = rate * dt
            close(step["withdrawn_component_mass_kg"][i], amount)
            total[i] = math.fsum((total[i], amount))
        clock = step["time_days"] * 86400
    close(clock, expected_days * 86400, atol=1e-7)
    close(result["elapsed_days"], expected_days, atol=1e-12, rtol=0)
    residual = []
    for i in range(3):
        close(total[i], result["withdrawn_component_mass_kg"][i])
        error = (
            result["initial_component_mass_kg"][i] - result["final_component_mass_kg"][i] - total[i]
        )
        close(error, result["mass_balance_error_kg"][i], atol=1e-7)
        if abs(error) > mass_atol_kg + mass_rtol * max(abs(total[i]), 1):
            raise ValueError("Producer-to-export mass balance exceeds budget")
        residual.append(error)
    return {
        "accepted": True,
        "wells": len(wells),
        "accepted_steps": len(steps),
        "elapsed_days": expected_days,
        "mode_samples": modes,
        "component_mass_residual_kg": residual,
        "mass_atol_kg": mass_atol_kg,
        "mass_rtol": mass_rtol,
        "well_block_backend": backend,
        "scope": __doc__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    args = parser.parse_args()
    with gzip.open(args.trajectory, "rt") as f:
        b = json.load(f)
    print(json.dumps(audit(b["config"], b["result"], expected_days=b["requested_days"]), indent=2))


if __name__ == "__main__":
    main()
