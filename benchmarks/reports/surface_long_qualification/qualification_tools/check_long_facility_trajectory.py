"""Compare actual well rates/BHP and cumulative mass at coarse report endpoints.

Both bundles must independently pass full facility accounting first. This is
step refinement, not a proof that either trajectory is exact or calibrated.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

from check_fim_facility_forecast import audit
from compare_fim_facility_forecast import physical_config


def compare(coarse, fine, *, allow_numerical_policy=False):
    bundles = []
    for path in (coarse, fine):
        with gzip.open(path, "rt") as handle:
            d = json.load(handle)
        audit(
            d["config"],
            d["result"],
            expected_schedule=d["schedule"],
            expected_days=d["requested_days"],
        )
        bundles.append(d)
    a, b = bundles
    if (
        physical_config(a, allow_numerical_policy=allow_numerical_policy)
        != physical_config(b, allow_numerical_policy=allow_numerical_policy)
        or a["schedule"] != b["schedule"]
        or a["requested_days"] != b["requested_days"]
    ):
        raise ValueError("Identical model, controls and horizon required")

    def series(bundle):
        result = {}
        amount = [0.0] * 3
        for step in bundle["result"]["accepted_substeps"]:
            for i, v in enumerate(step["withdrawn_component_mass_kg"]):
                amount[i] = math.fsum((amount[i], v))
            key = round(step["time_days"] * 86400, 6)
            if key in result:
                raise ValueError("Repeated accepted time")
            result[key] = (step, amount.copy())
        return result

    ca, cb = series(a), series(b)
    errors = {"bhp_abs_pa": 0.0, "component_rate_scaled": 0.0, "cumulative_mass_relative": 0.0}
    mode_mismatches = 0
    matched = []
    reports = a["result"]["interval_timings"]
    if not reports:
        raise ValueError("Coarse report endpoint inventory required")
    clocks = [round(r["time_days"] * 86400, 6) for r in reports]
    if len(set(clocks)) != len(clocks):
        raise ValueError("Repeated coarse report endpoint")
    for clock in clocks:
        if clock not in ca:
            raise ValueError("Coarse report lacks an accepted state")
        sa, ma = ca[clock]
        if clock not in cb:
            raise ValueError("Reference lacks a coarse accepted endpoint")
        sb, mb = cb[clock]
        if sa["facility_controls"] != sb["facility_controls"]:
            raise ValueError("Different controls at common endpoint")
        wa, wb = sa["surface"]["wells"], sb["surface"]["wells"]
        if set(wa) != set(wb):
            raise ValueError("Different well registry")
        for name, x in wa.items():
            y = wb[name]
            values = [
                x["bhp_pa"],
                y["bhp_pa"],
                *x["component_mass_rates_kg_s"],
                *y["component_mass_rates_kg_s"],
            ]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite accepted well state")
            errors["bhp_abs_pa"] = max(errors["bhp_abs_pa"], abs(x["bhp_pa"] - y["bhp_pa"]))
            for q, r in zip(
                x["component_mass_rates_kg_s"], y["component_mass_rates_kg_s"], strict=True
            ):
                errors["component_rate_scaled"] = max(
                    errors["component_rate_scaled"], abs(q - r) / max(abs(r), 1e-8)
                )
            mode_mismatches += x["mode"] != y["mode"]
        errors["cumulative_mass_relative"] = max(
            errors["cumulative_mass_relative"],
            *(abs(x - y) / max(abs(y), 1.0) for x, y in zip(ma, mb)),
        )
        matched.append(clock / 86400)
    budgets = {
        "bhp_abs_pa": 100.0,
        "component_rate_scaled": 0.002,
        "cumulative_mass_relative": 0.002,
    }
    return {
        "scope": __doc__,
        "accepted": mode_mismatches == 0 and all(errors[k] <= budgets[k] for k in errors),
        "matched_endpoints_days": matched,
        "mode_mismatches": mode_mismatches,
        "errors": errors,
        "budgets": budgets,
        "component_rate_floor_kg_s": 1e-8,
        "allow_numerical_policy": allow_numerical_policy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-numerical-policy", action="store_true")
    args = parser.parse_args()
    result = compare(args.coarse, args.fine, allow_numerical_policy=args.allow_numerical_policy)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
