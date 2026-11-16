"""Compare completed operational forecasts at the final common report endpoint.

Budgets must be supplied before inspecting the reference result. This is a
step-refinement check, not a proof that the finer trajectory is exact.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

from check_fim_facility_forecast import audit


def compare(coarse_path, fine_path, budgets):
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in budgets.values()):
        raise ValueError("Finite nonnegative refinement budgets required")
    bundles = []
    for path in (coarse_path, fine_path):
        with gzip.open(path, "rt") as handle:
            b = json.load(handle)
        audit(
            b["config"],
            b["result"],
            expected_schedule=b["schedule"],
            expected_days=b["requested_days"],
        )
        bundles.append(b)
    a, b = bundles
    if (
        a["config"] != b["config"]
        or a["schedule"] != b["schedule"]
        or a["requested_days"] != b["requested_days"]
    ):
        raise ValueError("Refinement requires the same model, controls and horizon")
    errors = {}
    for key in ("final_pressure_pa", "final_s_w", "final_s_g", "final_rs"):
        errors[key] = max(
            abs(x - y) for x, y in zip(a["result"][key], b["result"][key], strict=True)
        )
    errors["withdrawal_relative"] = max(
        abs(x - y) / max(abs(y), 1.0)
        for x, y in zip(
            a["result"]["withdrawn_component_mass_kg"],
            b["result"]["withdrawn_component_mass_kg"],
            strict=True,
        )
    )
    if set(errors) != set(budgets):
        raise ValueError("Explicit budgets for every compared quantity required")
    return {
        "accepted": all(errors[k] <= budgets[k] for k in errors),
        "errors": errors,
        "budgets": budgets,
        "coarse": str(coarse_path),
        "fine": str(fine_path),
        "scope": __doc__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.coarse, args.fine, json.loads(args.budgets.read_text()))
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
