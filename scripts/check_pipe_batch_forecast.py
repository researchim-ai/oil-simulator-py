"""Compare like-for-like producer forecasts, changing only pipe numerical policy.

Both trajectories must independently pass material/operating accounting. This
is scalar-reference equivalence, not independent experimental validation.
"""

import argparse
import copy
import gzip
import json
import math
from pathlib import Path

from check_fim_well_forecast import audit


def compare(scalar, batch, budgets):
    keys = {"final_pressure_pa", "final_s_w", "final_s_g", "final_rs", "withdrawal_relative"}
    if set(budgets) != keys or any(
        type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in budgets.values()
    ):
        raise ValueError("Explicit finite nonnegative budgets required for all quantities")
    bundles = []
    policies = []
    audits = []
    for path in (scalar, batch):
        with gzip.open(path, "rt") as handle:
            bundle = json.load(handle)
        audits.append(
            audit(bundle["config"], bundle["result"], expected_days=bundle["requested_days"])
        )
        config = copy.deepcopy(bundle["config"])
        network = config["simulation"]["physical_surface_network"]["network"]
        policies.append(network.pop("hydraulic_backend", "scalar"))
        size = network.pop("pipe_batch_size", 128)
        if type(size) is not int or not 1 <= size <= 256:
            raise ValueError("Invalid pipe batch size")
        bundles.append((bundle, config))
    (a, ca), (b, cb) = bundles
    if (
        ca != cb
        or a["requested_days"] != b["requested_days"]
        or policies != ["scalar", "torch_batch"]
    ):
        raise ValueError("Same physical model/horizon and scalar/batch policies required")
    for bundle in (a, b):
        for key in keys - {"withdrawal_relative"}:
            values = bundle["result"][key]
            if not values or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in values
            ):
                raise ValueError("Finite nonempty final reservoir fields required")
    errors = {
        key: max(abs(x - y) for x, y in zip(a["result"][key], b["result"][key], strict=True))
        for key in keys
        if key != "withdrawal_relative"
    }
    errors["withdrawal_relative"] = max(
        abs(x - y) / max(abs(x), 1.0)
        for x, y in zip(
            a["result"]["withdrawn_component_mass_kg"],
            b["result"]["withdrawn_component_mass_kg"],
            strict=True,
        )
    )
    execution = {"torch_batch_pipes": 0, "scalar_retry_pipes": 0, "scalar_pipes": 0}
    for step in b["result"]["accepted_substeps"]:
        report = step["surface"]["network"]["hydraulic_execution"]
        if report["backend"] != "torch_batch":
            raise ValueError("Missing actual batch network execution")
        if any(type(report[k]) is not int or report[k] < 0 for k in execution):
            raise ValueError("Nonnegative integer execution counts required")
        network = b["config"]["simulation"]["physical_surface_network"]["network"]
        if sum(report[k] for k in execution) != len(network["pipes"]) + len(
            network.get("valves", [])
        ):
            raise ValueError("Every network edge must have one execution count")
        for key in execution:
            execution[key] += report[key]
    if not execution["torch_batch_pipes"]:
        raise ValueError("No actual pipe batch executed")
    return {
        "accepted": all(errors[k] <= budgets[k] for k in keys),
        "scope": __doc__,
        "errors": errors,
        "budgets": budgets,
        "audits": audits,
        "accepted_step_pipe_counts": execution,
        "scalar": str(scalar),
        "batch": str(batch),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scalar", type=Path)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--budgets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.scalar, args.batch, json.loads(args.budgets.read_text()))
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
