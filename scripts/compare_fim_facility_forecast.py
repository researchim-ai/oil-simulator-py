"""Compare completed operational forecasts at the final common report endpoint.

Budgets must be supplied before inspecting the reference result. This is a
step-refinement check, not a proof that the finer trajectory is exact.
"""

import argparse
import copy
import gzip
import json
import math
from pathlib import Path

from check_fim_facility_forecast import audit


def physical_config(bundle, *, allow_numerical_policy=False):
    c = copy.deepcopy(bundle["config"])
    if allow_numerical_policy:
        for key in ("max_time_step_days", "initial_time_step_days"):
            value = c["simulation"].pop(key, None)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"Invalid numerical time-step policy: {key}")
        facility = c["simulation"]["coupled_facility"]
        facility.pop("time_control", None)
        facility.pop("temporal_control", None)
        if facility.get("control") is not None:
            initial_cache = facility['control'].pop('reuse_initial_cell_state', False)
            initial_budget = facility['control'].pop('initial_cell_state_cache_bytes', 64 * 1024**2)
            if type(initial_cache) is not bool or type(initial_budget) is not int or initial_budget <= 0:
                raise ValueError('Invalid initial cell state cache policy')
            reuse = facility['control'].pop('reuse_inactive_caps', False)
            if type(reuse) is not bool:
                raise ValueError('Invalid inactive-cap reuse policy')
            facility["control"].pop("search_method", None)
            commit = facility["control"].pop("commit_method", "replay")
            if commit not in ("replay", "state_restore"):
                raise ValueError("Unknown capacity commit method")
        network = c["simulation"]["physical_surface_network"]
        initialization = network.pop("bhp_initialization", "previous")
        if initialization not in ("previous", "rate_network"):
            raise ValueError("Unknown numerical BHP initialization")
        execution = network.pop("network_backend", "cpu_reference")
        if execution not in ("cpu_reference", "torch_device"):
            raise ValueError("Unknown numerical gathering backend")
        sensitivity = network.pop("sensitivity_backend", "dense")
        if sensitivity not in ("dense", "tree_operator"):
            raise ValueError("Unknown numerical sensitivity backend")
        well_backend = network.pop("well_block_backend", "dense")
        if well_backend not in ("dense", "tree_structured"):
            raise ValueError("Unknown well block backend")
        memory = network.pop("structured_memory_budget_bytes", 256 * 1024**2)
        if type(memory) is not int or memory <= 0:
            raise ValueError("Invalid structured memory budget")
        hydraulics = network["network"]
        backend = hydraulics.pop("hydraulic_backend", "scalar")
        if backend not in ("scalar", "torch_batch"):
            raise ValueError("Unknown pipe hydraulic backend")
        size = hydraulics.pop("pipe_batch_size", 128)
        if type(size) is not int or not 1 <= size <= 256:
            raise ValueError("Invalid pipe batch size")
    return c


def compare(coarse_path, fine_path, budgets, *, allow_numerical_policy=False):
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
        physical_config(a, allow_numerical_policy=allow_numerical_policy)
        != physical_config(b, allow_numerical_policy=allow_numerical_policy)
        or a["schedule"] != b["schedule"]
        or a["requested_days"] != b["requested_days"]
    ):
        raise ValueError("Refinement requires the same model, controls and horizon")
    errors = {}
    for key in ("final_pressure_pa", "final_s_w", "final_s_g", "final_rs"):
        fields = [x["result"][key] for x in (a, b)]
        if any(
            not values or any(isinstance(v, bool) or not math.isfinite(v) for v in values)
            for values in fields
        ):
            raise ValueError(f"Nonempty finite final field required: {key}")
        errors[key] = max(
            abs(x - y) for x, y in zip(*fields, strict=True)
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
        "allowed_numerical_policy_differences": [
            "simulation.max_time_step_days",
            "simulation.initial_time_step_days",
            "coupled_facility.time_control",
            "coupled_facility.temporal_control",
            "coupled_facility.control.search_method",
            "coupled_facility.control.commit_method",
            "coupled_facility.control.reuse_inactive_caps",
            "coupled_facility.control.reuse_initial_cell_state",
            "coupled_facility.control.initial_cell_state_cache_bytes",
            "physical_surface_network.sensitivity_backend",
            "physical_surface_network.network_backend",
            "physical_surface_network.bhp_initialization",
            "physical_surface_network.well_block_backend",
            "physical_surface_network.structured_memory_budget_bytes",
            "physical_surface_network.network.hydraulic_backend",
            "physical_surface_network.network.pipe_batch_size",
        ]
        if allow_numerical_policy
        else [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-numerical-policy", action="store_true")
    args = parser.parse_args()
    result = compare(
        args.coarse,
        args.fine,
        json.loads(args.budgets.read_text()),
        allow_numerical_policy=args.allow_numerical_policy,
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
