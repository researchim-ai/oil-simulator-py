"""Strict source-guarded regression for capacity-search AMG hierarchy reuse."""

import argparse
import copy
import json
from pathlib import Path

from check_facility_capacity_search import FIELD_BUDGETS, WELL_BUDGETS
from check_facility_refinement_pair import completed
from check_long_facility_trajectory import compare as compare_wells
from compare_fim_facility_forecast import compare as compare_fields


def check_policy_pair(reference, candidate):
    configs = [copy.deepcopy(c) for c in (reference, candidate)]
    flags = [
        c["simulation"]["coupled_facility"]["control"].pop("reuse_capacity_amg", False)
        for c in configs
    ]
    if (
        any(type(x) is not bool for x in flags)
        or flags != [False, True]
        or configs[0] != configs[1]
    ):
        raise ValueError("Only reuse_capacity_amg false/true may differ")


def compare(reference, candidate, days):
    sa, a, pa = completed(reference, days)
    sb, b, pb = completed(candidate, days)
    check_policy_pair(a["config"], b["config"])
    python_sources = lambda s: {k: v for k, v in s["source_sha256"].items() if k.endswith(".py")}
    if not python_sources(sa) or python_sources(sa) != python_sources(sb):
        raise ValueError("Identical executed Python sources required")
    pvt = a["config"]["fluid"]["pvt_path"]
    if pvt not in sa["source_sha256"] or sa["source_sha256"][pvt] != sb["source_sha256"].get(pvt):
        raise ValueError("Identical guarded PVT required")
    if a["schedule"] != b["schedule"] or sa["device"] != sb["device"]:
        raise ValueError("Identical schedule and device required")
    if sa.get("cuda_memory_fraction") != sb.get("cuda_memory_fraction"):
        raise ValueError("Identical allocator limits required")
    for key in (
        "torch_version",
        "torch_threads",
        "device_name",
        "gathering_backend",
        "hydraulic_backend",
    ):
        if sa["execution"].get(key) != sb["execution"].get(key):
            raise ValueError("Identical execution backend required")
    protocol = [json.loads((Path(p) / "protocol.json").read_text()) for p in (reference, candidate)]
    if protocol[0].get("cuda_visible_devices") != protocol[1].get("cuda_visible_devices"):
        raise ValueError("Identical selected CUDA UUID required")
    fields = compare_fields(pa, pb, FIELD_BUDGETS, allow_numerical_policy=True)
    wells = compare_wells(pa, pb, allow_numerical_policy=True)
    wells["budgets"] = WELL_BUDGETS
    wells["accepted"] = wells["mode_mismatches"] == 0 and all(
        wells["errors"][k] <= v for k, v in WELL_BUDGETS.items()
    )
    work = [x["result"]["temporal_work"] for x in (a, b)]
    amg = [x["capacity_amg_work"] for x in work]
    return {
        "accepted": fields["accepted"] and wells["accepted"],
        "required_days": days,
        "fewer_amg_builds": amg[1]["builds"] < amg[0]["builds"],
        "same_fim_work": work[0]["reservoir_evaluations"] == work[1]["reservoir_evaluations"],
        "fields": fields,
        "wells": wells,
        "reference_work": work[0],
        "candidate_work": work[1],
        "reference_solve_wall_s": sa["solve_wall_s"],
        "candidate_solve_wall_s": sb["solve_wall_s"],
        "observed_ratio": sa["solve_wall_s"] / sb["solve_wall_s"],
        "timing_scope": "Shared host/device, not isolated speedup or a longer-horizon prediction.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--days", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = compare(args.reference, args.candidate, args.days)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] and result["fewer_amg_builds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
