"""Release gate for completed, source-guarded time refinement on a declared horizon.

Audits material and controls, end fields and well histories at common reports.
It does not establish experimental calibration or global error against truth.
"""

import argparse
import copy
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_long_facility_trajectory import compare as compare_wells
from compare_fim_facility_forecast import compare as compare_fields

from simulator.facility_temporal_control import FacilityTemporalControl
from simulator.facility_time_control import FacilityTimeControl


def completed(directory, required_days):
    if isinstance(required_days, bool) or not math.isfinite(required_days) or required_days <= 0:
        raise ValueError("Positive finite qualification horizon required")
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if (directory / "failure.json").exists():
        raise ValueError("Failed forecast cannot qualify")
    if (
        summary.get("status") != "complete"
        or summary.get("sources_unchanged") is not True
        or json.loads((directory / "progress.json").read_text()).get("status") != "complete"
    ):
        raise ValueError("Completed source-guarded forecast required")
    path = directory / "trajectory.json.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary["trajectory_sha256"]:
        raise ValueError("Trajectory checksum mismatch")
    with gzip.open(path, "rt") as f:
        bundle = json.load(f)
    if (
        bundle["requested_days"] != required_days
        or bundle["result"]["elapsed_days"] != required_days
    ):
        raise ValueError("Forecast does not cover the required horizon")
    return summary, bundle, path


def stricter(candidate, reference):
    a, b = [x["config"]["simulation"] for x in (candidate, reference)]
    ta, tb = [x["coupled_facility"].get("temporal_control") for x in (a, b)]
    if ta is not None or tb is not None:
        if ta is None or tb is None:
            raise ValueError("Use the same temporal control family")
        a, b = FacilityTemporalControl(**ta), FacilityTemporalControl(**tb)
        if a.withdrawal_error_scale != b.withdrawal_error_scale:
            raise ValueError("Reference must preserve the withdrawal error scaling")
        if (
            a.safety_factor != b.safety_factor
            or a.maximum_growth != b.maximum_growth
            or b.startup_step_days > a.startup_step_days
        ):
            raise ValueError("Reference must preserve growth policy and not increase startup step")
        keys = (
            "pressure_tolerance_pa",
            "saturation_tolerance",
            "solution_ratio_tolerance",
            "withdrawal_relative_tolerance",
            "withdrawal_absolute_tolerance_kg",
        )
        wa, wb = a.well_bhp_tolerance_pa, b.well_bhp_tolerance_pa
        if wa is not None and (wb is None or wb > wa):
            raise ValueError("Reference must not weaken temporal BHP control")
        bhp_stricter = wb is not None and (wa is None or wb < wa)
        if not all(getattr(b, k) <= getattr(a, k) for k in keys) or (
            not any(getattr(b, k) < getattr(a, k) for k in keys) and not bhp_stricter
        ):
            raise ValueError("Reference temporal tolerances must be stricter")
        return "local_BE_defect_tolerances"
    ra, rb = [x["coupled_facility"].get("time_control") for x in (a, b)]
    if ra is not None or rb is not None:
        if ra is None or rb is None:
            raise ValueError("Use the same endpoint-rate control family")
        pa, pb = FacilityTimeControl(**ra), FacilityTimeControl(**rb)
        if (pa.safety_factor != pb.safety_factor or pa.maximum_growth != pb.maximum_growth
                or pb.startup_step_days > pa.startup_step_days):
            raise ValueError("Reference must preserve rate growth policy and not increase startup step")
        keys = ("rate_relative_tolerance", "rate_absolute_tolerance_kg_s")
        if not all(getattr(pb, k) <= getattr(pa, k) for k in keys) or not any(
            getattr(pb, k) < getattr(pa, k) for k in keys
        ):
            raise ValueError("Reference endpoint-rate tolerances must be stricter")
    ha, hb = a.get("max_time_step_days"), b.get("max_time_step_days")
    if (
        any(
            not isinstance(h, (int, float)) or isinstance(h, bool) or not math.isfinite(h) or h <= 0
            for h in (ha, hb)
        )
        or not hb < ha
    ):
        raise ValueError("Reference maximum time step must be smaller")
    initial = [s.get("initial_time_step_days", h) for s, h in ((a, ha), (b, hb))]
    if any(
        not isinstance(h, (int, float)) or isinstance(h, bool) or not math.isfinite(h) or h <= 0
        for h in initial
    ):
        raise ValueError("Positive finite initial time steps required")
    if min(initial[1], hb) > min(initial[0], ha):
        raise ValueError("Reference must not increase the effective startup step")
    return "endpoint_rate_and_maximum_step" if ra is not None else "maximum_time_step"


def same_other_settings(candidate, reference):
    configs = [copy.deepcopy(x["config"]) for x in (candidate, reference)]
    for c in configs:
        s = c["simulation"]
        for key in ("max_time_step_days", "initial_time_step_days"):
            s.pop(key, None)
        s["coupled_facility"].pop("temporal_control", None)
        s["coupled_facility"].pop("time_control", None)
    if configs[0] != configs[1]:
        raise ValueError("Only time refinement settings may differ in the release pair")


def qualify(candidate, reference, budgets, *, required_days=30.0):
    sa, a, pa = completed(candidate, required_days)
    sb, b, pb = completed(reference, required_days)
    python = lambda s: {k: v for k, v in s["source_sha256"].items() if k.endswith(".py")}
    if not python(sa) or python(sa) != python(sb):
        raise ValueError("Refinement requires identical Python source manifests")
    pvta, pvtb = [x["config"]["fluid"]["pvt_path"] for x in (a, b)]
    if (
        pvta != pvtb
        or pvta not in sa["source_sha256"]
        or pvtb not in sb["source_sha256"]
        or sa["source_sha256"][pvta] != sb["source_sha256"][pvtb]
    ):
        raise ValueError("Refinement requires identical guarded PVT")
    family = stricter(a, b)
    same_other_settings(a, b)
    na, nb = [len(x["result"]["accepted_substeps"]) for x in (a, b)]
    if nb <= na:
        raise ValueError("Reference did not actually increase temporal resolution")
    fields = compare_fields(pa, pb, budgets, allow_numerical_policy=True)
    wells = compare_wells(pa, pb, allow_numerical_policy=True)
    return {
        "scope": __doc__,
        "accepted": fields["accepted"] and wells["accepted"],
        "required_days": required_days,
        "same_python_and_pvt": True,
        "refinement_family": family,
        "accepted_stages": {"candidate": na, "reference": nb},
        "final_fields": fields,
        "well_trajectories": wells,
        "observed_solve_wall_s": {"candidate": sa["solve_wall_s"], "reference": sb["solve_wall_s"]},
        "timing_scope": "Shared host observations, not an isolated speed benchmark",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--budgets", required=True, type=Path)
    parser.add_argument("--required-days", type=float, default=30.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    r = qualify(
        args.candidate,
        args.reference,
        json.loads(args.budgets.read_text()),
        required_days=args.required_days,
    )
    with args.output.open("x") as f:
        json.dump(r, f, indent=2)
    print(json.dumps(r, indent=2))
    return 0 if r["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
