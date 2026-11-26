"""Time-refinement gate for completed waterflood studies with report snapshots."""

import argparse
import copy
import gzip
import hashlib
import json
import math
import tarfile
from dataclasses import asdict
from pathlib import Path

from check_fim_waterflood import audit

from simulator.facility_temporal_control import FacilityTemporalControl


def check_time_refinement(configs):
    cfg = copy.deepcopy(configs)
    maximum = [c["simulation"].pop("max_time_step_days") for c in cfg]
    policies = []
    for c in cfg:
        c["simulation"].pop("initial_time_step_days", None)
        policies.append(c["simulation"]["coupled_waterflood"].pop("temporal_control", None))
    if cfg[0] != cfg[1] or not 0 < maximum[1] <= maximum[0]:
        raise ValueError("Only stricter numerical time-step settings may differ")
    if policies == [None, None]:
        if maximum[0] == maximum[1]:
            raise ValueError("Reference maximum time step must decrease")
        return "maximum_step"
    if any(p is None for p in policies):
        raise ValueError("Both studies must use the same temporal-control model")
    normalized = []
    extras = {"component_relative_tolerance": 1e-3, "component_absolute_tolerance_kg": 1e-6,
              "drive_energy_relative_tolerance": .002, "drive_energy_absolute_tolerance_j": 1.}
    for raw in policies:
        raw = dict(raw)
        extra = {key: raw.pop(key, value) for key, value in extras.items()}
        normalized.append({**asdict(FacilityTemporalControl(**raw)), **extra})
    fixed = {"startup_step_days", "safety_factor", "maximum_growth", "withdrawal_error_scale"}
    stricter = maximum[1] < maximum[0]
    for key, a in normalized[0].items():
        b = normalized[1][key]
        if key in fixed or a is None or b is None:
            if a != b:
                raise ValueError("Different temporal-control definitions")
        elif (isinstance(b, bool) or not math.isfinite(b) or b <= 0 or b > a):
            raise ValueError("Reference temporal budgets must not be relaxed")
        else:
            stricter |= b < a
    if not stricter:
        raise ValueError("Reference temporal resolution must increase")
    return "temporal_tolerances_and_maximum_step"


def qualify(candidate, reference, budgets, *, execution=False, reference_source_archive=None):
    pairs = []
    metadata = []
    for directory in map(Path, (candidate, reference)):
        summary = json.loads((directory / "summary.json").read_text())
        if (
            summary["status"] != "complete"
            or not summary["sources_unchanged"]
            or (directory / "failure.json").exists()
        ):
            raise ValueError("Completed guarded waterflood study required")
        path = directory / "trajectory.json.gz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary["trajectory_sha256"]:
            raise ValueError("Waterflood checksum mismatch")
        with gzip.open(path, "rt") as stream:
            bundle = json.load(stream)
        audit(bundle["config"], bundle["result"])
        pairs.append(bundle)
        metadata.append(summary)
    if execution:
        cfg = [copy.deepcopy(b["config"]) for b in pairs]
        for c in cfg:
            c["simulation"]["coupled_waterflood"].pop("dispatch_solver", None)
        if cfg[0] != cfg[1]:
            raise ValueError("Execution comparison requires identical physics and temporal settings")
        refinement_axis = "dispatch_execution"
    else:
        refinement_axis = check_time_refinement([b["config"] for b in pairs])
    python = lambda s: {k: v for k, v in s["source_sha256"].items() if k.endswith(".py")}
    versions_identical = python(metadata[0]) == python(metadata[1])
    if not versions_identical and not execution:
        raise ValueError("Different numerical sources")
    if not versions_identical:
        if reference_source_archive is None:
            raise ValueError("Cross-version execution comparison requires the exact archived reference sources")
        with tarfile.open(reference_source_archive, "r:gz") as archive:
            for key, digest in metadata[1]["source_sha256"].items():
                with archive.extractfile(key) as stream:
                    if hashlib.sha256(stream.read()).hexdigest() != digest:
                        raise ValueError("Archived reference source/input mismatch")
    pvt = pairs[0]["config"]["fluid"]["pvt_path"]
    pvt_key = next(
        k
        for k in metadata[0]["source_sha256"]
        if Path(k).name == Path(pvt).name and k.endswith(".json")
    )
    if metadata[0]["source_sha256"][pvt_key] != metadata[1]["source_sha256"][pvt_key]:
        raise ValueError("Different guarded PVT")
    results = [b["result"] for b in pairs]
    if (
        results[0]["schedule"] != results[1]["schedule"]
        or results[0]["requested_days"] != results[1]["requested_days"]
    ):
        raise ValueError("Different operating studies")
    if not execution and len(results[1]["accepted_substeps"]) <= len(results[0]["accepted_substeps"]):
        raise ValueError("Reference did not increase actual resolution")

    def series(result):
        rows = {}
        amount = [0.0] * 12
        for step in result["accepted_substeps"]:
            report = step["coupling"]
            values = (
                report["material_account"]["external_in_kg"]
                + report["material_account"]["external_out_kg"]
            )
            values += [
                report["water_rates_kg_s"][name] * report["dt_s"]
                for name in ["produced", "injected", "recycled", "makeup", "disposed"]
            ]
            values += [report["total_drive_power_w"] * report["dt_s"]]
            for j, v in enumerate(values):
                amount[j] = math.fsum((amount[j], v))
            key = round(step["time_days"] * 86400, 6)
            if key in rows:
                raise ValueError("Repeated accepted time")
            rows[key] = (step, amount.copy())
        return rows

    a, b = map(series, results)
    fields = [
        {round(row["time_days"] * 86400, 6): row for row in r["reservoir_snapshots"]}
        for r in results
    ]
    if not fields[0] or fields[0].keys() != fields[1].keys():
        raise ValueError("Same saved report-field dates required")
    errors = {
        "pressure_pa": 0.0,
        "s_w": 0.0,
        "s_g": 0.0,
        "bhp_pa": 0.0,
        "component_rate_scaled": 0.0,
        "cumulative_transfer_mass_relative": 0.0,
        "cumulative_drive_energy_relative": 0.0,
    }
    mismatches = 0
    for clock in fields[0]:
        for key in ["pressure_pa", "s_w", "s_g"]:
            left, right = fields[0][clock][key], fields[1][clock][key]
            if len(left) != len(right):
                raise ValueError("Different field size")
            errors[key] = max(
                errors[key], max(abs(x - y) for x, y in zip(left, right, strict=True))
            )
        sa, ma = a[clock]
        sb, mb = b[clock]
        if sa["coupling"]["operator_controls"] != sb["coupling"]["operator_controls"]:
            raise ValueError("Different controls at common report")
        for name, x in sa["surface"]["wells"].items():
            y = sb["surface"]["wells"][name]
            mismatches += x["mode"] != y["mode"]
            if x.get("bhp_status") != y.get("bhp_status"):
                raise ValueError("Different physical well-pressure status")
            if x.get("bhp_status") != "isolated_gauge":
                errors["bhp_pa"] = max(errors["bhp_pa"], abs(x["bhp_pa"] - y["bhp_pa"]))
            errors["component_rate_scaled"] = max(
                errors["component_rate_scaled"],
                max(
                    abs(q - r) / max(abs(r), 1e-8)
                    for q, r in zip(
                        x["component_mass_rates_kg_s"], y["component_mass_rates_kg_s"], strict=True
                    )
                ),
            )
        errors["cumulative_transfer_mass_relative"] = max(
            errors["cumulative_transfer_mass_relative"],
            max(abs(x - y) / max(abs(y), 1.0) for x, y in zip(ma[:-1], mb[:-1], strict=True)),
        )
        errors["cumulative_drive_energy_relative"] = max(
            errors["cumulative_drive_energy_relative"], abs(ma[-1] - mb[-1]) / max(abs(mb[-1]), 1.0)
        )
    if set(budgets) != set(errors) or any(
        isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in budgets.values()
    ):
        raise ValueError("Predeclared finite budgets for every metric required")
    return {
        "accepted": mismatches == 0
        and all(math.isfinite(v) and v <= budgets[k] for k, v in errors.items()),
        "errors": errors,
        "budgets": budgets,
        "mode_mismatches": mismatches,
        "refinement_axis": refinement_axis,
        "source_versions_identical": versions_identical,
        "source_fingerprints": [hashlib.sha256(json.dumps(python(s), sort_keys=True).encode()).hexdigest()
                                for s in metadata],
        "reference_source_archive_sha256": (hashlib.sha256(Path(reference_source_archive).read_bytes()).hexdigest()
                                            if reference_source_archive is not None else None),
        "report_dates": len(fields[0]),
        "days": results[0]["requested_days"],
        "accepted_stages": [len(r["accepted_substeps"]) for r in results],
        "scope": "Matched report fields, actual well rates/BHP and external product integrals; refinement, not experimental truth or a global dispatch optimum.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument("--budgets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execution", action="store_true", help="Compare dispatch implementations at identical physics/time settings")
    p.add_argument("--reference-source-archive", type=Path)
    args = p.parse_args()
    result = qualify(args.candidate, args.reference, json.loads(args.budgets.read_text()),
                     execution=args.execution, reference_source_archive=args.reference_source_archive)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["accepted"] else 1)
