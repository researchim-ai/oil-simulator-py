"""Check a completed validate_norne.py run and compare saved Eclipse/OPM output.

No solver is run here. Numerical acceptance checks apply to the recorded
strict substeps; reference differences are reported, not hidden or fitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulator.ecl_io import EclSummary
from compare_eclipse import comparison_report


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_records(meta, status, manifest, rows, steps, summary):
    """Cross-check report and accepted-step clocks and strict convergence."""
    end = float(manifest["requested_days"])
    require(meta["status"] == "done" and status.get("status") == "complete", "Incomplete run or audit")
    require(not status.get("failed_audit", False), "Failed audit")
    require(abs(meta["days"] - end) <= 1e-7, "Requested horizon not reached")
    c = meta["convergence"]
    require(c["tol_cnv"] == 1e-3 and c["tol_mb"] == 1e-7 and c["relaxed_max_pv_fraction"] == 0,
            "Run does not use the strict convergence profile")
    require(len(rows) > 0 and steps.ndim == 2 and steps.shape[1] == 5, "Missing accepted substeps")
    accepted = steps[steps[:, 2] == 1]
    require(np.isfinite(steps).all() and np.all(steps[:, 1] > 0), "Invalid substep attempt")
    require(np.isin(steps[:, 2], [0, 1]).all(), "Invalid substep success flag")
    require(len(rows) == len(accepted) == status["accepted"], "Accepted-step counts disagree")
    require(int((steps[:, 2] == 0).sum()) == meta["failed_substeps"] == status["failed"],
            "Rejected-step counts disagree")
    require(meta["substeps"] == len(steps), "Attempt count disagrees with metadata")
    require(int(accepted[:, 3].sum()) == meta["newton_iters"], "Newton counts disagree")
    require(np.allclose(rows["newton"], accepted[:, 3], rtol=0, atol=0), "Audit Newton counts disagree")
    day, dt = rows["day"], rows["dt_days"]
    require(np.isfinite(day).all() and np.isfinite(dt).all() and (dt > 0).all(), "Invalid accepted clock")
    require(np.allclose(day, np.cumsum(dt), rtol=0, atol=1e-7), "Accepted-step clock has gaps or overlap")
    require(np.allclose(day, accepted[:, 0] + accepted[:, 1], rtol=0, atol=1e-7), "Audit clock disagrees")
    require(np.allclose(dt, accepted[:, 1], rtol=0, atol=1e-10), "Audit durations disagree")
    require(abs(day[-1] - end) <= 1e-7, "Last accepted substep misses requested end")
    clock = np.asarray(summary["TIME"])
    require(len(clock) > 1 and np.isfinite(clock).all() and (np.diff(clock) > 0).all(), "Invalid report clock")
    require(abs(clock[0]) < 1e-7 and abs(clock[-1] - end) < 1e-7, "Report horizon disagrees")
    require(all(np.asarray(v).shape == clock.shape for v in summary.values()), "SUMMARY vector shape mismatch")
    require(all(np.any(abs(day - t) < 1e-7) for t in clock[1:]), "Report has no accepted substep")
    result = {"accepted": len(rows), "failed_attempts": meta["failed_substeps"], "max_cnv": {},
              "max_mb": {}, "max_abs_accumulated_balance_rel": {}, "final_balance_rel": {}}
    for phase in ("oil", "water", "gas"):
        for prefix, limit in (("cnv_", 1e-3), ("mb_", 1e-7)):
            values = rows[prefix + phase]
            require(np.isfinite(values).all() and (values >= 0).all() and (values <= limit).all(),
                    f"Invalid or non-strict {prefix}{phase}")
            result["max_" + prefix[:-1]][phase] = float(values.max())
        values = rows["balance_rel_" + phase]
        require(np.isfinite(values).all(), f"Nonfinite balance for {phase}")
        require(float(np.max(abs(values))) <= 1e-4, f"Accumulated mass balance exceeds 1e-4 for {phase}")
        result["max_abs_accumulated_balance_rel"][phase] = float(np.max(abs(values)))
        result["final_balance_rel"][phase] = float(values[-1])
        require(np.isfinite(rows["mass_" + phase + "_kg"]).all(), f"Invalid {phase} mass")
    for phase in ("sw", "sg", "so"):
        values = rows["min_" + phase]
        require(np.isfinite(values).all() and (values >= -1e-10).all() and (values <= 1).all(),
                f"Invalid minimum saturation {phase}")
    return result


class MetricReference:
    """Blank legacy GOR units in this METRIC deck mean SM3/SM3."""
    def __init__(self, reference):
        self.reference = reference

    def __getattr__(self, name):
        return getattr(self.reference, name)

    def unit(self, key, name=None, num=None):
        unit = self.reference.unit(key, name, num)
        return "SM3/SM3" if key.endswith("GOR") and not unit.strip() else unit


def check_run(directory, deck_dir, previous_run=None):
    root, deck = Path(directory), Path(deck_dir)
    meta = json.loads((root / "meta.json").read_text())
    status = json.loads((root / "audit_status.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    require(not (root / "audit_failure.json").exists(), "Run has an audit_failure.json")
    rows = np.atleast_1d(np.genfromtxt(root / "accepted_substeps.csv", delimiter=",", names=True))
    with np.load(root / "substeps.npz") as z:
        steps = z["data"]
    with np.load(root / "summary.npz") as z:
        summary = dict(z)
    audit = check_records(meta, status, manifest, rows, steps, summary)
    for group, base in (("source_files", root / "source_snapshot"), ("input_files", deck), ("reference_files", deck)):
        require(bool(manifest[group]), f"Missing {group} hashes")
        for name, expected in manifest[group].items():
            require(hashlib.sha256((base / name).read_bytes()).hexdigest() == expected, f"Hash mismatch: {group}/{name}")
    references = {"eclipse": MetricReference(EclSummary(deck / "ECL.2014.2/NORNE_ATW2013"))}
    opm = deck / "opm-simulation-reference/flow_legacy/NORNE_ATW2013"
    if opm.with_suffix(".SMSPEC").exists():
        references["opm"] = MetricReference(EclSummary(opm))
    comparison = comparison_report(summary, references)
    field = {}
    for key in ("FOPT", "FWPT", "FGPT"):
        ours = float(summary[key][-1])
        ref = references["eclipse"]
        expected = float(np.interp(meta["days"], ref.days, ref.get(key)))
        field[key] = {"ours": ours, "eclipse": expected,
                      "relative_error_pct": 100 * (ours - expected) / abs(expected) if expected else None}
    for key, production, phase in (("FWIT", "FWPT", "water"), ("FGIT", "FGPT", "gas")):
        ours = float(summary[production][-1] - rows["net_surface_" + phase + "_m3"][-1])
        ref = references["eclipse"]
        expected = float(np.interp(meta["days"], ref.days, ref.get(key)))
        field[key] = {"ours": ours, "eclipse": expected,
                      "relative_error_pct": 100 * (ours - expected) / abs(expected) if expected else None}
    late = {}
    selected = {key: value for key, value in summary.items()
                if key == "TIME" or key.startswith(("WWPR:B-1H", "WWPT:B-1H", "WWPR:E-4AH", "WWPT:E-4AH"))}
    if len(selected) > 1:
        for begin, end in ((1095, 1825), (1825, 2555), (2555, 3312)):
            if np.any((summary["TIME"][1:] >= begin) & (summary["TIME"][1:] <= end)):
                late[f"{begin}-{end}"] = comparison_report(selected, references, begin, end)
    water_wells = [p for p in comparison["peaks"]
                   if p["reference"] == "eclipse" and p["vector"].startswith("WWPT:")]
    report = {"status": "passed_recorded_audit", "days": meta["days"], "wall_s": meta["wall_s"],
              "device": meta["device"], "threads": meta["threads"], "audit": audit,
              "field_cumulative": field, "references": comparison,
              "late_water_windows": late,
              "worst_water_wells_by_final_absolute_error": sorted(
                  water_wells, key=lambda p: abs(p["final_signed_error"]), reverse=True)[:12],
              "source_and_input_hashes_verified": True,
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Strict isothermal reservoir deck; reference agreement is reported separately from audit acceptance."}
    if previous_run is not None:
        with np.load(Path(previous_run) / "summary.npz") as old:
            field_old = {}
            for key in ("FOPT", "FWPT", "FGPT"):
                require(old["TIME"][0] <= meta["days"] <= old["TIME"][-1], "Historical output does not cover this horizon")
                expected = float(np.interp(meta["days"], old["TIME"], old[key]))
                field_old[key] = {"historical": expected, "absolute_difference": float(summary[key][-1] - expected),
                                  "relative_difference_pct": 100 * float(summary[key][-1] - expected) / abs(expected) if expected else None}
        report["historical_output_comparison"] = {"path": str(previous_run), "field_cumulative": field_old,
                                                   "note": "May differ in source version, device, and timestep decisions; not a controlled speed comparison."}
    return report


def compare_same_source_summaries(first, second):
    """Report cross-device differences on equal report clocks and frozen inputs.

    Both runs must be audited separately. There is no common accuracy threshold
    for all well rates, pressures, ratios and cumulative production vectors.
    """
    first, second = Path(first), Path(second)
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (first, second)]
    for key in ("source_files", "input_files", "reference_files"):
        require(manifests[0][key] == manifests[1][key], f"Different frozen {key}")
    vectors = []
    with np.load(first / "summary.npz") as a, np.load(second / "summary.npz") as b:
        require(set(a.files) == set(b.files), "Different SUMMARY vector sets")
        require(np.array_equal(a["TIME"], b["TIME"]), "Different report clocks")
        for key in sorted(set(a.files) - {"TIME"}):
            x, y = a[key], b[key]
            require(x.shape == y.shape == a["TIME"].shape, "Different SUMMARY vector shapes")
            valid = np.isfinite(x) & np.isfinite(y)
            mismatched = int(np.count_nonzero(np.isfinite(x) != np.isfinite(y)))
            if valid.any():
                indices = np.flatnonzero(valid)
                delta = abs(x[valid] - y[valid])
                peak = int(np.argmax(delta))
                scale = max(1., float(np.max(abs(x[valid]))), float(np.max(abs(y[valid]))))
                vectors.append({"vector": key, "max_absolute_difference": float(delta[peak]),
                                "max_scaled_difference": float(delta[peak] / scale),
                                "peak_day": float(a["TIME"][indices[peak]]),
                                "final_signed_difference": float(x[-1] - y[-1]) if valid[-1] else None,
                                "finite_mask_mismatches": mismatched})
            else:
                vectors.append({"vector": key, "max_absolute_difference": None,
                                "max_scaled_difference": None, "finite_mask_mismatches": mismatched})
    return {"first": str(first), "second": str(second), "sources_and_inputs_equal": True,
            "report_clocks_equal": True,
            "normalization": "max(1, max(abs(first)), max(abs(second))) per vector; errors reported without a common acceptance threshold",
            "vectors": vectors,
            "timing_scope": "Concurrent runs on a shared host are not an isolated hardware speed benchmark."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--deck-dir", required=True)
    parser.add_argument("--previous-run")
    parser.add_argument("--same-source-run", help="Another completed audited run with identical frozen sources and inputs")
    args = parser.parse_args()
    result = check_run(args.directory, args.deck_dir, args.previous_run)
    if args.same_source_run:
        other = check_run(args.same_source_run, args.deck_dir)
        result["same_source_comparison"] = compare_same_source_summaries(args.directory, args.same_source_run)
        result["same_source_comparison"]["other_audit"] = other["audit"]
    destination = Path(args.directory) / "validation.json"
    tmp = destination.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(destination)
    print(json.dumps({k: result[k] for k in ("status", "days", "wall_s", "field_cumulative", "audit")}, indent=2))
