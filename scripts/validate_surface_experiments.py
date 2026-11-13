"""Run external measurement challenges; negative qualification is a nonzero gate.

Internal conservation/manufactured/CoolProp implementation tests are deliberately
not promoted to experimental passes. Existing validators and raw data are reused.
"""

import argparse
import hashlib
import json
from pathlib import Path

from validate_annular_film_experiment import validate as annular
from validate_fetrig_film import run as fetrig
from validate_may2015_vle import ROOT
from validate_may2015_vle import run as vle
from validate_multiphase_experiments import run as darcy
from validate_pr_external_thermo import run as antoine


def run(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("External-validation directory must be empty")
    watched = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        *sorted((ROOT / "scripts").glob("validate*.py")),
        ROOT / "scripts/import_may2015_vle.py",
        ROOT / "scripts/fetrig_labview.py",
        ROOT / "scripts/audit_fetrig_measurements.py",
        *sorted(p for p in (ROOT / "tests/data/surface/fetrig").rglob("*") if p.is_file()),
        *sorted(
            p for p in (ROOT / "tests/data/surface/fetrig_processing").rglob("*") if p.is_file()
        ),
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    data = ROOT / "tests/data/surface/hannover_countercurrent"
    jobs = [
        (
            "methane_ethane_vle",
            lambda: vle(
                ROOT / "tests/data/surface/may2015_methane_ethane.json",
                ROOT / "configs/validation_may2015_pr.json",
            ),
        ),
        (
            "pure_saturation_correlation",
            lambda: antoine(ROOT / "tests/data/surface/nist_antoine_reference.json"),
        ),
        ("volume_wetted_control", lambda: darcy(data)),
        ("annular_mean_control", lambda: annular(data, shear_velocity="mean")),
        ("annular_interface", lambda: annular(data, shear_velocity="interface")),
        ("fetrig_conditioned_film", lambda: fetrig(ROOT / "tests/data/surface/fetrig")),
    ]
    reports, entries = {}, {}
    for name, job in jobs:
        report = job()
        target = output / (name + ".json")
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        reports[name] = report
        entries[name] = {
            "file": target.name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha for p, sha in hashes.items()):
        raise RuntimeError("Experimental comparison sources changed during run")
    current = reports["annular_interface"]
    v = reports["methane_ethane_vle"]
    qualified = current["accepted_as_validated_for_annular_countercurrent"]
    result = {
        "format": "surface_external_experiment_gate_v2",
        "source_sha256": hashes,
        "reports": entries,
        "fit_performed": False,
        "vle_measurement_comparison": {
            "points": v["experimental_points"],
            "predicted": v["predicted_points"],
            "pressure_within_archived_marginal_bounds": v["pressure_within_archived_bounds"],
            "vapor_within_archived_marginal_bounds": v["vapor_within_archived_bounds"],
            "pressure_mare": v["pressure_mean_absolute_relative_error"],
            "pressure_maximum_relative_error": v["maximum_pressure_absolute_relative_error"],
            "scope": "Binary pressure/composition only; no joint confidence or complete EOS/plant qualification",
        },
        "pure_saturation_comparison": {
            "points": len(reports["pure_saturation_correlation"]["cases"]),
            "maximum_relative_error": reports["pure_saturation_correlation"][
                "maximum_absolute_relative_error"
            ],
            "scope": "Empirical NIST correlation, not a new set of raw experiments",
        },
        "annular_pressure_gate": {
            "scope": "Hannover matched steady wet-minus-dry pressure increment, interface-velocity annular closure",
            "status": "within_declared_instrument_only_bounds"
            if qualified
            else "not_experimentally_qualified",
            "qualification_blockers": current["qualification_blockers"],
            "within_declared_pressure_bounds": current["within_declared_pressure_bounds"],
            "dry_series_sensitivity": {
                k: {
                    key: value
                    for key, value in r["dry_series_sensitivity"].items()
                    if key != "dry_series"
                }
                for k, r in current["roughness_sensitivity"].items()
            },
            "points": current["matched_wet_windows"],
            "criterion": "Existing unfitted protocol: all matched input-domain windows within +/-4 Pa for both predeclared dry roughness baselines; instrumental difference bound, not a statistical confidence interval",
            "mean_absolute_error_pa": {
                k: r["mean_absolute_error_pa"] for k, r in current["roughness_sensitivity"].items()
            },
            "within_bound_fraction": {
                k: r["fraction_within_instrument_only_4pa_bound"]
                for k, r in current["roughness_sensitivity"].items()
            },
            "passed": qualified,
        },
        "complete_surface_plant_experimentally_qualified": False,
        "fetrig_conditioning": {
            "cases": len(reports["fetrig_conditioned_film"]["cases"]),
            "input_steady_windows": sum(
                len(row["input_windows"]["accepted"])
                for row in reports["fetrig_conditioned_film"]["cases"]
            ),
            "experimentally_qualified": False,
            "status": "conditional_shear_free_reference_only; channel units and CCI clock alignment unconfirmed",
        },
        "uncovered_scopes": [
            "transient multiphase pipes and finite ports against time-resolved measurements",
            "separator levels and phase entrainment",
            "equipment maps",
            "oil/water and real PVT/thermal properties",
            "complete reservoir/well/surface assembly",
        ],
        "gate_requirements": {
            "annular_pressure_and_applicability": qualified,
            "all_vle_states_computed": v["predicted_points"] == v["experimental_points"],
            "fetrig_experimental_qualification": reports["fetrig_conditioned_film"][
                "experimentally_qualified"
            ],
        },
        "gate_passed": qualified
        and v["predicted_points"] == v["experimental_points"]
        and reports["fetrig_conditioned_film"]["experimentally_qualified"],
        "qualification": "Scoped external comparison gate, not a whole-project industrial certification. Numerical verification and experimental validation remain distinct. Failure is retained, not converted into a test pass.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Archive even an unsuccessful qualification; does not change gate_passed",
    )
    args = p.parse_args()
    result = run(args.output_dir)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("source_sha256", "reports")}, indent=2
        )
    )
    if not result["gate_passed"] and not args.report_only:
        raise SystemExit(1)
