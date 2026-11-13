"""Condition FETRig input records and challenge the shared shear-free film limit.

No filename setpoint is substituted for measured mass flow. Unit conversions
missing from the workbooks remain explicit unconfirmed candidates. A common
CCI/LabView clock is not available: ALL input-steady windows are reported as a
sensitivity range, never a pressure/thickness-selected favorable interval.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch
from audit_fetrig_measurements import cci_statistics
from fetrig_labview import channel, window_statistics, workbook_tables

from surface.planar_film import shear_free_gravity_thickness

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = {
    "liquid_flow": ("Mass Flow", "Timestamp-CF", "MF 01"),
    "gas_volume_flow": ("Volume Flow", "Timestamp-TT", "TT-Volume Flow"),
    "liquid_inlet_temperature": ("Temperature", "Timestamp-TC", "TC 14"),
    "gas_density": ("Density", "Timestamp-TT", "TT-Density"),
    "gas_viscosity": ("Viscosity", "Untitled", "TT-Viscosity"),
}
# Candidates explicitly declared BEFORE comparing thickness, never inferred by
# minimizing residuals. Empty workbook Unit fields do not confirm these units.
CANDIDATE_UNITS = {
    "liquid_flow": {"native": "kg/h", "SI": "kg/s", "scale": 1 / 3600, "offset": 0},
    "gas_volume_flow": {"native": "m3/s", "SI": "m3/s", "scale": 1, "offset": 0},
    "liquid_inlet_temperature": {"native": "degC", "SI": "K", "scale": 1, "offset": 273.15},
    "gas_density": {"native": "kg/m3", "SI": "kg/m3", "scale": 1, "offset": 0},
    "gas_viscosity": {"native": "Pa.s", "SI": "Pa.s", "scale": 1, "offset": 0},
}


def steady_windows(channels, *, duration_s=20.0):
    """Fixed nonoverlapping physical-time windows, input-only criteria.

    Every channel must cover a window with >=10 samples and no gaps >2s.
    Liquid mean >.001 kg/s, CV<=2%; gas mean |Q|<=1e-4 m3/s is the
    predeclared near-zero branch, otherwise Q>0 and CV<=2%. Units conditional.
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Positive physical window duration required")
    origin = max(c["times_excel_days"][0] for c in channels.values())
    end = min(c["times_excel_days"][-1] for c in channels.values())
    count = max(0, math.floor((end - origin) * 86400 / duration_s))
    accepted, rejected = [], {}
    for i in range(count):
        first, last = origin + i * duration_s / 86400, origin + (i + 1) * duration_s / 86400
        stats = {name: window_statistics(c, first, last) for name, c in channels.items()}
        reason = None
        if any(value is None for value in stats.values()):
            reason = "insufficient_independent_clock_coverage"
        else:
            liquid, gas = stats["liquid_flow"], stats["gas_volume_flow"]
            liquid_mean = liquid["mean_native"] * CANDIDATE_UNITS["liquid_flow"]["scale"]
            gas_mean = gas["mean_native"]
            if any(stats[name]["mean_native"] <= 0 for name in ("gas_density", "gas_viscosity")):
                reason = "nonphysical_gas_properties"
            elif liquid_mean <= 0.001:
                reason = "liquid_not_positive_operating_flow"
            elif liquid["coefficient_of_variation"] > 0.02:
                reason = "liquid_input_not_steady"
            elif abs(gas_mean) > 1e-4 and (gas_mean <= 0 or gas["coefficient_of_variation"] > 0.02):
                reason = "gas_input_not_steady"
            elif abs(gas_mean) <= 1e-4 and gas["sd_native"] > 1e-4:
                reason = "near_zero_gas_input_not_steady"
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        physical = {
            name: value["mean_native"] * CANDIDATE_UNITS[name]["scale"]
            + CANDIDATE_UNITS[name]["offset"]
            for name, value in stats.items()
        }
        accepted.append(
            {
                "index": i,
                "first_excel_day": first,
                "end_excel_day_exclusive": last,
                "duration_s": duration_s,
                "raw_channel_statistics": stats,
                "candidate_SI_means": physical,
            }
        )
    return {
        "fixed_windows": count,
        "accepted": accepted,
        "rejected_by_input": rejected,
        "selection_uses_thickness_or_model_error": False,
        "duration_s": duration_s,
        "conversion_confirmed": False,
    }


def checked_members(data):
    manifest_raw = (data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    members = {}
    for member in manifest["members"]:
        raw = gzip.decompress((data / member["file"]).read_bytes())
        if (
            len(raw) != member["original_size"]
            or hashlib.sha256(raw).hexdigest() != member["sha256"]
        ):
            raise ValueError("FETRig source identity changed")
        members[member["original_member"]] = raw
    return manifest_raw, manifest, members


def run(data, *, device="cpu"):
    data = Path(data)
    manifest_raw, manifest, members = checked_members(data)
    processing_raw, _, processing = checked_members(data.parent / "fetrig_processing")
    plotter_name = next(name for name in processing if name.endswith("/Keyence_Data_Plotter.m"))
    plotter = processing[plotter_name]
    if not all(
        text in plotter
        for text in (
            b"rho_liq = 997",
            b"mu_liq = 0.89*10^-3",
            b"Re_film_selected = 4.*[500",
        )
    ):
        raise ValueError("Declared source reference properties/Reynolds convention changed")
    cases = []
    for name, raw in members.items():
        if not name.endswith(".csv"):
            continue
        stem = Path(name).stem
        station = name.split("/Y_")[1].split("/")[0]
        matches = [n for n in members if f"/Y_{station}/" in n and n.endswith("/" + stem + ".xlsx")]
        if len(matches) != 1:
            raise ValueError("Ambiguous CCI/LabView experiment pairing")
        tables = workbook_tables(members[matches[0]])
        channels = {key: channel(tables, *spec) for key, spec in CHANNELS.items()}
        for key, value in channels.items():
            native_unit = value["unit_from_workbook"]
            if native_unit is not None and native_unit != CANDIDATE_UNITS[key]["native"]:
                raise ValueError(f"Declared conversion contradicts workbook unit for {key}")
        windows = steady_windows(channels)
        observation = cci_statistics(raw)
        references = []
        # Constant properties taken from the authors' Keyence_Data_Plotter.m,
        # not presented as measured temperature-dependent liquid PVT.
        rho, mu, width, g = 997.0, 0.00089, 0.1, 9.81
        for window in windows["accepted"]:
            mean = window["candidate_SI_means"]
            gamma = mean["liquid_flow"] / width
            reynolds = gamma / mu
            reference = None
            if abs(mean["gas_volume_flow"]) <= 1e-4:
                h = shear_free_gravity_thickness(
                    torch.tensor(gamma, dtype=torch.float64, device=device), rho, mu, g
                )
                reference = float(h.cpu())
            references.append(
                {
                    "window_index": window["index"],
                    "candidate_mass_flow_kg_s": mean["liquid_flow"],
                    "candidate_reynolds_gamma_over_mu": reynolds,
                    "candidate_reynolds_four_gamma_over_mu": 4 * reynolds,
                    "candidate_gas_reynolds_at_meter_conditions": (
                        mean["gas_volume_flow"]
                        / (0.1 * 0.03)
                        * (2 * 0.1 * 0.03 / (0.1 + 0.03))
                        * mean["gas_density"]
                        / mean["gas_viscosity"]
                    ),
                    "shear_free_reference_thickness_m": reference,
                    "within_existing_annular_reynolds_domain": reynolds <= 400,
                }
            )
        predictions = [
            row["shear_free_reference_thickness_m"]
            for row in references
            if row["shear_free_reference_thickness_m"] is not None
        ]
        cases.append(
            {
                "cci_member": name,
                "labview_member": matches[0],
                "cci_raw_sha256": hashlib.sha256(raw).hexdigest(),
                "labview_raw_sha256": hashlib.sha256(members[matches[0]]).hexdigest(),
                "observed_cci": observation,
                "input_channels": {
                    key: {
                        k: v
                        for k, v in c.items()
                        if k not in ("times_excel_days", "values_native", "source_rows")
                    }
                    for key, c in channels.items()
                },
                "input_windows": windows,
                "references": references,
                "shear_free_reference_range_m": [min(predictions), max(predictions)]
                if predictions
                else None,
                "conditional_reference_relative_difference_range": [
                    min(predictions) / observation["go_mean_thickness_m"] - 1,
                    max(predictions) / observation["go_mean_thickness_m"] - 1,
                ]
                if predictions and observation["go_mean_thickness_m"] > 0
                else None,
                "experimental_pass": False,
                "qualification_blockers": [
                    "raw_channel_units_unconfirmed",
                    "cci_labview_clock_alignment_unavailable",
                    "sensor_zero_and_dropout_effects_unconfirmed",
                    "uniform_shear_free_reference_does_not_predict_inertial_waves_or_entrance_development",
                ],
            }
        )
    return {
        "format": "fetrig_conditioned_film_challenge_v1",
        "source": manifest["source"],
        "publication": "https://doi.org/10.1063/5.0263158",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "processing_manifest_sha256": hashlib.sha256(processing_raw).hexdigest(),
        "processing_plotter_sha256": hashlib.sha256(plotter).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/validate_fetrig_film.py",
                "scripts/fetrig_labview.py",
                "scripts/audit_fetrig_measurements.py",
                "src/surface/planar_film.py",
                "src/surface/thin_film_dynamics.py",
            )
        },
        "device": device,
        "cases": cases,
        "fit_performed": False,
        "candidate_units": CANDIDATE_UNITS,
        "unit_confirmation_status": "Missing Unit fields in source books; conversions are conditional, not inferred measurements",
        "reference_properties": {
            "density_kg_m3": 997.0,
            "viscosity_pa_s": 0.00089,
            "temperature_k": 298.15,
            "gravity_m_s2": 9.81,
            "source_member": "Post_Processing/Keyence_Data_Plotter.m, lines 207-221",
            "measured_liquid_PVT": False,
        },
        "comparison_scope": "All input-steady windows, unaligned to CCI, as conditional reference sensitivity. No closest-error selection, empirical coefficient fit or synthetic confidence interval.",
        "experimentally_qualified": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "tests/data/surface/fetrig")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = run(args.data, device=args.device)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for case in report["cases"]:
        print(
            case["cci_member"].split("/Y_")[-1],
            "steady windows",
            len(case["references"]),
            "observed mm",
            case["observed_cci"]["go_mean_thickness_m"] * 1000,
            "reference m",
            case["shear_free_reference_range_m"],
        )
