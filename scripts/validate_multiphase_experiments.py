"""Unfitted constitutive challenge against Hannover air/water countercurrent data.

Uses published geometry and raw measurements (CC BY 4.0), not digitized plots.
Compares wet-minus-dry pressure loss at matched gas flow. The prediction is the
steady, constant-property, equal-pressure limit of the existing volume-wetted
Darcy source without interphase drag or chemical transfer. It is NOT an end-to-end
PR water simulation. Failure disqualifies this closure for these annular runs;
no coefficient is fitted and no out-of-envelope correlation is forced to run.
"""

import argparse
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import torch

from surface.pr_active_pipe_friction import darcy_factor

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_xlsx(path):
    """Read original numeric/cached cells without executing Excel formulas."""
    with zipfile.ZipFile(path) as z:
        texts = [
            "".join(x.itertext())
            for x in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", NAMESPACE)
        ]
        sheets = []
        names = ET.fromstring(z.read("xl/workbook.xml")).findall("s:sheets/s:sheet", NAMESPACE)
        for index, sheet in enumerate(names, 1):
            rows = []
            tree = ET.fromstring(z.read(f"xl/worksheets/sheet{index}.xml"))
            header = None
            units = None
            for row in tree.findall("s:sheetData/s:row", NAMESPACE):
                fields = {}
                for cell in row.findall("s:c", NAMESPACE):
                    value = cell.find("s:v", NAMESPACE)
                    if value is None:
                        continue
                    text = value.text
                    if cell.get("t") == "s":
                        text = texts[int(text)]
                    fields["".join(c for c in cell.get("r") if c.isalpha())] = text
                if row.get("r") == "1":
                    header = fields
                    continue
                if row.get("r") == "2":
                    units = fields
                    continue
                try:
                    values = {k: float(fields[k]) for k in "CDEFGH"}
                except (KeyError, ValueError):
                    continue
                rows.append({"row": int(row.get("r")), **values})
            if (
                header is None
                or header.get("C") != "massflow"
                or header.get("D") != "volumeflow"
                or header.get("E") != "dp"
                or units.get("C") != "kg/s"
                or units.get("D") != "m³/h"
                or units.get("E") != "Pa"
                or units.get("H") != "bar"
            ):
                raise ValueError("Unexpected experimental workbook columns or units")
            sheets.append((sheet.get("name"), rows))
    return sheets


def windows(path, *, wet, block=60):
    accepted = []
    rejected = {}
    for name, rows in read_xlsx(path):
        for offset in range(0, len(rows) - block + 1, block):
            group = rows[offset : offset + block]
            reason = None
            if any(not math.isfinite(r[k]) for r in group for k in "CDEFGH"):
                reason = "nonfinite_measurements"
            else:
                q = np.array([r["D"] for r in group])
                m = np.array([r["C"] for r in group])
                if not 100 <= q.mean() <= 200:
                    reason = "outside_published_gas_flow_range"
                elif q.std() / q.mean() > 0.05:
                    reason = "gas_flow_not_steady"
                elif wet and not 0.03 <= m.mean() <= 0.13:
                    reason = "outside_published_liquid_flow_range"
                elif wet and m.std() / m.mean() > 0.10:
                    reason = "liquid_flow_not_steady"
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            accepted.append(
                {
                    "sheet": name,
                    "first_row": group[0]["row"],
                    "last_row": group[-1]["row"],
                    "gas_volume_flow_m3_h": float(q.mean()),
                    "liquid_mass_flow_kg_s": float(m.mean()) if wet else 0.0,
                    "pressure_loss_pa": statistics.fmean(r["E"] for r in group),
                    "pressure_sample_sd_pa": statistics.pstdev(r["E"] for r in group),
                    "temperature_k": 273.15
                    + statistics.fmean((r["F"] + r["G"]) / 2 for r in group),
                    "absolute_pressure_pa": 1e5 * statistics.fmean(r["H"] for r in group),
                    "gas_flow_cv": float(q.std() / q.mean()),
                    "liquid_flow_cv": float(m.std() / abs(m.mean())) if wet else None,
                }
            )
    return accepted, rejected


def predicted_increment(
    gas_flux, liquid_mass_flow, rho_g, *, roughness, diameter=0.085, length=3.0
):
    """Thin-liquid steady branch of the IMPLEMENTED two-momentum wall closure.

    Upward coordinate; gas flows up, liquid down. Equal phase pressure gradient
    implies -rho_l*g+f_l*rho_l*u_l^2/(2D) = -rho_g*g-f_g*rho_g*u_g^2/(2D).
    Select the continuous gas-core branch alpha_l in (0,.1), no hidden fit.
    """
    tensor = lambda v: torch.tensor(v, dtype=torch.float64)
    jl = -tensor(liquid_mass_flow) / (998.2 * math.pi * diameter**2 / 4)
    jg = tensor(gas_flux)
    rg = tensor(rho_g)
    lo = torch.full_like(jg, 1e-12)
    hi = torch.full_like(jg, 0.1)

    def evaluate(alpha):
        ul = jl / alpha
        ug = jg / (1 - alpha)
        fl = darcy_factor(998.2 * ul.abs() * diameter / 0.001002, roughness / diameter)
        fg = darcy_factor(rg * ug.abs() * diameter / 1.81e-5, roughness / diameter)
        gas_wall = fg * rg * ug.square() / (2 * diameter)
        residual = (
            -998.2 * 9.80665 + fl * 998.2 * ul.square() / (2 * diameter) + rg * 9.80665 + gas_wall
        )
        return residual, gas_wall

    if not bool(((evaluate(lo)[0] > 0) & (evaluate(hi)[0] < 0)).all()):
        raise ValueError("No declared thin-film branch for these measured flow rates")
    for _ in range(48):
        mid = (lo + hi) / 2
        positive = evaluate(mid)[0] > 0
        lo, hi = torch.where(positive, mid, lo), torch.where(positive, hi, mid)
    alpha = (lo + hi) / 2
    wet = evaluate(alpha)[1]
    fd = darcy_factor(rg * jg.abs() * diameter / 1.81e-5, roughness / diameter)
    dry = fd * rg * jg.square() / (2 * diameter)
    return (length * (wet - dry)).numpy(), alpha.numpy()


def run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Experimental source checksum changed")
    dry, dry_rejected = windows(directory / "dry_pipe.xlsx", wet=False)
    wet, wet_rejected = windows(directory / "countercurrent.xlsx", wet=True)
    # Fixed 5 m3/h bins reduce many repeated dry samples to one baseline point.
    bins = {}
    for row in dry:
        key = math.floor(row["gas_volume_flow_m3_h"] / 5)
        bins.setdefault(key, []).append(row)
    baseline = sorted(
        [
            {
                "q": statistics.fmean(r["gas_volume_flow_m3_h"] for r in rows),
                "dp": statistics.fmean(r["pressure_loss_pa"] for r in rows),
                "temperature_k": statistics.fmean(r["temperature_k"] for r in rows),
                "absolute_pressure_pa": statistics.fmean(r["absolute_pressure_pa"] for r in rows),
            }
            for rows in bins.values()
        ],
        key=lambda r: r["q"],
    )
    if len(baseline) < 3:
        raise ValueError("Insufficient steady dry baseline coverage")
    x = np.array([r["q"] for r in baseline])
    y = np.array([r["dp"] for r in baseline])
    usable = [r for r in wet if x[0] <= r["gas_volume_flow_m3_h"] <= x[-1]]
    q = np.array([r["gas_volume_flow_m3_h"] for r in usable])
    # Air density from each measured absolute P and mean T; water reference
    # properties fixed at 20 C, not disguised as measured PVT or PR validation.
    rg = np.array([r["absolute_pressure_pa"] / (287.05 * r["temperature_k"]) for r in usable])
    jg = q / 3600 / (math.pi * 0.085**2 / 4)
    ml = np.array([r["liquid_mass_flow_kg_s"] for r in usable])
    actual = np.array([r["pressure_loss_pa"] for r in usable]) - np.interp(q, x, y)
    roughness_cases = {}
    for roughness in (0.0, 0.0002):
        prediction, alpha = predicted_increment(jg, ml, rg, roughness=roughness)
        errors = prediction - actual
        roughness_cases[str(roughness)] = {
            "roughness_m": roughness,
            "fit_performed": False,
            "mean_absolute_error_pa": float(np.mean(np.abs(errors))),
            "bias_pa": float(np.mean(errors)),
            "fraction_within_instrument_only_4pa_bound": float(np.mean(np.abs(errors) <= 4.0)),
            "predicted_increment_range_pa": [float(prediction.min()), float(prediction.max())],
            "predicted_liquid_holdup_range": [float(alpha.min()), float(alpha.max())],
            "cases": [
                {
                    **r,
                    "measured_dry_baseline_pa": float(base),
                    "measured_wet_increment_pa": float(obs),
                    "predicted_wet_increment_pa": float(pred),
                    "predicted_liquid_holdup": float(a),
                    "error_pa": float(err),
                }
                for r, base, obs, pred, a, err in zip(
                    usable, np.interp(q, x, y), actual, prediction, alpha, errors
                )
            ],
        }
    best_coverage = max(
        r["fraction_within_instrument_only_4pa_bound"] for r in roughness_cases.values()
    )
    return {
        "format": "multiphase_experimental_challenge_v1",
        "source": manifest["source"],
        "license": manifest["license"],
        "source_sha256": manifest["sha256"],
        "geometry": manifest["geometry"],
        "qualification_scope": "steady constitutive pressure augmentation; not full PR water or transient validation",
        "closure": "darcy_volume_wetted_v1, no interphase drag, no chemical transfer",
        "held_out": True,
        "fit_performed": False,
        "steady_selection": "nonoverlapping 60-row blocks; input-flow CV only, no pressure-error selection",
        "dry_windows": len(dry),
        "wet_windows": len(wet),
        "matched_wet_windows": len(usable),
        "rejected_dry": dry_rejected,
        "rejected_wet": wet_rejected,
        "wet_outside_dry_coverage": len(wet) - len(usable),
        "measured_wet_increment_range_pa": [float(actual.min()), float(actual.max())],
        "dry_baseline": baseline,
        "roughness_sensitivity": roughness_cases,
        "instrument_bound_note": "Each DP sensor bound is +/-2 Pa in the publication; +/-4 Pa is worst-case wet-minus-dry instrumental bound ONLY, not a statistical confidence interval; no SE independence assumption.",
        "water_properties_assumption": {
            "temperature_k": 293.15,
            "density_kg_m3": 998.2,
            "viscosity_pa_s": 0.001002,
        },
        "accepted_as_validated_for_annular_countercurrent": best_coverage == 1.0,
        "exclusions": {
            "relap7": "Annular alpha_g >.7 lies outside declared bubbly/slug envelope; not scored.",
            "stratified_horizontal": "Vertical geometry is outside the declared near-horizontal envelope; not scored.",
            "kth_hwat": "Not scored: radius/power units and the velocity definition require confirmation from the paper; no silent unit inference.",
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "tests/data/surface/hannover_countercurrent")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = run(args.data)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("roughness_sensitivity", "dry_baseline")}
        )
    )
    for key, row in report["roughness_sensitivity"].items():
        print(json.dumps({"roughness_m": key, **{k: v for k, v in row.items() if k != "cases"}}))


if __name__ == "__main__":
    main()
