"""Inspect the author's separate dry-air reference without replacing wet scoring."""

import argparse
import datetime
import hashlib
import json
import math
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import torch
from validate_multiphase_experiments import ROOT, run

from surface.pr_active_pipe_friction import darcy_factor

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def cached_sheets(path):
    """Read cached source values through workbook relationships, never formulas."""
    with zipfile.ZipFile(path) as z:
        strings = [
            "".join(s.itertext())
            for s in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)
        ]
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        props = workbook.find("m:workbookPr", NS)
        if props is not None and props.get("date1904") in ("1", "true"):
            raise ValueError("Unsupported source date system")
        relationships = {
            r.get("Id"): r for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        }
        result = {}
        for sheet in workbook.findall("m:sheets/m:sheet", NS):
            rid = sheet.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            rel = relationships[rid]
            if rel.get("TargetMode") == "External":
                raise ValueError("External worksheet not allowed")
            target = posixpath.normpath(posixpath.join("xl", rel.get("Target")))
            cells = {}
            for c in ET.fromstring(z.read(target)).findall("m:sheetData/m:row/m:c", NS):
                v = c.find("m:v", NS)
                if v is not None:
                    cells[c.get("r")] = strings[int(v.text)] if c.get("t") == "s" else v.text
            result[sheet.get("name")] = cells
        return result


def author_summary(path):
    metadata = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Author workbook checksum changed")
    sheets = cached_sheets(path)
    cells = sheets[metadata["sheet"]]
    setup = sheets["Uebersicht_Experiment"]
    for cell, label in {
        "S4": "Rohrdurchmesser Glattrohr [m]",
        "S5": "Rohrdurchmesser Unten [m]",
    }.items():
        if setup.get(cell) != label:
            raise ValueError("Author geometry labels changed")
    metadata["declared_geometry"] = {
        "pipe_diameter_m": float(setup["T4"]),
        "lower_diameter_m": float(setup["T5"]),
        "source_cells": "Uebersicht_Experiment!S4:T5",
    }
    metadata["measurement_notes"] = {f"C{i}": setup.get(f"C{i}") for i in (65, 66, 67, 68, 70)}
    metadata["measurement_notes_sheet"] = "Uebersicht_Experiment"
    dates = {}
    for name, sheet_cells in sheets.items():
        if not name.startswith("Messdaten"):
            continue
        serial = float(sheet_cells["A3"])
        if not 40000 < serial < 50000:
            raise ValueError("Unexpected author acquisition date")
        dates[name] = (
            (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=serial)).date().isoformat()
        )
    metadata["acquisition_start_dates"] = dates
    metadata["same_campaign_as_wet_data_confirmed"] = False
    expected = {
        "B4": "V_pkt [m³/h]",
        "B9": "dP1 [Pa]",
        "B13": "dP2 [Pa]",
        "B18": "rho_mittel[kg/m³]",
        "B30": "mu_gas [Pas]",
        "B20": "U_Rohr",
        "B21": "U_Unten",
        "B33": "DP [Pa]",
    }
    if any(cells.get(k) != v for k, v in expected.items()):
        raise ValueError("Author summary labels or units changed")
    rows = []
    for col in "CDEFGHIJKLMNO":

        def number(row, col=col):
            value = cells.get(f"{col}{row}")
            if value is None or value.startswith("#"):
                return None
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("Nonfinite author summary")
            return result

        q, rho, mu = number(4), number(18), number(30)
        if any(v is None or v <= 0 for v in (q, rho, mu)):
            raise ValueError("Missing positive author inputs")
        u = q / 3600 / (math.pi * 0.085**2 / 4)
        re = rho * u * 0.085 / mu
        fd = float(darcy_factor(torch.tensor(re, dtype=torch.float64), 0.0))
        friction = 3 * fd * rho * u * u / (2 * 0.085)
        simulation = number(33)
        if simulation is None:
            raise ValueError("Missing author simulation")
        rows.append(
            {
                "column": col,
                "q_m3_h": q,
                "author_pipe_velocity_m_s": number(20),
                "author_lower_velocity_m_s": number(21),
                "rho_kg_m3": rho,
                "mu_pa_s": mu,
                "measured_dp1_pa": number(9),
                "measured_dp2_pa": number(13),
                "author_simulation_pa": simulation,
                "fully_developed_friction_pa": friction,
                "unexplained_difference_pa": simulation - friction,
            }
        )
    return metadata, rows


def audit(directory):
    metadata, rows = author_summary(directory / "air_reference.xlsx")
    baseline = run(directory)
    curve = np.array([[r["q_m3_h"], r["author_simulation_pa"]] for r in rows])
    comparisons = []
    for r in baseline["roughness_sensitivity"]["0.0"]["cases"]:
        q = r["gas_volume_flow_m3_h"]
        if not curve[0, 0] <= q <= curve[-1, 0]:
            raise ValueError("Wet window outside independent dry coverage")
        alternate = float(np.interp(q, curve[:, 0], curve[:, 1]))
        comparisons.append(
            {
                "window": [r["sheet"], r["first_row"], r["last_row"]],
                "legacy_dry_pa": r["measured_dry_baseline_pa"],
                "author_dry_simulation_interpolated_pa": alternate,
                "baseline_shift_pa": alternate - r["measured_dry_baseline_pa"],
            }
        )
    return {
        "author": metadata,
        "reference_rows": rows,
        "wet_window_dry_comparison": comparisons,
        "summary": {
            "author_points": len(rows),
            "wet_windows": len(comparisons),
            "baseline_shift_range_pa": [
                min(r["baseline_shift_pa"] for r in comparisons),
                max(r["baseline_shift_pa"] for r in comparisons),
            ],
            "simulation_minus_friction_range_pa": [
                min(r["unexplained_difference_pa"] for r in rows),
                max(r["unexplained_difference_pa"] for r in rows),
            ],
        },
        "experimental_qualification": False,
        "legacy_scoring_changed": False,
        "notes": [
            "Independent dry campaign and CFD output are diagnostic references, not interchangeable measurements of wet runs.",
            "Author cached results are read without executing formulas. Erroneous Kelvin labels in B8/B28 are not used.",
            "Pure distributed Darcy friction does not reproduce the author full-domain CFD pressure: geometry, pressure taps and losses must be matched before assigning the difference to film physics.",
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(ROOT / "tests/data/surface/hannover_countercurrent")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
