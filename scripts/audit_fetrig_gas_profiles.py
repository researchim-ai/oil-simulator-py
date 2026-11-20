"""Identify measured observables before using public profiles as shear data."""

import argparse
import io
import json
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from audit_hannover_air_reference import cached_sheets
from validate_fetrig_film import checked_members

ROOT = Path(__file__).resolve().parents[1]


def run(data):
    _, manifest, members = checked_members(Path(data))
    book = next(v for n, v in members.items() if n.endswith(".xlsx"))
    plotter = next(v for n, v in members.items() if n.endswith(".m"))
    doc = next(v for n, v in members.items() if n.endswith(".docx"))
    with zipfile.ZipFile(io.BytesIO(doc)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        documentation = " ".join(e.text or "" for e in root.iter() if e.tag.endswith("}t"))
    if not re.search(r"Liquid:\s*-", documentation) or b"DryChannel" not in plotter:
        raise ValueError("Declared dry-channel measurement contract changed")
    if b"u [m/s]" not in plotter or b"z [mm]" not in plotter:
        raise ValueError("Author velocity/position units changed")
    with tempfile.TemporaryDirectory(prefix="fetrig-gas-audit-") as temp:
        path = Path(temp) / "overview.xlsx"
        path.write_bytes(book)
        sheets = cached_sheets(path)
    cells = sheets["Zusammenfassung_Geschwindigkeit"]
    if cells["A2"] != "Z [mm]" or cells["G2"] != "Uncertainity SVM":
        raise ValueError("Unexpected gas-profile coordinates/uncertainty columns")
    positions = sorted(int(k[1:]) for k in cells if re.fullmatch(r"A[0-9]+", k) and int(k[1:]) >= 3)
    profiles = []
    for column, nominal in zip("BCDEF", (4900, 8900, 12900, 16900, 20900)):
        if cells[column + "2"] != f"ReG{nominal}":
            raise ValueError("Gas-profile setpoint column changed")
        values = [
            {
                "source_row": r,
                "coordinate_m": float(cells[f"A{r}"]) / 1000,
                "velocity_m_s": float(cells[f"{column}{r}"]),
                "instrument_velocity_uncertainty_m_s": float(cells[f"G{r}"]),
            }
            for r in positions
        ]
        profiles.append({"nominal_gas_reynolds": nominal, "points": values})
    return {
        "format": "fetrig_gas_profile_observation_contract_v1",
        "source": manifest["source"],
        "profiles": profiles,
        "points": sum(len(p["points"]) for p in profiles),
        "measurement": "dry-channel gas velocity; campaign explicitly has no liquid",
        "coordinate_zero_is_wall_confirmed": False,
        "author_plotter_also_constructs_shifted_coordinates": b"Z_mm_(:)+2" in plotter,
        "wall_coordinate_shift_applied": False,
        "direct_wet_interfacial_shear_available": False,
        "fit_performed": False,
        "experimental_wet_shear_qualification": False,
        "qualification_blockers": [
            "no_liquid_interface_in_this_campaign",
            "unconfirmed_probe_to_wall_coordinate_zero",
            "velocity_uncertainty_is_not_shear_uncertainty",
        ],
        "source_sha256": {r["original_member"]: r["sha256"] for r in manifest["members"]},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "tests/data/surface/fetrig_gas_profiles"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("profiles", "source_sha256")}, indent=2
        )
    )
