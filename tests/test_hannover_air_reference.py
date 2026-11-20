"""Verify provenance, cell semantics and missing measurements in author data."""

from pathlib import Path

import pytest
from audit_hannover_air_reference import author_summary

DATA = Path(__file__).resolve().parents[1] / "tests/data/surface/hannover_countercurrent"


def test_author_cached_reference_is_read_without_reinterpreting_kelvin_label():
    metadata, rows = author_summary(DATA / "air_reference.xlsx")
    assert metadata["sheet"] == "Vergleich_Num_vs_Exp"
    assert len(rows) == 13
    assert rows[0]["q_m3_h"] == pytest.approx(140.57551844679483)
    assert rows[0]["rho_kg_m3"] == pytest.approx(1.2051031008283237)
    assert rows[0]["mu_pa_s"] == pytest.approx(1.8300797493057531e-5)
    assert rows[0]["author_simulation_pa"] == pytest.approx(42.9946)
    assert sum(r["measured_dp1_pa"] is None for r in rows) == 6
    assert all(r["measured_dp2_pa"] is not None for r in rows)
    assert all(a["q_m3_h"] < b["q_m3_h"] for a, b in zip(rows, rows[1:]))


def test_modified_reference_cannot_silently_enter_audit(tmp_path):
    p = tmp_path / "air_reference.xlsx"
    p.write_bytes((DATA / p.name).read_bytes() + b"changed")
    p.with_suffix(".json").write_text((DATA / "air_reference.json").read_text())
    with pytest.raises(ValueError, match="checksum"):
        author_summary(p)


def test_geometry_campaign_and_instrument_notes_are_explicit():
    metadata, _ = author_summary(DATA / "air_reference.xlsx")
    assert metadata["declared_geometry"]["pipe_diameter_m"] == pytest.approx(0.085)
    assert metadata["declared_geometry"]["lower_diameter_m"] == pytest.approx(0.1643)
    assert set(metadata["acquisition_start_dates"].values()) == {
        "2024-09-17",
        "2024-09-18",
        "2024-09-19",
    }
    assert "5 Pa" in metadata["measurement_notes"]["C65"]
    assert "5 Pa" in metadata["measurement_notes"]["C66"]
    assert "Flexschlauch" in metadata["measurement_notes"]["C70"]
    assert metadata["same_campaign_as_wet_data_confirmed"] is False
