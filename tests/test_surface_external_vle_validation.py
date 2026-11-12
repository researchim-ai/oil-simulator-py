"""Scientific input integrity and honest handling of failed model predictions."""

import hashlib
import json
from copy import deepcopy

import pytest
from import_may2015_vle import extract
from validate_may2015_vle import ROOT, run

DATA = ROOT / "tests/data/surface"


def test_frozen_nist_excerpt_reproduces_all_17_paired_measurements():
    manifest = json.loads((DATA / "may2015_manifest.json").read_text())
    excerpt = DATA / "may2015_thermoml_excerpt.json"
    assert hashlib.sha256(excerpt.read_bytes()).hexdigest() == manifest["excerpt_sha256"]
    actual = extract(excerpt)
    reference = json.loads((DATA / "may2015_methane_ethane.json").read_text())
    assert actual["points"] == reference["points"]
    assert actual["uncertainty_confidence_percent"] == 95
    assert actual["points"][0]["pressure_pa"] == 3949000
    assert actual["points"][0]["vapor_ethane_mole_fraction"] == 0.3336


def test_exact_coordinate_join_is_independent_of_vapor_row_order(tmp_path):
    raw = json.loads((DATA / "may2015_thermoml_excerpt.json").read_text())
    raw["PureOrMixtureData"][1]["NumValues"].reverse()
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(raw))
    result = extract(path)
    expected = json.loads((DATA / "may2015_methane_ethane.json").read_text())
    for a, b in zip(result["points"], expected["points"], strict=True):
        assert {k: v for k, v in a.items() if k != "vapor_index"} == {
            k: v for k, v in b.items() if k != "vapor_index"
        }


@pytest.mark.parametrize("change", ["component", "unit", "missing", "duplicate"])
def test_ambiguous_or_incomplete_measurements_are_rejected_not_silently_rescaled(change, tmp_path):
    raw = json.loads((DATA / "may2015_thermoml_excerpt.json").read_text())
    pressure, vapor = raw["PureOrMixtureData"]
    if change == "component":
        vapor["Property"][0]["Property-MethodID"]["RegNum"]["nOrgNum"] = 1
    elif change == "unit":
        pressure["Property"][0]["Property-MethodID"]["PropertyGroup"][
            "VaporPBoilingTAzeotropTandP"
        ]["ePropName"] = "Vapor or sublimation pressure, MPa"
    elif change == "missing":
        vapor["NumValues"].pop()
    else:
        vapor["NumValues"][-1] = deepcopy(vapor["NumValues"][0])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        extract(path)


def test_solver_failures_remain_in_the_experimental_denominator(monkeypatch):
    import validate_may2015_vle as module

    calls = []

    def unavailable(fluid, temperature, liquid):
        calls.append((temperature, liquid))
        return {"status": "no_unique_stable_prediction", "candidates": [], "attempts": []}

    monkeypatch.setattr(module, "bubble_prediction", unavailable)
    result = run(DATA / "may2015_methane_ethane.json", ROOT / "configs/validation_may2015_pr.json")
    points = json.loads((DATA / "may2015_methane_ethane.json").read_text())["points"]
    assert calls == [(p["temperature_k"], p["liquid_methane_mole_fraction"]) for p in points]
    assert result["experimental_points"] == 17 and result["predicted_points"] == 0
    assert result["failed_point_indices"] == list(range(17))
    assert result["pressure_mean_absolute_relative_error"] is None
    json.dumps(result, allow_nan=False)
