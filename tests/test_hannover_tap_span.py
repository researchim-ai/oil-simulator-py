"""Keep conditional rig reconstruction distinct from physical qualification."""

import pytest
from check_hannover_tap_span import run_span
from validate_multiphase_experiments import ROOT


def test_complete_conditional_span_preserves_original_increment_and_negative_result():
    report = run_span(ROOT / "tests/data/surface/hannover_countercurrent")
    assert report["summary"]["dry_points"] == 13
    assert report["summary"]["wet_points"] == 16
    assert report["summary"]["wet_increment_mae_pa"] == pytest.approx(34.955090001, abs=1e-6)
    assert report["summary"]["positive_additional_loss_can_match_count"] == 0
    assert report["fully_identified_rig"] is False
    assert report["experimentally_qualified"] is False
    assert report["fit_performed"] is False
    assert len(report["missing_identification"]) > 0
    for row in report["dry"]:
        terms = row["pressure_terms_pa"]
        expected = terms["distributed_friction_pa"] + terms["minor_losses_pa"]
        assert row["prediction_pa"] == pytest.approx(expected, abs=1e-10)
