"""A completed experimental comparison must not imply a passed physical gate."""

import json

import pytest
import validate_annular_film_experiment as validator


@pytest.mark.parametrize("accepted,report_only,code", [
    (False, False, 1), (False, True, 0), (True, False, 0),
])
def test_failed_qualification_is_nonzero_without_explicit_report_mode(
    tmp_path, monkeypatch, accepted, report_only, code
):
    result = {"accepted_as_validated_for_annular_countercurrent": accepted,
              "roughness_sensitivity": {}, "fit_performed": False}
    monkeypatch.setattr(validator, "validate", lambda *a, **kw: result)
    output = tmp_path / "report.json"
    args = ["--output", str(output), "--shear-velocity", "interface"]
    if report_only:
        args.append("--report-only")
    assert validator.main(args) == code
    assert json.loads(output.read_text()) == result
