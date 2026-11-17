"""Reject incomplete trajectories, broken clocks and non-strict accepted steps."""
import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_norne_run import check_records, compare_same_source_summaries


@pytest.fixture
def records():
    names = ["day", "dt_days", "newton"]
    names += [prefix + phase for prefix in ("cnv_", "mb_", "mass_", "balance_rel_")
              for phase in ("oil", "water", "gas")]
    names = [n + "_kg" if n.startswith("mass_") else n for n in names]
    names += ["min_sw", "min_sg", "min_so"]
    rows = np.zeros(2, dtype=[(name, float) for name in names])
    rows["day"] = [.5, 1]
    rows["dt_days"] = [.5, .5]
    rows["newton"] = [3, 4]
    for phase in ("oil", "water", "gas"):
        rows["cnv_" + phase] = 1e-4
        rows["mb_" + phase] = 1e-8
        rows["mass_" + phase + "_kg"] = 100
        rows["balance_rel_" + phase] = 1e-7
    for phase in ("sw", "sg", "so"):
        rows["min_" + phase] = .1
    steps = np.array([[0, .5, 1, 3, 1], [.5, .5, 0, -1, 1], [.5, .5, 1, 4, 1]])
    meta = dict(status="done", days=1., failed_substeps=1, substeps=3, newton_iters=7,
                convergence=dict(tol_cnv=1e-3, tol_mb=1e-7, relaxed_max_pv_fraction=0))
    status = dict(status="complete", accepted=2, failed=1)
    manifest = dict(requested_days=1.)
    summary = {"TIME": np.array([0., .5, 1.]), "FOPT": np.array([0., 1., 2.])}
    return [meta, status, manifest, rows, steps, summary]


def test_records_with_a_rejected_attempt_are_valid(records):
    result = check_records(*records)
    assert result["accepted"] == 2
    assert result["failed_attempts"] == 1


@pytest.mark.parametrize("field,value", [("cnv_oil", 1.01e-3), ("mb_water", 1.01e-7),
                                        ("cnv_gas", np.nan), ("balance_rel_gas", 1.01e-4),
                                        ("mass_oil_kg", np.inf), ("min_sw", -1e-4)])
def test_physical_audit_rejects_invalid_values(records, field, value):
    records[3][field][-1] = value
    with pytest.raises(ValueError):
        check_records(*records)


def test_partial_run_cannot_pass_as_full_history(records):
    records[2]["requested_days"] = 3312
    with pytest.raises(ValueError, match="horizon"):
        check_records(*records)


def test_incomplete_audit_rejected_even_if_solver_finished(records):
    records[1]["status"] = "running"
    with pytest.raises(ValueError, match="Incomplete"):
        check_records(*records)


def test_step_clock_gap_rejected(records):
    records[3]["dt_days"][0] = .4
    with pytest.raises(ValueError, match="clock"):
        check_records(*records)


def test_false_report_between_accepted_times_rejected(records):
    records[5]["TIME"][1] = .4
    with pytest.raises(ValueError, match="no accepted"):
        check_records(*records)


def test_omitted_attempt_rejected(records):
    records[4] = records[4][[0, 2]]
    with pytest.raises(ValueError, match="Rejected-step"):
        check_records(*records)


def test_relaxed_profile_rejected(records):
    records[0]["convergence"]["relaxed_max_pv_fraction"] = .03
    with pytest.raises(ValueError, match="strict"):
        check_records(*records)


@pytest.fixture
def pair(tmp_path):
    paths = [tmp_path / "cpu", tmp_path / "gpu"]
    for path in paths:
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"source_files": {"core.py": "same"},
                                                       "input_files": {"case.DATA": "same"},
                                                       "reference_files": {"case.UNSMRY": "same"}}))
        np.savez(path / "summary.npz", TIME=[0., 1., 2.], FOPT=[0., 10., 20.],
                 FWCT=[np.nan, 0., .1])
    return paths


def test_same_source_comparison_reports_errors_and_undefined_samples(pair):
    np.savez(pair[1] / "summary.npz", TIME=[0., 1., 2.], FOPT=[0., 10.1, 20.], FWCT=[np.nan, 0., .1])
    result = compare_same_source_summaries(*pair)
    oil = next(v for v in result["vectors"] if v["vector"] == "FOPT")
    assert oil["peak_day"] == 1
    assert oil["max_absolute_difference"] == pytest.approx(.1)
    assert oil["final_signed_difference"] == 0
    assert all(v["finite_mask_mismatches"] == 0 for v in result["vectors"])


@pytest.mark.parametrize("section", ["source_files", "input_files", "reference_files"])
def test_cross_device_comparison_rejects_changed_sources_or_inputs(pair, section):
    path = pair[1] / "manifest.json"
    data = json.loads(path.read_text())
    data[section][next(iter(data[section]))] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Different frozen"):
        compare_same_source_summaries(*pair)


def test_cross_device_comparison_rejects_shifted_report_dates(pair):
    np.savez(pair[1] / "summary.npz", TIME=[0., .5, 2.], FOPT=[0., 10., 20.], FWCT=[np.nan, 0., .1])
    with pytest.raises(ValueError, match="clocks"):
        compare_same_source_summaries(*pair)
