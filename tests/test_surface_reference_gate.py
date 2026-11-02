"""The reference suite must not advertise a failed local gate or stale output."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_surface_reference_suite as suite


def test_strict_gate_does_not_turn_global_pass_into_local_pass(monkeypatch, tmp_path):
    def successful_child(command, **kwargs):
        path = Path(command[command.index("--output") + 1])
        if path.stem == "pr":
            value = {"passed": True, "section": "all", "checks": 148}
        elif path.stem == "thermal_shock":
            value = {"passed": True, "cases": [{}, {}], "stagnation_temperature_gate_passed": False}
        else:
            value = {"cases": [{}, {}]}
        path.write_text(json.dumps(value))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", successful_child)
    monkeypatch.setattr(
        sys, "argv", ["suite", "--output-dir", str(tmp_path), "--require-local-temperature"]
    )
    with pytest.raises(SystemExit) as error:
        suite.main()
    assert error.value.code == 1
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["standard_gates_passed"]
    assert not report["passed"] and not report["strict_local_temperature_passed"]


def test_missing_outputs_do_not_reuse_stale_success(monkeypatch, tmp_path):
    for name in ("pr", "isothermal_acoustics", "thermal_acoustics", "thermal_shock"):
        (tmp_path / (name + ".json")).write_text(
            json.dumps(
                {"passed": True, "cases": [{}, {}], "stagnation_temperature_gate_passed": True}
            )
        )
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(sys, "argv", ["suite", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        suite.main()
    assert error.value.code == 1
    report = json.loads((tmp_path / "summary.json").read_text())
    assert not report["passed"] and not report["standard_gates_passed"]
    assert all(not r["artifact_valid"] for r in report["jobs"])


def test_extended_suite_rejects_failed_matrix_even_if_original_shock_passes(monkeypatch, tmp_path):
    def child(command, **kwargs):
        path = Path(command[command.index("--output") + 1])
        if path.stem == "pr":
            value = {"passed": True, "section": "all", "checks": 148}
        elif path.stem == "thermal_shock":
            value = {"passed": True, "cases": [{}, {}], "stagnation_temperature_gate_passed": True}
        elif path.stem == "thermal_shock_matrix":
            value = {"passed": False, "matrix": "all", "cases": [{}] * 12}
        else:
            value = {"cases": [{}, {}]}
        path.write_text(json.dumps(value))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", child)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "suite",
            "--output-dir",
            str(tmp_path),
            "--require-local-temperature",
            "--include-shock-matrix",
        ],
    )
    with pytest.raises(SystemExit) as error:
        suite.main()
    assert error.value.code == 1
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["shock_matrix_included"]
    assert not report["passed"]


def test_optional_compressor_gate_cannot_be_masked_by_other_successes(monkeypatch, tmp_path):
    def child(command, **kwargs):
        path = Path(command[command.index("--output") + 1])
        if path.stem == "pr":
            value = {"passed": True, "section": "all", "checks": 148}
        elif path.stem == "thermal_shock":
            value = {"passed": True, "cases": [{}, {}], "stagnation_temperature_gate_passed": True}
        elif path.stem == "compressor_train":
            value = {"passed": False, "blowdown": [{}] * 3, "equilibria": [{}] * 2}
        else:
            value = {"cases": [{}, {}]}
        path.write_text(json.dumps(value))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", child)
    monkeypatch.setattr(
        sys, "argv", ["suite", "--output-dir", str(tmp_path), "--include-compressor"]
    )
    with pytest.raises(SystemExit) as error:
        suite.main()
    assert error.value.code == 1
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["compressor_included"] and not report["passed"]
