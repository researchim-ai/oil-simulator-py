"""Physics selection must never silently downgrade a requested study."""

import gzip
import json
from pathlib import Path

import pytest
import run_engineering_project as runner

from simulator.engineering_modes import load_case, read_json

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/engineering_project_example.json"


def project(tmp_path, change=None):
    value = json.loads(EXAMPLE.read_text())
    value["root"] = str(ROOT)
    if change:
        change(value)
    path = tmp_path / "project.json"
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize(
    "case,unit", [("development", "day"), ("linepack", "s"), ("controlled_plant", "s")]
)
def test_plan_resolves_without_mutating_inputs(case, unit):
    plan = load_case(EXAMPLE, case)
    before = plan.fingerprint
    request = plan.request()
    request.clear()
    assert plan.request()
    assert plan.fingerprint == before
    assert plan.describe()["simulation_horizon"]["unit"] == unit
    plan.verify_sources()


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda c: c.update(mode="transient"), "never silently downgraded"),
        (lambda c: c.update(required_features=["acoustic_waves"]), "cannot provide"),
        (
            lambda c: c.update(options={"duration_days": True, "report_days": 1}),
            "positive and finite",
        ),
        (lambda c: c.update(options={"duration_s": 1}), "forecast options"),
    ],
)
def test_invalid_physics_rejected_before_execution(tmp_path, monkeypatch, change, match):
    path = project(tmp_path, lambda p: change(p["cases"]["development"]))
    monkeypatch.setattr(runner, "_execute", lambda *a: pytest.fail("solver must not run"))
    output = tmp_path / "result"
    with pytest.raises(ValueError, match=match):
        runner.run_project(path, "development", output)
    assert not output.exists()


def test_source_guard_and_independent_plans(tmp_path):
    path = project(tmp_path)
    plan = load_case(path, "linepack")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="changed after planning"):
        plan.verify_sources()


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_strict_json(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        read_json(path)


def test_requested_pi_requires_controller_in_input(tmp_path):
    cfg = json.loads((ROOT / "configs/surface_pr_controlled_surface.json").read_text())
    cfg["model"]["controllers"] = []
    input_path = tmp_path / "plant.json"
    input_path.write_text(json.dumps(cfg))
    path = project(tmp_path, lambda p: p["cases"]["controlled_plant"].update(input=str(input_path)))
    with pytest.raises(ValueError, match="no controllers"):
        load_case(path, "controlled_plant")


def test_rejected_audit_preserves_evidence_without_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "_execute", lambda *a: ({}, {"value": 1}, lambda raw: {"accepted": False})
    )
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="audit rejected"):
        runner.run_project(EXAMPLE, "linepack", output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["stage"] == "numerical_audit"
    assert not (output / "summary.json").exists()
    with gzip.open(output / "trajectory.json.gz", "rt") as stream:
        assert json.load(stream)["result"] == {"value": 1}


def test_changed_project_cannot_publish_success(tmp_path, monkeypatch):
    path = project(tmp_path)

    def execute(*args):
        path.write_text(path.read_text() + "\n")
        return {}, {}, lambda raw: {"accepted": True}

    monkeypatch.setattr(runner, "_execute", execute)
    output = tmp_path / "changed"
    with pytest.raises(RuntimeError, match="changed after planning"):
        runner.run_project(path, "linepack", output)
    status = json.loads((output / "status.json").read_text())
    assert status["stage"] == "source_guard"
    assert status["status"] == "failed"
    assert not (output / "summary.json").exists()


def test_output_never_overwrites_and_gpu_requires_explicit_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        runner.run_project(EXAMPLE, "linepack", tmp_path / "gpu", device="cuda")
    with pytest.raises(FileExistsError):
        runner.run_project(EXAMPLE, "linepack", tmp_path)


@pytest.mark.parametrize("case", ["development", "linepack", "controlled_plant"])
def test_existing_physics_runs_with_independent_audit(tmp_path, case):
    output = tmp_path / case
    result = runner.run_project(EXAMPLE, case, output)
    assert result["status"] == "complete"
    assert result["numerical_audit"]["accepted"]
    assert result["sources_unchanged"]
    assert result["physical_calibration"] == "not_established_by_this_run"
    with gzip.open(output / "trajectory.json.gz", "rt") as stream:
        raw = json.load(stream)
    assert raw["case"]["fingerprint"] == result["fingerprint"]
    assert raw["effective_request"]
