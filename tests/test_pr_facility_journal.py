"""Durable accepted work independently audits and retains future commands."""

import json

import pytest
from check_pr_controlled_surface import audit
from run_pr_checkpointed_facility import accepted_prefix, journal_run, merge_operating_reports
from run_pr_controlled_surface import run
from test_surface_pr_subcycling import small_network


def test_journal_recovers_auditable_prefix_and_exact_remaining_schedule(tmp_path):
    _, _, model, state = small_network()
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [
                    {"duration_s": 2e-5, "targets": {"v": 1.0}},
                    {
                        "duration_s": 1e-5,
                        "targets": {"v": 1.0},
                        "openings": {"af": False, "fb": False},
                    },
                    {"duration_s": 1e-5, "targets": {"v": 1.0}},
                ],
            }
        )
    )
    output = tmp_path / "run.json"
    status = journal_run(config, output, budget_s=60)
    assert status["completed"] and status["sources_unchanged"]
    journal = output.with_suffix(".accepted.jsonl")
    report, remaining = accepted_prefix(journal)
    assert remaining is None
    assert audit(report["config"], report)["accepted"]
    full = json.loads(output.with_suffix(".trajectory.json").read_text())
    assert report["intervals"][-1]["state"] == full["intervals"][-1]["state"]
    lines = journal.read_text().splitlines(keepends=True)
    prefix = tmp_path / "prefix.jsonl"
    prefix.write_text("".join(lines[:2]) + '{"kind":"accepted"')
    partial, remaining = accepted_prefix(prefix)
    assert audit(partial["config"], partial)["accepted"]
    assert remaining["schedule"][1:] == config["schedule"][1:]
    assert remaining["schedule"][0]["duration_s"] == pytest.approx(1e-5)
    assert remaining["initial"]["state"] == partial["intervals"][-1]["state"]
    continued = json.loads(json.dumps(run(remaining)))
    merged = merge_operating_reports(config, partial, continued)
    assert audit(config, merged)["accepted"]
    assert merged["intervals"][-1]["state"] == full["intervals"][-1]["state"]
    with pytest.raises(FileExistsError):
        journal_run(config, output, budget_s=60)


def test_native_journal_preserves_auditable_restart_and_runtime_metadata(tmp_path):
    _, _, model, state = small_network()
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [{"duration_s": 1e-5, "targets": {"v": 1.0}}],
            }
        )
    )
    path = tmp_path / "native.json"
    status = journal_run(config, path, native_kernels=True, maximum_native_kernels=1, budget_s=120)
    assert status["completed"] and status["sources_unchanged"]
    assert status["native_kernels"] and not status["native_execution"]["disabled"]
    assert status["native_execution"]["compilations"] == 1
    assert status["native_execution"]["compiled_calls"] > 0
    report, remaining = accepted_prefix(path.with_suffix(".accepted.jsonl"))
    assert remaining is None and audit(config, report)["accepted"]
    full = json.loads(path.with_suffix(".trajectory.json").read_text())
    assert report["intervals"][-1]["state"] == full["intervals"][-1]["state"]
