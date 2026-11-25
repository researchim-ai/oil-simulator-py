"""Benchmark orchestration must compare vessel maps and restore EOS selection."""

import json
from types import SimpleNamespace

import benchmark_pr_linepack_eos as timing
import pytest

from surface.pr_linepack import PRLinepackNetwork
from surface.stratified_pr_network import StratifiedPRNetwork


def test_benchmark_compares_states_and_preserves_operator_setting(tmp_path, monkeypatch):
    states = {"vessels": {}, "time_s": 1.0, "next_step_s": 1.0, "model_id": "fixture"}
    selections = []

    def run(*args, **kwargs):
        selections.append(StratifiedPRNetwork._batch_connector_eos)
        return {"intervals": [{"state": states, "steps": []}]}

    def error(left, right):
        assert isinstance(left, dict) and isinstance(right, dict)
        return 0.0

    monkeypatch.setattr(timing, "run", run)
    monkeypatch.setattr(timing, "audit", lambda *args: {"accepted": True})
    monkeypatch.setattr(
        PRLinepackNetwork,
        "from_dict",
        lambda *args, **kwargs: SimpleNamespace(network=SimpleNamespace(_error=error)),
    )
    original = StratifiedPRNetwork._batch_connector_eos
    output = tmp_path / "result"
    result = timing.benchmark({"model": {}, "schedule": [{"duration_s": 1.0}]}, output, repeats=2)
    assert selections == [False, True, True, False]
    assert len(result["samples"]) == 4
    assert result["accepted"]
    assert json.loads((output / "samples.json").read_text()) == result["samples"]
    assert StratifiedPRNetwork._batch_connector_eos is original


def test_benchmark_rejects_failed_audit_and_restores_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "run", lambda *args, **kwargs: {})
    monkeypatch.setattr(timing, "audit", lambda *args: {"accepted": False})
    original = StratifiedPRNetwork._batch_connector_eos
    with pytest.raises(ValueError, match="audit rejected"):
        timing.benchmark({"model": {}, "schedule": []}, tmp_path / "rejected")
    assert StratifiedPRNetwork._batch_connector_eos is original
    assert not (tmp_path / "rejected/summary.json").exists()
