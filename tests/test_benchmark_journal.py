"""An interrupted or invalid write cannot manufacture completed evidence."""

import json

import pytest
from benchmark_journal import BenchmarkJournal, atomic_json


def test_partial_series_retains_protocol_and_certificate_without_claiming_completion(tmp_path):
    directory = tmp_path / "run"
    journal = BenchmarkJournal(directory, {"source_sha256": {"solver.py": "abc"}})
    journal.checkpoint([{"pair": 0}], [], {"trajectory": {"audited_steps": 2}})
    assert json.loads((directory / "protocol.json").read_text())["source_sha256"] == {
        "solver.py": "abc"
    }
    progress = json.loads((directory / "progress.json").read_text())
    assert progress["status"] == "incomplete"
    assert progress["audit_certificates"]["trajectory"]["audited_steps"] == 2
    assert not (directory / "summary.json").exists()
    with pytest.raises(FileExistsError):
        BenchmarkJournal(directory, {"source_sha256": {"solver.py": "different"}})
    journal.complete({"runs": progress["runs"]})
    assert json.loads((directory / "summary.json").read_text())["status"] == "complete"


def test_invalid_json_preserves_previous_checkpoint_and_cleans_temporary_file(tmp_path):
    path = tmp_path / "progress.json"
    atomic_json(path, {"completed_runs": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"elapsed": float("nan")})
    assert json.loads(path.read_text()) == {"completed_runs": 1}
    assert list(tmp_path.iterdir()) == [path]
