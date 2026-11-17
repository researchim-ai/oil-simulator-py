"""Do not publish cross-device evidence with stale or corrupted artifacts."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_facility_device_pair import qualify


@pytest.fixture
def artifacts(tmp_path):
    paths = [tmp_path / "cpu", tmp_path / "cuda"]
    for path, device in zip(paths, ("cpu", "cuda"), strict=True):
        path.mkdir()
        data = b"preflight-only dummy trajectory; must not be parsed"
        (path / "trajectory.json.gz").write_bytes(data)
        (path / "summary.json").write_text(json.dumps({
            "status": "complete", "sources_unchanged": True, "device": device,
            "source_sha256": {"physics.py": "same"},
            "trajectory_sha256": hashlib.sha256(data).hexdigest(),
        }))
        (path / "progress.json").write_text(json.dumps({"status": "complete"}))
    return paths


def test_preexisting_failure_cannot_be_overridden_by_old_complete_summary(artifacts):
    (artifacts[1] / "failure.json").write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="failed forecast"):
        qualify(*artifacts, {})


def test_tampered_trajectory_rejected_before_numerical_comparison(artifacts):
    (artifacts[1] / "trajectory.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        qualify(*artifacts, {})


def test_unfinished_publication_cannot_pass_as_complete(artifacts):
    (artifacts[1] / "progress.json").write_text('{"status":"incomplete"}')
    with pytest.raises(ValueError, match="completion"):
        qualify(*artifacts, {})


def test_different_physics_source_rejected(artifacts):
    path = artifacts[1] / "summary.json"
    data = json.loads(path.read_text())
    data["source_sha256"]["physics.py"] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identical sources"):
        qualify(*artifacts, {})
