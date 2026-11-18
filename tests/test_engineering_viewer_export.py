"""Postprocessing must preserve actual frames and reject unqualified sources."""

import copy
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_engineering_viewer import build, validate_topology

RUN = ROOT / "benchmarks/reports/surface_pipe_value_only/month"


@pytest.fixture(scope="module")
def archive():
    with gzip.open(RUN / "trajectory.json.gz", "rt") as f:
        return json.load(f)


def test_frames_components_and_field_order_are_preserved(archive):
    out = build(RUN, max_frames=12)
    original = archive["result"]
    assert out["frames"][0]["source_step"] == 0
    assert out["frames"][-1]["source_step"] == len(original["accepted_substeps"]) - 1
    for frame in out["frames"]:
        source = original["accepted_substeps"][frame["source_step"]]
        assert frame["time_days"] == source["time_days"]
        assert frame["wells"] == source["surface"]["wells"]
        assert (
            frame["facility"]["measurements"]
            == source["facility_coupling"]["facility"]["measurements"]
        )
    assert out["field"]["pressure_pa"] == original["final_pressure_pa"]
    assert out["field"]["s_g"] == original["final_s_g"]
    assert out["field"]["time_days"] == 30
    assert out["provenance"]["accepted_steps"] == 1128
    assert "solver_log" not in out
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("failure", ["checksum", "incomplete", "mass"])
def test_reject_unqualified_source(tmp_path, failure):
    for name in ("protocol.json", "summary.json", "progress.json", "trajectory.json.gz"):
        shutil.copyfile(RUN / name, tmp_path / name)
    if failure == "incomplete":
        p = tmp_path / "progress.json"
        d = json.loads(p.read_text())
        d["status"] = "running"
        p.write_text(json.dumps(d))
    elif failure == "checksum":
        with (tmp_path / "trajectory.json.gz").open("ab") as f:
            f.write(b"corrupt")
    else:
        p = tmp_path / "trajectory.json.gz"
        with gzip.open(p, "rt") as f:
            d = json.load(f)
        d["result"]["withdrawn_component_mass_kg"][0] += 1000
        with gzip.open(p, "wt") as f:
            json.dump(d, f)
        summary = json.loads((tmp_path / "summary.json").read_text())
        summary["trajectory_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises((ValueError, AssertionError)):
        build(tmp_path, 2)


def test_do_not_draw_unsupported_equipment_as_two_stage_plant(archive):
    config = copy.deepcopy(archive["config"])
    config["simulation"]["coupled_facility"]["facility"]["separator_stages"].pop()
    with pytest.raises(ValueError, match="two separator stages"):
        validate_topology(config)
