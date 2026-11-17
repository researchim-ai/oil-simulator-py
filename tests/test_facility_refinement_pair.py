"""A short, incomplete or unrefined run cannot masquerade as long qualification."""

import gzip
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_facility_refinement_pair import completed, qualify, stricter

CANDIDATE = ROOT / "benchmarks/reports/surface_time_accuracy/depletion_fine_short"
REFERENCE = ROOT / "benchmarks/reports/surface_time_accuracy/depletion_reference_short"
BUDGETS = json.loads(
    (ROOT / "benchmarks/reports/surface_time_accuracy/field_budgets.json").read_text()
)


def test_actual_guarded_short_refinement_passes_its_declared_scope_only():
    r = qualify(CANDIDATE, REFERENCE, BUDGETS, required_days=0.02)
    assert r["accepted"] and r["same_python_and_pvt"]
    assert r["accepted_stages"] == {"candidate": 130, "reference": 248}
    with pytest.raises(ValueError, match="required horizon"):
        qualify(CANDIDATE, REFERENCE, BUDGETS)


def test_equal_resolution_cannot_be_called_refinement():
    with pytest.raises(ValueError, match="stricter"):
        qualify(CANDIDATE, CANDIDATE, BUDGETS, required_days=0.02)


@pytest.mark.parametrize("days", [True, 0, -1, float("inf"), float("nan")])
def test_invalid_horizon_is_rejected_before_reading_artifacts(tmp_path, days):
    with pytest.raises(ValueError, match="qualification horizon"):
        completed(tmp_path, days)


def fixed_policy(maximum, initial):
    return {
        "config": {
            "simulation": {
                "max_time_step_days": maximum,
                "initial_time_step_days": initial,
                "coupled_facility": {},
            }
        }
    }


@pytest.mark.parametrize("maximum", [True, 0, float("nan"), float("inf")])
def test_invalid_maximum_is_not_refinement(maximum):
    with pytest.raises(ValueError, match="maximum time step"):
        stricter(fixed_policy(1, 0.01), fixed_policy(maximum, 0.01))


def test_smaller_cap_cannot_hide_a_coarser_startup():
    with pytest.raises(ValueError, match="startup step"):
        stricter(fixed_policy(1, 0.001), fixed_policy(0.5, 0.01))
    assert stricter(fixed_policy(1, 1), fixed_policy(0.5, 1)) == "maximum_time_step"


@pytest.mark.parametrize(
    "mutation", ["failure", "incomplete", "checksum", "source", "pvt", "false_guard"]
)
def test_corrupted_completion_or_changed_numerics_cannot_qualify(tmp_path, mutation):
    target = tmp_path / "reference"
    shutil.copytree(REFERENCE, target)
    path = target / "summary.json"
    s = json.loads(path.read_text())
    if mutation == "failure":
        (target / "failure.json").write_text("{}")
    elif mutation == "incomplete":
        s["status"] = "running"
    elif mutation == "checksum":
        s["trajectory_sha256"] = "0" * 64
    elif mutation == "source":
        key = next(k for k in s["source_sha256"] if k.endswith("blackoil_fim.py"))
        s["source_sha256"][key] = "0" * 64
    elif mutation == "false_guard":
        s["sources_unchanged"] = "false"
    else:
        with gzip.open(target / "trajectory.json.gz", "rt") as f:
            key = json.load(f)["config"]["fluid"]["pvt_path"]
        s["source_sha256"][key] = "0" * 64
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError):
        qualify(CANDIDATE, target, BUDGETS, required_days=0.02)


def test_temporal_reference_can_add_bhp_control_but_cannot_drop_or_weaken_it():
    import copy
    a = {'config': {'simulation': {'coupled_facility': {'temporal_control': {}}}}}
    b = copy.deepcopy(a)
    b['config']['simulation']['coupled_facility']['temporal_control']['well_bhp_tolerance_pa'] = 100.
    assert stricter(a, b) == 'local_BE_defect_tolerances'
    for value in (None, 101.):
        c = copy.deepcopy(b)
        policy = c['config']['simulation']['coupled_facility']['temporal_control']
        policy.update(well_bhp_tolerance_pa=value, pressure_tolerance_pa=50.)
        with pytest.raises(ValueError, match='weaken temporal BHP'):
            stricter(b, c)


def test_reference_cannot_change_withdrawal_error_normalization():
    import copy
    a={'config':{'simulation':{'coupled_facility':{'temporal_control':{}}}}}
    b=copy.deepcopy(a)
    b['config']['simulation']['coupled_facility']['temporal_control'].update(
        withdrawal_error_scale='cumulative',pressure_tolerance_pa=50.)
    with pytest.raises(ValueError,match='withdrawal error scaling'):
        stricter(a,b)
