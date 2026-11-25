"""Mixed audit checks real fallback records and continuity across reports."""

import copy
import json

import pytest
from audit_pr_controlled_mixed import audit
from run_pr_controlled_surface import run
from test_surface_pr_surface_control import controlled

from surface.liquid import PropertyModelError
from surface.pr_surface_collocation import PRSurfaceCollocation


@pytest.fixture
def mixed(monkeypatch):
    model, state = controlled(feedback=True)
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [
                    {"duration_s": 1e-7},
                    {"duration_s": 1e-7, "openings": {"port": False}},
                    {"duration_s": 1e-7},
                ],
            }
        )
    )
    original, calls = PRSurfaceCollocation._step, 0

    def once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PropertyModelError("Exercise the real BE phase/event fallback")
        return original(*args, **kwargs)

    monkeypatch.setattr(PRSurfaceCollocation, "_step", once)
    return config, json.loads(json.dumps(run(config, integrator="sdirk2-common")))


def test_independent_mixed_audit_accepts_actual_common_and_fallback(mixed):
    result = audit(*mixed)
    assert result["accepted"]
    assert result["checked_records"] == {
        "sdirk2-common": 2,
        "controlled_backward_euler_fallback": 2,
    }


@pytest.mark.parametrize("change", ["missing_step", "endpoint", "heat", "clock", "control"])
def test_independent_mixed_audit_rejects_corruption(mixed, change):
    config, report = mixed
    bad = copy.deepcopy(report)
    if change == "missing_step":
        bad["intervals"][1]["steps"].clear()
    elif change == "endpoint":
        bad["intervals"][1]["state"]["positions"]["v"] += 0.001
    elif change == "heat":
        bad["total_heat_j"] += 0.01
    elif change == "clock":
        bad["intervals"][1]["steps"][0]["dt_s"] *= 2
    else:
        bad["intervals"][1]["steps"][0]["plant"]["transport"]["stage_couplings"][0]["controls"][
            "velocities"
        ][0] += 1
    with pytest.raises((ValueError, AssertionError)):
        audit(config, bad)
