"""Only the discarded split probe borrows certified slow-plant quadrature."""

import json
from pathlib import Path

import pytest

from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState


def fixture():
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/surface_pr_integrated_facility_subcycled.json"
        ).read_text()
    )
    m = PRControlledSurface.from_dict(config["model"])
    return m, PRControlledSurfaceState.from_dict(config["initial"]["state"])


def test_subcycled_trial_saves_one_global_solve_without_borrowing_accepted_certificates(
    monkeypatch,
):
    m, s = fixture()
    calls = []
    original = m.slow.plant.transport._step

    depth = 0

    def counted(*args, **kwargs):
        nonlocal depth
        # Count requested physical steps, excluding internal active-set retries.
        if depth == 0:
            calls.append(True)
        depth += 1
        try:
            return original(*args, **kwargs)
        finally:
            depth -= 1

    monkeypatch.setattr(m.slow.plant.transport, "_step", counted)
    m._share_probe_quadrature = False
    old = m.advance(s, 1e-8)
    oldcount = len(calls)
    calls.clear()
    m._share_probe_quadrature = True
    new = m.advance(s, 1e-8)
    assert oldcount == 6 and len(calls) == 5
    a = PRControlledSurfaceState.from_dict(old["state"])
    b = PRControlledSurfaceState.from_dict(new["state"])
    assert m._error(a, b) < 1e-5
    for step in new["steps"]:
        stages = step["local_subcycling"]["stages"]
        assert [x["operator"] for x in stages] == ["fast", "slow", "fast"]
        for physical in stages[1]["report"]["plant"]["steps"]:
            assert not physical.get("quadrature_probe", False)
            assert 0 <= physical["quadrature_error"] <= 1


@pytest.mark.parametrize(
    "second_order,error,probe",
    [
        (8, 0.1, False),
        (4, None, False),
        (4, True, False),
        (4, float("nan"), False),
        (4, 2.0, False),
        (4, 0.1, True),
    ],
)
def test_missing_mixed_or_probe_certificates_require_original_refinement(
    monkeypatch, second_order, error, probe
):
    m, s = fixture()
    calls = []

    def staged(old, dt, *args, **kwargs):
        calls.append(kwargs)
        record = {
            "quadrature_order": 4 if len(calls) == 1 else second_order,
            "quadrature_error": 0.1 if len(calls) == 1 else error,
            "quadrature_probe": probe,
        }
        return old, {
            "local_subcycling": {"stages": [{}, {"report": {"plant": {"steps": [record]}}}, {}]}
        }

    monkeypatch.setattr(m, "_step", staged)
    m._temporal_trial(s, 1e-8, 5e-9, 5e-9, {}, {}, {})
    assert len(calls) == 3 and all(not kw for kw in calls)
