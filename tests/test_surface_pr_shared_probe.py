"""Discarded temporal probes may borrow only independently certified orders."""

import copy

import pytest
from test_surface_pr_surface_control import controlled

from surface.liquid import HydraulicConvergenceError
from surface.torch_nonlinear import TorchNewton


def test_five_global_solves_keep_both_accepted_halfsteps_certified(monkeypatch):
    model, state = controlled()
    original = TorchNewton.solve
    calls = []

    def counted(self, *args, **kwargs):
        # Count coupled plant systems, not the independent two-variable
        # thermodynamic sensor sampling solve that can run during controls.
        if args[1].numel() > 2:
            calls.append(args[1].numel())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TorchNewton, "solve", counted)
    model._share_probe_quadrature = False
    reference = model.advance(state, 1e-7, targets={"v": 0.5})
    assert len(calls) == 6
    calls.clear()
    model._share_probe_quadrature = True
    actual = model.advance(state, 1e-7, targets={"v": 0.5})
    assert len(calls) == 5
    assert actual["attempts"] == 1
    assert actual["state"] == reference["state"]
    assert actual["steps"] == reference["steps"]
    for half in actual["steps"]:
        assert len(half["plant"]["steps"]) == 1
        accepted = half["plant"]["steps"][0]
        assert not accepted.get("quadrature_probe", False)
        assert accepted["quadrature_error"] <= 1


def test_failed_direct_probe_falls_back_without_rejecting_accepted_halfsteps(monkeypatch):
    model, state = controlled()
    model._share_probe_quadrature = False
    expected = model.advance(state, 1e-7, targets={"v": 0.5})
    original = model._step
    probe_orders = []

    def recoverable(*args, **kwargs):
        if kwargs.get("_probe_quadrature_order") is not None:
            probe_orders.append(kwargs["_probe_quadrature_order"])
            raise HydraulicConvergenceError("Manufactured direct-probe failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_step", recoverable)
    model._share_probe_quadrature = True
    actual = model.advance(state, 1e-7, targets={"v": 0.5})
    assert probe_orders == [4]
    assert actual == expected


def test_rejected_full_probe_does_not_publish_precomputed_halfsteps(monkeypatch):
    model, state = controlled()
    before = state.to_dict()
    original = model._step

    def fail_large(old, dt, *args, **kwargs):
        if dt > 0.75e-7:
            raise HydraulicConvergenceError("Manufactured oversized full-step failure")
        return original(old, dt, *args, **kwargs)

    monkeypatch.setattr(model, "_step", fail_large)
    events = []
    result = model.advance(state, 1e-7, targets={"v": 0.5}, progress=events.append)
    assert any(e["event"] == "rejected" for e in events)
    assert result["attempts"] == 3
    assert len(result["steps"]) == 4
    assert result["state"]["plant"]["transport"]["time_s"] == 1e-7
    assert state.to_dict() == before


def test_different_halfstep_orders_require_original_coarse_refinement(monkeypatch):
    model, state = controlled()
    calls = []

    def staged(old, dt, *args, **kwargs):
        calls.append((dt, kwargs))
        order = 4 if len(calls) == 1 else 8
        return old, {"plant": {"steps": [{"quadrature_order": order, "quadrature_error": 0.1}]}}

    monkeypatch.setattr(model, "_step", staged)
    model._temporal_trial(state, 1e-7, 0.5e-7, 0.5e-7, {}, {}, {})
    assert [dt for dt, _ in calls] == [0.5e-7, 0.5e-7, 1e-7]
    assert all(not kwargs for _, kwargs in calls)


@pytest.mark.parametrize("error", [None, True, float("nan"), -1.0, 2.0])
def test_missing_or_invalid_spatial_certificate_is_not_reused(monkeypatch, error):
    model, state = controlled()
    calls = []

    def uncertified(old, dt, *args, **kwargs):
        calls.append(kwargs)
        return old, {"plant": {"steps": [{"quadrature_order": 4, "quadrature_error": error}]}}

    monkeypatch.setattr(model, "_step", uncertified)
    model._temporal_trial(state, 1e-7, 0.5e-7, 0.5e-7, {}, {}, {})
    assert len(calls) == 3 and all(not kw for kw in calls)


def test_unverified_probe_is_not_a_public_accepted_advance():
    model, state = controlled()
    before = state.to_dict()
    with pytest.raises(ValueError, match="only allowed for a controlled temporal probe"):
        model.plant.advance(state.plant, 1e-7, _probe_quadrature_order=4)
    for bad in (True, 2, 3, 32):
        with pytest.raises(ValueError, match="supported refined column order"):
            model.plant._step(
                state.plant, 1e-7, {"port": True}, {}, {}, {}, _probe_quadrature_order=bad
            )
    assert state.to_dict() == before


def test_independent_audit_rejects_a_probe_in_accepted_history():
    import json
    from pathlib import Path

    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/surface_pr_controlled_surface.json").read_text())
    config["schedule"] = config["schedule"][:1]
    report = run(config)
    assert audit(config, report)["accepted"]
    broken = copy.deepcopy(report)
    broken["intervals"][0]["steps"][0]["plant"]["steps"][0]["quadrature_probe"] = True
    with pytest.raises(ValueError, match="Unverified quadrature probe"):
        audit(config, broken)
    broken = copy.deepcopy(report)
    broken["intervals"][0]["steps"][0]["plant"]["steps"][0]["quadrature_error"] = None
    with pytest.raises(ValueError, match="valid quadrature certificate"):
        audit(config, broken)
