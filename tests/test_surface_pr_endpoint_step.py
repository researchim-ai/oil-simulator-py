"""A reporting boundary must not masquerade as a resolved physical timescale."""

from dataclasses import replace

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def closed(request):
    model = PREquilibriumTransportNetwork(
        binary(request.param),
        [PREquilibriumNode("closed", 0.001)],
        initial_step_s=0.01,
        minimum_step_s=1e-9,
        maximum_step_s=0.1,
    )
    state = model.initialize(
        {
            "closed": {
                "pressure_pa": 1e6,
                "temperature_k": 350.0,
                "mole_fractions": [1.0, 0.0],
            }
        }
    )
    return model, state


def fake_steps(model, monkeypatch):
    calls = []

    def step(state, dt, *args, **kwargs):
        calls.append(dt)
        return replace(state, time_s=state.time_s + dt), {
            "dt_s": dt,
            "ambient_heat_j": 0.0,
            "external_heat_j": 0.0,
            "external_work_j": 0.0,
            "entropy_production_j_k": 0.0,
        }

    monkeypatch.setattr(model, "_step", step)
    return calls


@pytest.mark.parametrize(
    "error,expected", [(0.0, 0.01), (0.249, 0.01), (0.25, 0.001), (0.9, 0.001)]
)
def test_only_low_error_endpoint_clip_preserves_proposal(closed, monkeypatch, error, expected):
    model, state = closed
    calls = fake_steps(model, monkeypatch)
    monkeypatch.setattr(model, "_error", lambda a, b: error)
    report = model.advance(state, 0.001)
    assert calls == [0.001, 0.0005, 0.0005]
    assert report["state"]["time_s"] == 0.001
    assert report["state"]["next_step_s"] == pytest.approx(expected)
    assert report["conservation_residuals"] == pytest.approx([0.0, 0.0, 0.0])


def test_rejection_discards_preclipping_proposal(closed, monkeypatch):
    model, state = closed
    calls = fake_steps(model, monkeypatch)
    errors = iter([2.0, 0.0, 0.0])
    monkeypatch.setattr(model, "_error", lambda a, b: next(errors))
    report = model.advance(state, 0.001)
    assert report["attempts"] == 3
    assert calls[0] == 0.001
    assert report["state"]["next_step_s"] == pytest.approx(0.00075)
    assert report["state"]["next_step_s"] < state.next_step_s


def test_unclipped_step_retains_original_growth_limit(closed, monkeypatch):
    model, state = closed
    fake_steps(model, monkeypatch)
    monkeypatch.setattr(model, "_error", lambda a, b: 0.0)
    report = model.advance(state, 0.01)
    assert report["state"]["next_step_s"] == pytest.approx(0.015)


def test_reference_policy_still_reproduces_endpoint_shrink(closed, monkeypatch):
    model, state = closed
    fake_steps(model, monkeypatch)
    monkeypatch.setattr(model, "_error", lambda a, b: 0.0)
    monkeypatch.setattr(model, "_preserve_endpoint_step", False)
    report = model.advance(state, 0.001)
    assert report["state"]["next_step_s"] == pytest.approx(0.0015)


def test_physical_closed_vessel_retains_hint_and_conservation(closed):
    model, state = closed
    report = model.advance(state, 0.001)
    assert report["state"]["next_step_s"] == 0.01
    assert report["state"]["time_s"] == 0.001
    assert max(map(abs, report["conservation_residuals"])) < 1e-8
