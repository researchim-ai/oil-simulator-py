"""BDF2 temporal convergence, independent port ledgers and restart transactions."""

import copy
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from check_pr_momentum_bdf import audit
from test_surface_pr_momentum_pipe import model

from surface.liquid import ConservationError
from surface.pr_momentum_bdf import PRMomentumBDF, PRMomentumBDFState
from surface.pr_momentum_pipe import PRDirichletBoundary

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def wave(device="cpu", **options):
    p = model(device, cells=4, length_m=10, initial_step_s=1e-4, **options)
    b = PRMomentumBDF(p)
    s = b.initialize(
        [1e6 + 100 * math.cos(2 * math.pi * (i + 0.5) / 4) for i in range(4)],
        350.0,
        [0.5, 0.5],
        0.0,
    )
    return b, s


def test_continuous_residual_is_inventory_derivative_not_a_time_step():
    b, s = wave()
    p = b.pipe
    y, _, _, continuous = p.continuous_problem(s.current, (None, None))
    _, details = continuous(y)
    row, _ = continuous(y, details["rate"])
    torch.testing.assert_close(row, torch.zeros_like(row), atol=1e-12, rtol=0)
    for h in [1e-4, 3e-4]:
        _, _, _, be = p._problem(s.current, h, (None, None))
        r, values = be(y)
        selected = p._support(s.current.component_moles).tolist() + [2, 3]
        expected = -h * details["rate"] / p._scales(p._conserved(s.current))
        torch.testing.assert_close(r, expected[:, selected].flatten(), atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(values["inventory"], details["inventory"])


@pytest.mark.parametrize("device", DEVICES)
def test_wave_balances_and_checkpoint_resume_keep_history(device):
    b, s = wave(device)
    initial = json.dumps(s.to_dict(), sort_keys=True)
    result = b.advance(s, 0.002)
    assert audit(result)["accepted"]
    assert json.dumps(s.to_dict(), sort_keys=True) == initial
    state = PRMomentumBDFState.from_dict(json.loads(json.dumps(result["state"])))
    first = b.advance(state, 0.001)
    restored = PRMomentumBDF.from_dict(json.loads(json.dumps(b.to_dict())), device=device)
    second = restored.advance(state, 0.001)
    assert not first["history_reset"]
    assert first["steps"][0]["point"]["order"] == 2
    np.testing.assert_allclose(
        first["final_summary"]["pressure_pa"],
        second["final_summary"]["pressure_pa"],
        atol=1e-8,
        rtol=0,
    )
    assert audit(second)["accepted"]
    assert max(map(abs, second["conservation_residuals"])) < 2e-6


def test_uniform_viscous_relaxation_has_second_order_temporal_convergence():
    errors = []
    for h in [0.01, 0.005, 0.0025]:
        p = model(
            cells=2,
            length_m=10,
            dynamic_viscosity_pa_s=0.01,
            initial_step_s=h,
            maximum_step_s=h,
            relative_tolerance=0.1,
            temperature_tolerance_k=1.0,
            velocity_tolerance_m_s=1.0,
        )
        b = PRMomentumBDF(p)
        s = b.initialize(1e6, 350.0, [0.5, 0.5], 1.0)
        rho = p.describe(s.current)["density_kg_m3"][0]
        expected = math.exp(-32 * 0.01 / (rho * p.diameter_m**2) * 0.05)
        r = b.advance(s, 0.05)
        assert audit(r)["accepted"]
        errors.append(abs(r["final_summary"]["velocity_m_s"][0] - expected))
        assert max(map(abs, r["conservation_residuals"])) < 2e-6
    assert errors[1] < errors[0] / 2.5
    assert errors[2] < errors[1] / 2.5


def test_pressure_switch_resets_history_and_signed_port_ledgers():
    p = model(cells=3, length_m=2, boundary_mode="ends", initial_step_s=1e-5)
    b = PRMomentumBDF(p)
    s = b.initialize(1e6, 350.0, [0.5, 0.5], 0.0)
    left = PRDirichletBoundary(1.001e6, 350.0, (0.5, 0.5), p.fluid.model_id)
    right = replace(left, pressure_pa=1e6)
    r = b.advance(s, 0.0005, left=left, right=right)
    assert audit(r)["accepted"]
    current = PRMomentumBDFState.from_dict(r["state"])
    changed = replace(left, pressure_pa=0.999e6)
    out = b.advance(current, 0.0005, left=changed, right=right)
    assert out["history_reset"]
    assert out["steps"][0]["point"]["order"] == 1
    assert any(step["point"]["left_exchange"][0] < 0 for step in out["steps"])
    assert audit(out)["accepted"]
    bad = copy.deepcopy(out)
    point = bad["steps"][-1]["point"]
    point["left_exchange"] = list(point["left_exchange"])
    point["left_exchange"][0] += 1e-3
    with pytest.raises((ValueError, AssertionError)):
        audit(bad)


def test_rejected_trial_does_not_publish_history(monkeypatch):
    b, s = wave()
    raw = b._error
    calls = []

    def reject_first(*args):
        calls.append(1)
        return math.inf if len(calls) == 1 else raw(*args)

    monkeypatch.setattr(b, "_error", reject_first)
    old = s.to_dict()
    r = b.advance(s, 0.002)
    assert r["rejected_steps"] >= 1
    assert s.to_dict() == old
    assert audit(r)["accepted"]


def test_tampered_history_exchange_is_rejected():
    b, s = wave()
    r = b.advance(s, 0.001)
    raw = json.loads(json.dumps(r["state"]))
    raw["history"][-1]["left_exchange"][0] += 1e-3
    with pytest.raises(ConservationError):
        b.advance(PRMomentumBDFState.from_dict(raw), 0.001)
