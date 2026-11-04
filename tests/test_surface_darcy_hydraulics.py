"""Independent laminar/Colebrook limits and Darcy-vessel coupling."""

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from scipy.optimize import brentq

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).parent)]
from test_surface_stratified_pr_network import setup

from surface.darcy_hydraulics import darcy_pressure_loss
from surface.stratified_pr_network import StratifiedPRNetwork


@pytest.mark.parametrize("v", [0.0, -0.01, 0.01, 1.0, -10.0])
def test_friction_is_finite_dissipative_and_has_finite_derivative(v):
    value = torch.tensor(v, dtype=torch.float64)
    loss, re = darcy_pressure_loss(value, 1000.0, 10.0, 0.05, 0.001, 1e-5)
    derivative = torch.func.jacfwd(
        lambda x: darcy_pressure_loss(x, 1000.0, 10.0, 0.05, 0.001, 1e-5)[0]
    )(value)
    assert torch.isfinite(loss) & torch.isfinite(re) & torch.isfinite(derivative)
    assert float(loss) * v >= 0
    assert derivative > 0
    if re < 2000:
        assert float(loss) == pytest.approx(32 * 0.001 * 10 / 0.05**2 * v)


@pytest.mark.parametrize("re", [1e4, 1e5, 1e6])
def test_turbulent_approximation_against_implicit_colebrook(re):
    rho, diameter, mu, length, roughness = 1000.0, 0.05, 0.001, 10.0, 1e-5
    v = re * mu / (rho * diameter)
    loss, _ = darcy_pressure_loss(
        torch.tensor(v, dtype=torch.float64), rho, length, diameter, mu, roughness
    )
    factor = brentq(
        lambda f: (
            1 / math.sqrt(f)
            + 2 * math.log10(roughness / (3.7 * diameter) + 2.51 / (re * math.sqrt(f)))
        ),
        0.005,
        0.1,
    )
    reference = factor * length / diameter * rho * v * v / 2
    assert float(loss) == pytest.approx(reference, rel=0.02)


@pytest.mark.parametrize("reverse", [False, True])
def test_coupled_laminar_connector_matches_poiseuille(reverse):
    old, state = setup(reverse=reverse)
    link = replace(
        old.connections[0],
        volume_conductance_m3_s_pa=0.0,
        hydraulic_model="darcy_weisbach",
        length_m=100.0,
        dynamic_viscosity_pa_s=0.01,
        maximum_velocity_m_s=10.0,
    )
    model = StratifiedPRNetwork(old.vessels, [link])
    final, report = model._step(model.initialize(dict(state.vessels)), 0.005, model._commands(None))
    transfer = report["transfers"][0]
    h = transfer["hydraulics"]
    assert h["reynolds"] < 2000
    expected = (
        math.pi
        * link.diameter_m**4
        * h["pressure_drop"]
        / (128 * link.dynamic_viscosity_pa_s * link.length_m)
    )
    assert transfer["volume_rate_m3_s"] == pytest.approx(expected, rel=1e-7)
    assert h["friction_loss"] == pytest.approx(h["pressure_drop"], abs=1e-5)
    assert (transfer["volume_rate_m3_s"] < 0) == reverse
    assert abs(report["energy_residual_j"]) < 1e-6
    assert final.time_s == 0.005


def test_isolation_does_not_equalize_pressure():
    old, state = setup()
    link = replace(
        old.connections[0],
        volume_conductance_m3_s_pa=0.0,
        hydraulic_model="darcy_weisbach",
        length_m=100.0,
        dynamic_viscosity_pa_s=0.01,
        maximum_velocity_m_s=10.0,
    )
    model = StratifiedPRNetwork(old.vessels, [link])
    initial = model.initialize(dict(state.vessels))
    final, report = model._step(initial, 0.01, model._commands({"link": 0.0}))
    assert report["transfers"][0]["volume_rate_m3_s"] == 0.0
    assert report["transfers"][0]["hydraulics"]["pressure_drop"] > 9000
    assert (
        dict(final.vessels)["a"].column.component_moles
        == dict(state.vessels)["a"].column.component_moles
    )
    with pytest.raises(ValueError, match="binary isolation"):
        model.advance(initial, 0.01, openings={"link": 0.5})


def test_darcy_cross_derivatives_match_finite_difference_away_from_reversal():
    old, state = setup()
    link = replace(
        old.connections[0],
        volume_conductance_m3_s_pa=0.0,
        hydraulic_model="darcy_weisbach",
        length_m=100.0,
        dynamic_viscosity_pa_s=0.01,
        maximum_velocity_m_s=10.0,
    )
    m = StratifiedPRNetwork(old.vessels, [link])
    states = dict(state.vessels)
    y, _lo, _hi, evaluate, _ = m._problem(states, m._seeds(states), 0.005, 4, m._commands(None))
    y[-1] = 0.01
    jac = torch.func.jacfwd(lambda x: evaluate(x)[0])(y)
    direction = torch.linspace(-0.01, 0.01, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert torch.isfinite(jac).all()
    assert (jac @ direction - fd).abs().max() < 1e-6


def test_legacy_linear_serialization_keeps_its_physical_identity():
    from surface.operation import _digest

    m, _s = setup()
    payload = m.to_dict()
    assert "hydraulic_model" not in payload["connections"][0]
    assert "length_m" not in payload["connections"][0]
    assert _digest(payload) == m.model_id
    assert StratifiedPRNetwork.from_dict(payload).model_id == m.model_id


def test_coupled_turbulent_connector_passes_independent_pr_audit():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_stratified_pr_network import audit

    m, s = setup()
    link = replace(
        m.connections[0],
        volume_conductance_m3_s_pa=0.0,
        hydraulic_model="darcy_weisbach",
        length_m=100.0,
        dynamic_viscosity_pa_s=1e-5,
        roughness_m=1e-5,
        maximum_velocity_m_s=10.0,
    )
    m = StratifiedPRNetwork(m.vessels, [link])
    s = m.initialize(dict(s.vessels))
    final, step = m._step(s, 0.001, m._commands(None))
    assert step["transfers"][0]["hydraulics"]["reynolds"] > 4000
    compiled = m.to_dict()
    config = {
        "vessels": {n: {"model": v} for n, v in compiled["vessels"].items()},
        "connections": compiled["connections"],
        "schedule": [{"duration_s": 0.001}],
    }
    payload = {
        "config": config,
        "initial": s.to_dict(),
        "intervals": [{**step, "state": final.to_dict(), "steps": [step]}],
    }
    assert audit(config, payload)["accepted"]
