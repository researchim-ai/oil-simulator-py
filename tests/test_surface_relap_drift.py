"""RELAP-7 eqs. 585--590 and mass-average drift, independently evaluated."""

import math

import pytest
import torch

from surface.drift_flux_closure import RELAP7VerticalDriftClosure, drift_closure_from_dict


@pytest.mark.parametrize("regime", ["dispersed_bubbly", "cap_slug"])
@pytest.mark.parametrize("direction", [-1, 1])
def test_relap_parameters_and_mass_average_flow_match_published_equations(regime, direction):
    closure = RELAP7VerticalDriftClosure(regime, 0.072, 0.001)
    rho = torch.tensor([1.2, 1000.0], dtype=torch.float64)
    context = {"diameter_m": 0.05, "axial_gravity_m_s2": direction * 9.80665}
    c0, vd = closure.parameters(rho, **context)
    scale = (0.072 * 9.80665 * 998.8 / 1000**2) ** 0.25
    expected_c0 = 1.2 - 0.2 * math.sqrt(1.2 / 1000)
    if regime == "dispersed_bubbly":
        expected_vd = math.sqrt(2) * scale
    else:
        capillary = math.sqrt(0.072 / (9.80665 * 998.8))
        dstar = min(30, 0.05 / capillary)
        nmu = 0.001 / math.sqrt(1000 * 0.072 * capillary)
        expected_vd = 0.0019 * dstar**0.809 * (1.2 / 1000) ** (-0.157) * nmu ** (-0.562) * scale
    assert float(c0) == pytest.approx(expected_c0, rel=1e-14)
    assert float(vd) == pytest.approx(direction * expected_vd, rel=1e-14)
    alpha = torch.tensor([0.1, 0.9], dtype=torch.float64)
    for u in [-1.0, 0.0, 1.0]:
        velocities, valid = closure.velocities(alpha, rho, rho.new_tensor(u), **context)
        assert bool(valid)
        assert float((alpha * rho * velocities).sum() / (alpha * rho).sum()) == pytest.approx(
            u, abs=1e-14
        )
        assert float(velocities[0]) == pytest.approx(
            float(c0 * (alpha * velocities).sum() + vd), abs=1e-14
        )
    jac = torch.func.jacfwd(lambda r: torch.stack(closure.parameters(r, **context)))(rho)
    eps = 1e-4
    direction_rho = rho.new_tensor([0.1, 10])
    fd = (
        torch.stack(closure.parameters(rho + eps * direction_rho, **context))
        - torch.stack(closure.parameters(rho - eps * direction_rho, **context))
    ) / (2 * eps)
    torch.testing.assert_close(jac @ direction_rho, fd, atol=1e-10, rtol=1e-7)


def test_relap_geometry_and_regime_limits_are_not_silently_extrapolated():
    from dataclasses import asdict

    from test_surface_compositional_separator import binary

    from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork

    c = RELAP7VerticalDriftClosure("dispersed_bubbly", 0.072, 0.001)
    assert drift_closure_from_dict(asdict(c)) == c
    with pytest.raises(ValueError, match="fraction"):
        RELAP7VerticalDriftClosure(
            "dispersed_bubbly", 0.072, 0.001, maximum_gas_volume_fraction=0.7
        )
    node = PREquilibriumNode(
        "pipe",
        math.pi * 0.05**2 / 4 * 0.1,
        "pipe_cell",
        length_m=0.1,
        diameter_m=0.05,
        drift_flux=c,
    )
    with pytest.raises(ValueError, match="vertical"):
        PREquilibriumTransportNetwork(binary(), [node], gravity_m_s2=9.80665)
    _, valid = c.velocities(
        torch.tensor([0.8, 0.2]),
        torch.tensor([1.2, 1000.0]),
        torch.tensor(0.0),
        diameter_m=0.05,
        axial_gravity_m_s2=9.80665,
    )
    assert not bool(valid)


@pytest.mark.parametrize("void_fraction", [0.025, 0.05, 0.075, 0.1, 0.2, 0.6])
@pytest.mark.parametrize("direction", [-1, 1])
def test_local_equilibrium_regime_transition_uses_inl_2017_temperature_limit(
    void_fraction, direction
):
    from dataclasses import asdict

    from surface.drift_flux_closure import RELAP7EquilibriumTransition

    law = RELAP7EquilibriumTransition(0.072, 0.001)
    rho = torch.tensor([1.2, 1000.0], dtype=torch.float64)
    volumes = rho.new_tensor([void_fraction, 1 - void_fraction])
    context = {"diameter_m": 0.05, "axial_gravity_m_s2": direction * 9.80665}
    c0, vd, weight = law.parameters(volumes, rho, **context)
    assert float(c0)==pytest.approx(1.2-.2*math.sqrt(1.2/1000))
    scale = (0.072 * 9.80665 * 998.8 / 1000**2) ** 0.25
    capillary = math.sqrt(0.072 / (9.80665 * 998.8))
    visc = 0.001 / math.sqrt(1000 * 0.072 * capillary)
    cs = (
        0.0019
        * min(30, 0.05 / capillary) ** 0.809
        * (1.2 / 1000) ** (-0.157)
        * visc ** (-0.562)
        * scale
    )
    expected_weight = max(0, 1 - void_fraction / 0.1)
    assert float(weight) == pytest.approx(expected_weight)
    assert float(vd) == pytest.approx(
        direction * (expected_weight * math.sqrt(2) * scale + (1 - expected_weight) * cs)
    )
    assert drift_closure_from_dict(asdict(law)) == law
    for u in [-1.0, 0.0, 1.0]:
        velocities, valid = law.velocities(volumes, rho, rho.new_tensor(u), **context)
        assert bool(valid)
        assert float((volumes * rho * velocities).sum() / (volumes * rho).sum()) == pytest.approx(
            u, abs=1e-13
        )
    if void_fraction != 0.1:
        jac = torch.func.jacfwd(
            lambda a: law.velocities(a, rho, rho.new_tensor(0.2), **context)[0]
        )(volumes)
        delta = rho.new_tensor([0.1, -0.1])
        eps = 1e-5
        finite = (
            law.velocities(volumes + eps * delta, rho, rho.new_tensor(0.2), **context)[0]
            - law.velocities(volumes - eps * delta, rho, rho.new_tensor(0.2), **context)[0]
        ) / (2 * eps)
        torch.testing.assert_close(jac @ delta, finite, rtol=1e-7, atol=1e-9)


def test_transition_is_live_in_pr_graph_and_independent_audit(monkeypatch):
    import json
    from pathlib import Path

    from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork

    base, _, initial = vertical_pipe()
    config = base.to_dict()
    for node in config["nodes"]:
        node["drift_flux"] = {
            "model": "relap7_vertical_transition",
            "surface_tension_n_m": 0.015,
            "liquid_viscosity_pa_s": 0.0002,
        }
    model = PREquilibriumTransportNetwork.from_dict(config)
    tie = sorted(
        model.fluid.flash_pt(1e6, 196.2, [0.5, 0.5])["phases"],
        key=lambda p: float(p["molar_volume_m3_mol"]),
    )
    for i, alpha in enumerate([0.05, 0.15]):
        vl, vg = [float(p["molar_volume_m3_mol"]) for p in tie]
        beta = alpha * vl / ((1 - alpha) * vg + alpha * vl)
        initial[str(i)]["mole_fractions"] = (
            (1 - beta) * tie[0]["mole_fractions"] + beta * tie[1]["mole_fractions"]
        ).tolist()
    request = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 1e-6}],
            }
        )
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    result = json.loads(json.dumps(run(request)))
    assert audit(request, result)["accepted"]


def vertical_pipe(regime="dispersed_bubbly"):
    from test_surface_compositional_separator import binary

    from surface.pr_equilibrium_transport import (
        PREquilibriumLink,
        PREquilibriumNode,
        PREquilibriumTransportNetwork,
    )

    f = binary()
    tie = sorted(
        f.flash_pt(1e6, 196.2, [0.5, 0.5])["phases"], key=lambda q: float(q["molar_volume_m3_mol"])
    )
    alpha = 0.1 if regime == "dispersed_bubbly" else 0.4
    vl, vg = [float(p["molar_volume_m3_mol"]) for p in tie]
    beta = alpha * vl / ((1 - alpha) * vg + alpha * vl)
    z = ((1 - beta) * tie[0]["mole_fractions"] + beta * tie[1]["mole_fractions"]).tolist()
    closure = RELAP7VerticalDriftClosure(
        regime,
        0.015,
        0.0002,
        maximum_gas_volume_fraction=0.2 if regime == "dispersed_bubbly" else 0.7,
    )
    nodes = [
        PREquilibriumNode(
            str(i),
            math.pi * 0.05**2 / 4 * 0.1,
            "pipe_cell",
            elevation_m=0.05 + i * 0.1,
            length_m=0.1,
            diameter_m=0.05,
            elevation_change_m=0.1,
            drift_flux=closure,
        )
        for i in range(2)
    ]
    m = PREquilibriumTransportNetwork(
        f, nodes, [PREquilibriumLink("internal", "0", "1", 0.05)], gravity_m_s2=9.80665
    )
    initial = {
        str(i): {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": z} for i in range(2)
    }
    return m, m.initialize(initial), initial


@pytest.mark.parametrize("regime", ["dispersed_bubbly", "cap_slug"])
def test_relap_closure_is_live_in_full_pr_transport_jacobian(regime):
    import json

    m, state, _ = vertical_pipe(regime)
    seeds = [m._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, ev, _, _, _ = m._problem(state, seeds, 1e-7, {"internal": True}, [0.0] * 2)
    jac = torch.func.jacfwd(lambda z: ev(z)[0])(y)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (ev(y + eps * direction)[0] - ev(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-5, atol=1e-8)
    new, report = m._step(state, 1e-7, {"internal": True}, [0.0] * 2, [196.2] * 2)
    assert max(map(abs, report["conservation_residuals"])) < 1e-6
    type(m).from_dict(json.loads(json.dumps(m.to_dict()))).inspect(new)
