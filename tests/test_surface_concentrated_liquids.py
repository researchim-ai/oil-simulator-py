"""Dense immobile-drop closure, explicit carrier transition and live PR wall drag."""

import json
from dataclasses import asdict

import pytest
import torch

from surface.concentrated_liquid_drift import ConcentratedOilWaterClosure


def closure():
    return ConcentratedOilWaterClosure(
        gas_component_index=0,
        oil_component_index=1,
        water_component_index=2,
        droplet_diameter_m=1e-5,
        oil_viscosity_pa_s=0.005,
        water_viscosity_pa_s=0.001,
        surface_tension_n_m=0.03,
        phase_inversion_oil_fraction=0.5,
        transition_half_width=0.02,
        packing_fraction=0.64,
        intrinsic_viscosity=2.5,
        hindered_settling_exponent=4.65,
        gas_viscosity_pa_s=1.8e-5,
    )


@pytest.fixture(autouse=True)
def script_path(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.mark.parametrize("fraction", [0.01, 0.2, 0.49, 0.5, 0.51, 0.8, 0.99])
@pytest.mark.parametrize("bulk", [-0.2, 0.0, 0.2])
def test_dense_velocities_and_viscosity_match_independent_constraints(fraction, bulk):
    from reference_oil_water_drift import dense_rheology, velocities

    c = closure()
    alpha = torch.tensor([0.1, 0.9 * fraction, 0.9 * (1 - fraction)], dtype=torch.float64)
    rho = alpha.new_tensor([10.0, 800.0, 1000.0])
    x = torch.eye(3, dtype=alpha.dtype)
    velocity, valid = c.velocities(
        alpha,
        rho,
        alpha.new_tensor(bulk),
        phase_mole_fractions=x,
        diameter_m=0.05,
        axial_gravity_m_s2=9.80665,
        gravity_m_s2=9.80665,
    )
    assert bool(valid)
    expected = velocities(
        asdict(c), alpha.numpy(), rho.numpy(), x.numpy(), bulk, 0.05, 9.80665, 9.80665, 1e6
    )
    torch.testing.assert_close(velocity, alpha.new_tensor(expected), rtol=1e-12, atol=1e-14)
    viscosity, weight, _ = dense_rheology(asdict(c), alpha.numpy(), x.numpy())
    actual = c.rheology(alpha, x)
    assert float(actual["viscosity_pa_s"]) == pytest.approx(viscosity, rel=1e-12)
    assert float(actual["oil_continuous_weight"]) == pytest.approx(weight, abs=1e-14)


def test_richardson_zaki_volume_frame_and_dilute_viscosity_limit():
    c = closure()
    rho = torch.tensor([800.0, 1000.0], dtype=torch.float64)
    x = torch.eye(3, dtype=rho.dtype)[[1, 2]]
    alpha = rho.new_tensor([0.2, 0.8])
    v, ok = c.velocities(
        alpha,
        rho,
        rho.new_tensor(0.0),
        phase_mole_fractions=x,
        diameter_m=0.05,
        axial_gravity_m_s2=9.80665,
    )
    assert bool(ok)
    free = (1000 - 800) * 9.80665 * 1e-10 / (18 * 0.001)
    assert float(v[0] - (alpha * v).sum()) == pytest.approx(free * 0.8**4.65, rel=1e-12)
    dilute = c.rheology(rho.new_tensor([1e-6, 1 - 1e-6]), x)["viscosity_pa_s"]
    assert float(dilute) / 0.001 == pytest.approx(1 + 2.5e-6, abs=1e-11)


def setup(dense_limit=128):
    from test_surface_compositional_separator import ternary

    from surface.pr_equilibrium_transport import (
        PREquilibriumLink,
        PREquilibriumNode,
        PREquilibriumTransportNetwork,
    )

    f = ternary()
    law = closure()
    nodes = [
        PREquilibriumNode(
            str(i),
            0.0002,
            "pipe_cell",
            length_m=0.1,
            diameter_m=(0.0002 * 4 / (torch.pi * 0.1)) ** 0.5,
            drift_flux=law,
        )
        for i in range(2)
    ]
    m = PREquilibriumTransportNetwork(
        f,
        nodes,
        [PREquilibriumLink("link", "0", "1", nodes[0].diameter_m)],
        dense_limit=dense_limit,
    )
    initial = {
        n.name: {
            "pressure_pa": 1e6,
            "temperature_k": 350.0,
            "mole_fractions": [0.0, 0.1, 0.9],
            "velocity_m_s": 0.1,
        }
        for n in nodes
    }
    return m, m.initialize(initial), initial


@pytest.mark.parametrize("dense_limit", [1, 128])
def test_live_viscosity_drives_common_pr_momentum_and_energy(dense_limit):
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    m, s, initial = setup(dense_limit)
    end, r = m._step(s, 1e-6, m.default_openings, [0.0, 0.0], [350.0, 350.0])
    assert max(map(abs, r["conservation_residuals"])) < 1e-7
    assert float(m._state_data(0, end.nodes[0][1])["kinetic"]) < float(
        m._state_data(0, s.nodes[0][1])["kinetic"]
    )
    cfg = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 1e-6}],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]


def test_carrier_transition_changes_during_conservative_transport():
    from surface.pr_equilibrium_transport import PREquilibriumLink, PREquilibriumTransportNetwork

    m, _, initial = setup()
    tie = m.fluid.flash_pt(1e6, 350.0, [0.0, 0.1, 0.9])
    for name, fraction in (("0", 0.49), ("1", 0.8)):
        amounts = torch.stack(
            [
                m.fluid.tensor(fraction if int(p["mole_fractions"].argmax()) == 1 else 1 - fraction)
                / p["molar_volume_m3_mol"]
                for p in tie["phases"]
            ]
        )
        composition = (
            torch.stack([p["mole_fractions"] for p in tie["phases"]]) * amounts[:, None]
        ).sum(0) / amounts.sum()
        initial[name]["mole_fractions"] = composition.tolist()
    model = PREquilibriumTransportNetwork(
        m.fluid, m.nodes, [*m.links, PREquilibriumLink("return", "1", "0", m.nodes[0].diameter_m)]
    )
    old = model.initialize(initial)

    def weight(state):
        d = model._state_data(0, state.nodes[0][1])
        return float(
            model.nodes[0].drift_flux.rheology(
                d["phase_n"] * d["q"]["molar_volume_m3_mol"], d["x"]
            )["oil_continuous_weight"]
        )

    assert weight(old) < 0.5
    end, r = model._step(old, 0.1, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    assert weight(end) > 0.5
    assert max(map(abs, r["conservation_residuals"])) < 1e-7


def test_dense_live_jacobian_includes_viscosity_and_phase_volume_tangents():
    m, s, _ = setup()
    seeds = [m._seed(i, v) for i, (_, v) in enumerate(s.nodes)]
    y, _, _, evaluate, *_ = m._problem(s, seeds, 1e-5, m.default_openings, [0.0, 0.0])
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    eps = 0.001
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-5, atol=1e-8)


def test_dense_parameter_limits_do_not_silently_extend_the_law():
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(closure(), interface="clean")
    with pytest.raises(ValueError):
        replace(closure(), maximum_reynolds_number=0.5)
    with pytest.raises(ValueError):
        replace(closure(), phase_inversion_oil_fraction=0.9)
    with pytest.raises(ValueError):
        replace(closure(), packing_fraction=0.5)
