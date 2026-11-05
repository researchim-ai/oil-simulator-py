"""Independent creeping-drop limits and gas/oil/water kinematic constraints."""

import math
from dataclasses import replace

import pytest
import torch

from surface.dispersed_liquid_drift import DispersedOilWaterClosure


def closure(**options):
    return DispersedOilWaterClosure(0, 1, 2, 1e-5, 0.005, 0.001, 0.03, **options)


@pytest.mark.parametrize("interface", ["immobile", "clean"])
@pytest.mark.parametrize("bulk", [-0.2, 0.0, 0.2])
@pytest.mark.parametrize("gravity", [-9.80665, 0.0, 9.80665])
def test_three_velocities_satisfy_mass_average_gas_law_and_droplet_law(interface, bulk, gravity):
    c = closure(interface=interface, distribution_parameter=1.1, drift_velocity_m_s=0.1)
    alpha = torch.tensor([0.2, 0.04, 0.76], dtype=torch.float64)
    rho = alpha.new_tensor([10.0, 800.0, 1000.0])
    v, valid = c.velocities(
        alpha,
        rho,
        rho.new_tensor(bulk),
        phase_mole_fractions=torch.eye(3, dtype=rho.dtype),
        diameter_m=0.05,
        axial_gravity_m_s2=gravity,
        gravity_m_s2=9.80665,
    )
    assert bool(valid)
    expected = (1000 - 800) * gravity * 1e-10 / (18 * 0.001)
    if interface == "clean":
        expected *= 3 * (0.005 + 0.001) / (3 * 0.005 + 2 * 0.001)
    assert float(v[1] - v[2]) == pytest.approx(expected, abs=1e-15)
    assert float((alpha * rho * v).sum() / (alpha * rho).sum()) == pytest.approx(bulk, abs=1e-15)
    assert float(v[0]) == pytest.approx(1.1 * float((alpha * v).sum()) + 0.1, abs=1e-15)


@pytest.mark.parametrize("roles", [[0, 1], [0, 2], [1, 2]])
def test_absent_phase_and_permuted_phase_order_are_explicit(roles):
    c = closure()
    rho = torch.tensor([10.0, 800.0, 1000.0], dtype=torch.float64)[roles]
    alpha = rho.new_tensor([0.05, 0.95])
    x = torch.eye(3, dtype=rho.dtype)[roles]
    context = {"phase_mole_fractions": x, "diameter_m": 0.05, "axial_gravity_m_s2": 9.80665}
    v, valid = c.velocities(alpha, rho, rho.new_tensor(0), **context)
    assert bool(valid)
    permuted, ok = c.velocities(
        alpha.flip(0),
        rho.flip(0),
        rho.new_tensor(0),
        **{**context, "phase_mole_fractions": x.flip(0)},
    )
    assert bool(ok)
    torch.testing.assert_close(v.flip(0), permuted, rtol=1e-14, atol=1e-14)
    assert float((alpha * rho * v).sum()) == pytest.approx(0, abs=1e-14)


def test_drop_drag_limits_and_domain_checks_include_transverse_gravity():
    c = closure(interface="clean")
    rho = torch.tensor([800.0, 1000.0], dtype=torch.float64)
    solid = replace(c, interface="immobile").liquid_slip(*rho, 9.80665)[0]
    inviscid = replace(c, oil_viscosity_pa_s=1e-12).liquid_slip(*rho, 9.80665)[0]
    rigid = replace(c, oil_viscosity_pa_s=1e8).liquid_slip(*rho, 9.80665)[0]
    assert float(inviscid / solid) == pytest.approx(1.5, rel=1e-8)
    assert float(rigid / solid) == pytest.approx(1.0, rel=1e-10)
    large = replace(c, droplet_diameter_m=0.002)
    _, valid = large.velocities(
        rho.new_tensor([0.05, 0.95]),
        rho,
        rho.new_tensor(0),
        phase_mole_fractions=torch.eye(3, dtype=rho.dtype)[[1, 2]],
        diameter_m=1.0,
        axial_gravity_m_s2=0.0,
        gravity_m_s2=9.80665,
    )
    assert not bool(valid)  # Horizontal projection is zero; full Re/Eo still apply.
    with pytest.raises(ValueError):
        replace(c, water_component_index=1)
    with pytest.raises(ValueError):
        replace(c, maximum_dispersed_liquid_fraction=0.6)


def test_role_ambiguity_and_concentrated_oil_are_rejected():
    c = closure()
    rho = torch.tensor([800.0, 1000.0], dtype=torch.float64)
    kwargs = {"diameter_m": 0.05, "axial_gravity_m_s2": 9.80665}
    for alpha, x in [
        ([0.5, 0.5], [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ([0.05, 0.95], [[0.4, 0.3, 0.3], [0.0, 0.0, 1.0]]),
    ]:
        _, valid = c.velocities(
            rho.new_tensor(alpha),
            rho,
            rho.new_tensor(0),
            phase_mole_fractions=rho.new_tensor(x),
            **kwargs,
        )
        assert not bool(valid)


def three_phase_pipe(interface="clean", orientation=1, dense_limit=128, dispersed_liquid="oil"):
    from test_surface_compositional_separator import ternary

    from surface.pr_equilibrium_transport import (
        PREquilibriumLink,
        PREquilibriumNode,
        PREquilibriumTransportNetwork,
    )

    fluid = ternary()
    tie = fluid.flash_pt(1e6, 350.0, [0.4, 0.3, 0.3])
    fractions = (
        {0: 0.2, 1: 0.04, 2: 0.76} if dispersed_liquid == "oil" else {0: 0.2, 1: 0.76, 2: 0.04}
    )
    phases = tie["phases"]
    amounts = torch.stack(
        [
            fluid.tensor(fractions[int(phase["mole_fractions"].argmax())])
            / phase["molar_volume_m3_mol"]
            for phase in phases
        ]
    )
    composition = (
        torch.stack([phase["mole_fractions"] for phase in phases]) * amounts[:, None]
    ).sum(0) / amounts.sum()
    law = closure(
        interface=interface,
        drift_velocity_m_s=orientation * 0.03,
        dispersed_liquid=dispersed_liquid,
    )
    nodes = [
        PREquilibriumNode(
            str(i),
            math.pi * 0.05**2 / 4 * 0.1,
            "pipe_cell",
            elevation_m=orientation * (0.05 + i * 0.1),
            length_m=0.1,
            diameter_m=0.05,
            elevation_change_m=orientation * 0.1,
            drift_flux=law,
        )
        for i in range(2)
    ]
    model = PREquilibriumTransportNetwork(
        fluid,
        nodes,
        [PREquilibriumLink("internal", "0", "1", 0.05)],
        gravity_m_s2=9.80665,
        dense_limit=dense_limit,
    )
    initial = {
        str(i): {"pressure_pa": 1e6, "temperature_k": 350.0, "mole_fractions": composition.tolist()}
        for i in range(2)
    }
    return model, model.initialize(initial), initial


@pytest.mark.parametrize("orientation", [-1, 1])
@pytest.mark.parametrize("dense_limit", [1, 128])
def test_live_three_phase_pr_drift_is_conservative_and_differentiable(orientation, dense_limit):
    import json

    m, old, _ = three_phase_pipe(orientation=orientation, dense_limit=dense_limit)
    assert all(len(body.phase_roots) == 3 for _, body in old.nodes)
    seeds = [m._seed(i, body) for i, (_, body) in enumerate(old.nodes)]
    y, _, _, ev, offsets, _, _ = m._problem(old, seeds, 1e-7, {"internal": True}, [0.0] * 2)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    jac = torch.func.jacfwd(lambda z: ev(z)[0])(y)
    eps = 1e-4
    fd = (ev(y + eps * direction)[0] - ev(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, atol=1e-7, rtol=1e-5)
    pattern = m._sparsity(offsets, {"internal": True})
    mask = torch.zeros_like(jac, dtype=torch.bool)
    mask[pattern[0], pattern[1]] = True
    assert not bool((jac[~mask].abs() > 1e-12).any())
    new, report = m._step(old, 1e-7, {"internal": True}, [0.0] * 2, [350.0] * 2)
    assert max(map(abs, report["conservation_residuals"])) < 1e-6
    data = m._state_data(0, new.nodes[0][1])
    (_gas, oil, water), valid = m.nodes[0].drift_flux.roles(data["x"])
    assert bool(valid)
    oil_v, water_v = [(data["phase_velocity"] * mask).sum() for mask in (oil, water)]
    assert float((oil_v - water_v) * orientation) > 1e-6
    restored = type(m).from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(new)


def test_independent_constraint_solver_audits_full_three_phase_graph(monkeypatch):
    import copy
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    m, _, initial = three_phase_pipe()
    config = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [
                    {"duration_s": 1e-7},
                    {"duration_s": 1e-7, "openings": {"internal": False}},
                ],
            }
        )
    )
    result = json.loads(json.dumps(run(config)))
    assert audit(config, result)["accepted"]
    broken = copy.deepcopy(result)
    broken["intervals"][0]["steps"][0]["nodes"]["0"]["phase_velocity_m_s"][0] += 0.01
    with pytest.raises(ValueError):
        audit(config, broken)


@pytest.mark.parametrize("dispersed", ["oil", "water"])
@pytest.mark.parametrize("densities", [[800.0, 1000.0], [1000.0, 800.0], [900.0, 900.0]])
def test_both_dispersion_types_and_buoyancy_signs(dispersed, densities):
    c = closure(dispersed_liquid=dispersed, interface="clean")
    alpha = torch.tensor([0.05, 0.95] if dispersed == "oil" else [0.95, 0.05], dtype=torch.float64)
    rho = alpha.new_tensor(densities)
    x = torch.eye(3, dtype=alpha.dtype)[[1, 2]]
    v, valid = c.velocities(
        alpha,
        rho,
        rho.new_tensor(0),
        phase_mole_fractions=x,
        diameter_m=0.05,
        axial_gravity_m_s2=9.80665,
    )
    assert bool(valid)
    mu_c, mu_d = (0.001, 0.005) if dispersed == "oil" else (0.005, 0.001)
    expected = (
        (densities[1] - densities[0])
        * 9.80665
        * 1e-10
        / (18 * mu_c)
        * 3
        * (mu_c + mu_d)
        / (2 * mu_c + 3 * mu_d)
    )
    assert float(v[0] - v[1]) == pytest.approx(expected, abs=1e-15)
    assert float((alpha * rho * v).sum()) == pytest.approx(0, abs=1e-14)
    assert bool(c.pressure_valid(x, rho.new_tensor(1e6)))
    assert not bool(c.pressure_valid(x, rho.new_tensor(1e4)))


def test_water_droplets_also_advance_in_full_three_phase_pr_graph():
    m, old, _ = three_phase_pipe(dispersed_liquid="water")
    new, r = m._step(old, 1e-7, {"internal": True}, [0.0] * 2, [350.0] * 2)
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    data = m._state_data(0, new.nodes[0][1])
    (_, oil, water), valid = m.nodes[0].drift_flux.roles(data["x"])
    assert bool(valid)
    assert float((data["phase_velocity"] * water).sum()) < float(
        (data["phase_velocity"] * oil).sum()
    )
