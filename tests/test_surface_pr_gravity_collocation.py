"""Common gravitational PR inventories and finite-aperture stage fluxes."""

import pytest
import torch
from test_surface_pr_gravity_transport import setup

from surface.pr_gravity_collocation import PRGravityCollocation


@pytest.mark.parametrize("reverse,isolated", [(False, False), (True, False), (False, True)])
def test_common_radau_component_energy_and_reverse(reverse, isolated):
    model, old, commands = setup(reverse=reverse, isolated=isolated)
    solver = PRGravityCollocation(model, phase_fallback=False)
    new, record = solver._step(old, 1e-6, commands, {}, {}, {})
    assert record["method"] == "radau3"
    assert record["quadrature_error"] <= 1
    assert max(map(abs, record["conservation_residuals"])) < 1e-6
    assert record["entropy_production_j_k"] >= -1e-6
    port = record["columns"]["ports"]["port"]
    assert (
        port["flux"][0] < 0
        if reverse
        else port["flux"][0] == 0
        if isolated
        else port["flux"][0] > 0
    )
    before = torch.tensor(old.columns[0][1].component_moles, dtype=torch.float64)
    after = torch.tensor(new.columns[0][1].component_moles, dtype=torch.float64)
    torch.testing.assert_close(
        after - before,
        -torch.tensor(port["integrated_flux"][:-2], dtype=before.dtype),
        rtol=1e-5,
        atol=1e-10,
    )


def test_gravitational_common_jacobian_and_exact_declared_support():
    model, old, commands = setup()
    solver = PRGravityCollocation(model, phase_fallback=False)
    problem, _ = solver._problem(old, 2, commands, {}, {}, {})
    y, _, _, evaluate, rates, *_ = problem

    def row(z):
        storage, point = evaluate(z)
        return storage - 1e-6 * rates(point)

    jac = torch.func.jacfwd(row)(y)
    direction = torch.linspace(-0.001, 0.001, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (row(y + eps * direction) - row(y - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-4, atol=2e-6)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    mask[evaluate.pattern] = True
    assert not bool((jac[~mask].abs() > 1e-10).any())
    assert bool(torch.isfinite(row(y)).all())


def test_public_advance_resolves_exact_interval_without_external_step_doubling():
    model, old, commands = setup()
    solver = PRGravityCollocation(model, phase_fallback=False)
    result = solver.advance(old, 2e-6, openings=commands)
    assert result["state"]["transport"]["time_s"] == 2e-6
    assert result["fallback_intervals"] == 0
    assert result["rejected_attempts"] == 0
    assert all(r["method"] == "radau3" for r in result["steps"])
    assert max(map(abs, result["conservation_residuals"])) < 1e-6


def test_column_wall_is_a_joint_inventory_and_conserves_total_energy():
    from dataclasses import asdict

    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.vessel_wall import VesselWall

    base, state, commands = setup(isolated=True)
    config = base.to_dict()
    config["walls"] = {"separator": asdict(VesselWall(10.0, 1.0, 0.0, 298.15))}
    model = PRGravityTransportNetwork.from_dict(config)
    old = model.initialize(state.transport, dict(state.columns), {"separator": 330.0})
    solver = PRGravityCollocation(model, phase_fallback=False)
    new, record = solver._step(old, 1e-6, commands, {}, {}, {})
    assert dict(new.walls)["separator"] < 330
    assert new.columns[0][1].total_energy_j > old.columns[0][1].total_energy_j
    assert max(map(abs, record["conservation_residuals"])) < 1e-6
