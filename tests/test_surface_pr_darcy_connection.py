"""Reduced Darcy links: friction, exact AD, finite stocks and independent audit."""

import copy
import math

import pytest
import torch
from check_pr_gravity_transport import audit
from check_stratified_pr_network import reference_darcy
from run_pr_gravity_transport import run
from test_surface_compositional_separator import binary

from surface.pr_darcy_connection import PRDarcyConnection
from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork
from surface.pr_gravity_transport import PRGravityTransportNetwork


@pytest.mark.parametrize("velocity", [-25.0, -2.0, 0.0, 2.0, 25.0])
@pytest.mark.parametrize("loss_coefficient", [0.0, 1.0])
def test_inverse_matches_independent_friction_and_implicit_derivatives(velocity, loss_coefficient):
    c = PRDarcyConnection(
        "line",
        "a",
        "b",
        0.2,
        0.001,
        1e-4,
        roughness_m=1e-5,
        local_loss_coefficient=loss_coefficient,
    )
    rho = torch.tensor(10.0, dtype=torch.float64)
    dp = torch.tensor(
        reference_darcy(velocity, 10.0, 0.2, 0.001, 1e-4, 1e-5)[0]
        + 0.5 * loss_coefficient * 10.0 * velocity * abs(velocity),
        dtype=torch.float64,
    )
    actual = c.velocity(dp, rho)
    assert float(actual) == pytest.approx(velocity, abs=1e-11)
    derivative = torch.func.jacfwd(c.velocity, argnums=(0, 1))(dp, rho)
    eps = 1e-3
    fd = (c.velocity(dp + eps, rho) - c.velocity(dp - eps, rho)) / (2 * eps)
    torch.testing.assert_close(derivative[0], fd, rtol=1e-6, atol=1e-10)
    fd_rho = (c.velocity(dp, rho + 1e-4) - c.velocity(dp, rho - 1e-4)) / 2e-4
    torch.testing.assert_close(derivative[1], fd_rho, rtol=1e-6, atol=1e-9)


def fixture(reverse=False, sparse=False):
    f = binary()
    transport = PREquilibriumTransportNetwork(
        f,
        [PREquilibriumNode("a", 0.1), PREquilibriumNode("b", 0.1)],
        dense_limit=1 if sparse else 128,
        initial_step_s=0.001,
        maximum_step_s=0.01,
    )
    graph = PRGravityTransportNetwork(
        transport, {}, [], darcy_connections=[PRDarcyConnection("line", "a", "b", 0.2, 0.001, 1e-4)]
    )
    initial = {
        n: {"pressure_pa": p, "temperature_k": 350.0, "mole_fractions": [1.0, 0.0]}
        for n, p in (("a", 1e6 if reverse else 1.001e6), ("b", 1.001e6 if reverse else 1e6))
    }
    return {
        "model": graph.to_dict(),
        "initial": {"transport": initial, "columns": {}, "wall_temperatures": {}},
        "schedule": [{"duration_s": 0.001}],
    }


@pytest.mark.parametrize("reverse,sparse", [(False, False), (True, False), (False, True)])
def test_finite_conservative_graph_independent_audit_restart_and_tamper(reverse, sparse):
    config = fixture(reverse, sparse)
    report = run(config)
    assert audit(config, report)["accepted"]
    row = report["intervals"][0]["steps"][0]["transport"]["coupling"]["darcy_connections"]["line"]
    assert (row["volume_rate_m3_s"] < 0) is reverse
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    corrupted = copy.deepcopy(report)
    corrupted["intervals"][0]["steps"][0]["transport"]["coupling"]["darcy_connections"]["line"][
        "total_energy_rate_w"
    ] += 1.0
    with pytest.raises(ValueError):
        audit(config, corrupted)
    restart = {
        "model": config["model"],
        "initial": {"state": report["intervals"][-1]["state"]},
        "schedule": config["schedule"],
    }
    assert audit(restart, run(restart))["accepted"]


def test_isolation_keeps_exact_stocks_and_legacy_model_identity():
    config = fixture()
    config["model"]["darcy_connections"][0]["open"] = False
    report = run(config)
    assert audit(config, report)["accepted"]
    for name, before in report["initial"]["transport"]["nodes"].items():
        after = report["intervals"][-1]["state"]["transport"]["nodes"][name]
        for key in ("component_moles", "total_energy_j", "momentum_kg_m_s"):
            assert before[key] == after[key]
    parsed = PRGravityTransportNetwork.from_dict(config["model"])
    old = PRGravityTransportNetwork(parsed.transport, {}, [])
    assert "darcy_connections" not in old.to_dict()


def test_invalid_geometry_and_unsupported_endpoint_are_rejected():
    with pytest.raises(ValueError):
        PRDarcyConnection("line", "a", "b", 0.2, 0.001, 0.0)
    config = fixture()
    config["model"]["transport"]["nodes"][1]["elevation_m"] = 1.0
    with pytest.raises(ValueError, match="equal elevation"):
        PRGravityTransportNetwork.from_dict(config["model"])


def test_domain_failure_is_explicit_and_internal_exchange_cannot_create_energy():
    from surface.liquid import EquipmentOperatingError

    config = fixture()
    config["initial"]["transport"]["a"]["pressure_pa"] = 1.2e6
    with pytest.raises(EquipmentOperatingError, match="pressure drop"):
        run(config)
    config = fixture()
    report = run(config)
    report["intervals"][0]["steps"][0]["material_energy_j"] = 1.0
    with pytest.raises(ValueError):
        audit(config, report)


def test_conservative_mapping_preserves_initial_component_energy_and_volume():
    import json
    from pathlib import Path

    from reduce_pr_recycle import reduce_cell

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_integrated_facility.json"
        ).read_text()
    )
    before = copy.deepcopy(config)
    reduced, mapping = reduce_cell(config)
    assert config == before
    a = config["initial"]["state"]["plant"]["transport"]["nodes"]
    b = reduced["initial"]["state"]["plant"]["transport"]["nodes"]
    for i in range(3):
        assert math.fsum(s["component_moles"][i] for s in a.values()) == pytest.approx(
            math.fsum(s["component_moles"][i] for s in b.values()), abs=1e-10, rel=0
        )
    assert math.fsum(s["total_energy_j"] for s in a.values()) == pytest.approx(
        math.fsum(s["total_energy_j"] for s in b.values()), abs=1e-8, rel=0
    )
    assert math.fsum(
        n["volume_m3"] for n in config["model"]["plant"]["transport"]["nodes"]
    ) == pytest.approx(
        math.fsum(n["volume_m3"] for n in reduced["model"]["plant"]["transport"]["nodes"]),
        abs=1e-12,
        rel=0,
    )
    assert mapping["relocated_volume_m3"] > 0


def test_sparse_jacobian_includes_both_finite_endpoint_dependencies():
    from surface.pr_darcy_connection import DarcyMaterial

    config = fixture(sparse=True)
    graph = PRGravityTransportNetwork.from_dict(config["model"])
    t = graph.transport
    state = t.initialize(config["initial"]["transport"])
    openings = {"line": True}
    coupling = DarcyMaterial(graph, state, openings)
    seeds = [t._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, lo, hi, base, offsets, *_ = t._problem(state, seeds, 0.001, {}, [0.0, 0.0])
    y, lo, hi, fn, pattern = coupling.prepare(
        y, lo, hi, base, offsets, 0.001, t._sparsity(offsets, {})
    )
    jac = torch.func.jacfwd(lambda z: fn(z)[0])(y)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    mask[pattern] = True
    assert not bool((jac[~mask].abs() > 1e-12).any())
    assert bool((jac[: offsets[1], offsets[1] :].abs() > 1e-8).any())
    direction = torch.linspace(-1e-3, 1e-3, len(y), dtype=y.dtype)
    delta = 1e-4
    fd = (fn(y + delta * direction)[0] - fn(y - delta * direction)[0]) / (2 * delta)
    torch.testing.assert_close(jac @ direction, fd, atol=1e-8, rtol=1e-5)


def test_blocked_reverse_flow_does_not_flash_a_nonexistent_stream(monkeypatch):
    from surface.equilibrium_properties import EquilibriumFluid

    config = fixture(reverse=True)
    config["model"]["darcy_connections"][0]["allow_reverse"] = False

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Zero flow must not invoke a delivery flash")

    monkeypatch.setattr(EquilibriumFluid, "flash_ph", forbidden)
    report = run(config)
    assert audit(config, report)["accepted"]
    row = report["intervals"][0]["steps"][0]["transport"]["coupling"]["darcy_connections"]["line"]
    assert row["volume_rate_m3_s"] == 0 and row["delivery_temperature_k"] is None


def test_reduced_link_and_external_feed_share_one_conservative_solve():
    from check_pr_open_gravity_transport import audit as open_audit

    from surface.component_energy import ComponentEnergyStream
    from surface.pr_open_gravity_transport import PROpenGravityTransportNetwork
    from surface.pr_open_transport import PRTransportFeed

    config = fixture()
    graph = PRGravityTransportNetwork.from_dict(config["model"])
    old = graph.initialize(graph.transport.initialize(config["initial"]["transport"]), {})
    feed = ComponentEnergyStream.from_pt(
        graph.transport.fluid,
        [1e-4, 0.0],
        1.002e6,
        351.0,
        gravity_m_s2=graph.transport.gravity_m_s2,
    )
    model = PROpenGravityTransportNetwork(graph, [PRTransportFeed("feed", "a", feed)])
    result = model.advance(old, 0.001)
    assert open_audit(model.to_dict(), old.to_dict(), result)["accepted"]
    assert result["boundary_component_moles"][0] == pytest.approx(1e-7)


def test_reduced_flow_accuracy_is_controlled_even_when_stock_errors_are_small():
    config = fixture()
    graph = PRGravityTransportNetwork.from_dict(config["model"])
    initial = config["initial"]["transport"]
    coarse = graph.initialize(graph.transport.initialize(initial), {})
    changed = copy.deepcopy(initial)
    changed["a"]["pressure_pa"] += 1.0
    fine = graph.initialize(graph.transport.initialize(changed), {})
    stock_error = graph.transport._error(coarse.transport, fine.transport)
    assert stock_error < 1
    assert graph._error(coarse, fine, openings={"line": True}) > 1
    assert graph._error(coarse, fine, openings={"line": False}) == stock_error


@pytest.mark.parametrize("velocity", [-20.0, -2.0, 0.0, 2.0, 20.0])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_laminar_inverse_and_jacobian_do_not_iterate_friction(velocity, monkeypatch, device):
    import surface.pr_darcy_connection as module

    c = PRDarcyConnection("line", "a", "b", 0.2, 0.001, 1e-4, local_loss_coefficient=3.0)
    rho = torch.tensor(10.0, dtype=torch.float64, device=device)
    dp = torch.tensor(
        640.0 * velocity + 15.0 * velocity * abs(velocity), dtype=torch.float64, device=device
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("laminar inverse entered iterative friction kernel")

    monkeypatch.setattr(module, "darcy_pressure_loss", forbidden)
    assert float(c.velocity(dp, rho)) == pytest.approx(velocity, abs=1e-12)
    derivatives = torch.func.jacfwd(c.velocity, argnums=(0, 1))(dp, rho)
    slope = 640.0 + 30.0 * abs(velocity)
    assert float(derivatives[0]) == pytest.approx(1 / slope, rel=1e-12)
    assert float(derivatives[1]) == pytest.approx(
        -1.5 * velocity * abs(velocity) / slope, abs=1e-12
    )


def test_mixed_regime_vmap_and_transition_are_preserved():
    c = PRDarcyConnection("line", "a", "b", 0.2, 0.001, 1e-4)
    speeds = [-40.0, -20.000001, -20.0, -19.999999, 0.0, 19.999999, 20.0, 20.000001, 40.0]
    dp = torch.tensor(
        [reference_darcy(v, 10.0, 0.2, 0.001, 1e-4, 0.0)[0] + 5.0 * v * abs(v) for v in speeds],
        dtype=torch.float64,
    )
    rho = torch.full_like(dp, 10.0)
    result = torch.vmap(c.velocity)(dp, rho)
    torch.testing.assert_close(
        result, torch.tensor(speeds, dtype=torch.float64), atol=1e-11, rtol=1e-12
    )
