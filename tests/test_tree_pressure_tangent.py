"""Compact tree derivatives: transpose, physical FD and deep-chain storage."""

import json
from dataclasses import replace

import pytest
import torch

from surface.tree_pressure_tangent import TreePressureTangent


def spec():
    return {
        "model": "blackoil_tree_pressure_tangent_v1",
        "node_names": ["A", "root", "header", "B"],
        "parent_names": ["header", None, "root", "header"],
        "source_names": ["A", "header", "B", "root"],
        "pressure_coefficients": [1.1, 0, 0.9, 1.2],
        "mass_coefficients": [[1, 2, 3], [0, 0, 0], [2, 1, 4], [-1, 3, 2]],
        "valve_names": ["trim", "header_trim"],
        "valve_nodes": ["A", "header"],
        "valve_coefficients": [-3, -2],
    }


@pytest.mark.parametrize("batch", [False, True])
def test_joint_adjoint_matches_autograd_and_dot_product(batch):
    t = TreePressureTangent(spec())
    g = torch.Generator().manual_seed(23)
    rand = lambda *shape: torch.randn(shape, dtype=torch.float64, generator=g, requires_grad=True)
    q = rand(4, 3, 5) if batch else rand(4, 3)
    u = rand(2, 5) if batch else rand(2)
    bp = rand(5) if batch else rand()
    y = t.apply(q, receiving_pressure=bp, valve_openings=u)
    z = rand(*y.shape).detach()
    expected = torch.autograd.grad((y * z).sum(), (q, bp, u))
    actual = t.adjoint(z)
    for a, b in zip(expected, actual.values(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    dot = (
        (q * actual["component_mass_rates"]).sum()
        + (bp * actual["receiving_pressure"]).sum()
        + (u * actual["valve_openings"]).sum()
    )
    torch.testing.assert_close((y * z).sum(), dot, rtol=1e-12, atol=1e-12)
    assert t.to_dict() == spec()
    torch.testing.assert_close(
        TreePressureTangent(json.loads(json.dumps(t.to_dict()))).matvec(q), t.matvec(q)
    )


def test_selected_rows_and_explicit_budgets():
    t = TreePressureTangent(spec())
    q = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    rows = t.jacobian_rows(["A", "B", "A"], maximum_rhs_entries=4)
    expected = t.matvec(q)[[0, 3, 0]]
    torch.testing.assert_close(torch.einsum("rsc,sc->r", rows, q), expected)
    with pytest.raises(ValueError, match="output"):
        t.jacobian_rows(["A", "B"], maximum_output_entries=1)
    with pytest.raises(ValueError, match="RHS"):
        t.jacobian_rows(["A"], maximum_rhs_entries=1)
    with pytest.raises(ValueError, match="dimensions"):
        t.matvec(torch.zeros(3, 4, dtype=torch.float64))


def test_deep_tree_does_not_store_node_by_source_matrix():
    n = 4096
    d = {
        "model": "blackoil_tree_pressure_tangent_v1",
        "node_names": [str(i) for i in range(n)],
        "parent_names": [str(i + 1) for i in range(n - 1)] + [None],
        "source_names": [str(i) for i in range(n)],
        "pressure_coefficients": [1.0] * (n - 1) + [0.0],
        "mass_coefficients": [[1.0, 2.0, 3.0]] * (n - 1) + [[0.0, 0.0, 0.0]],
        "valve_names": [],
        "valve_nodes": [],
        "valve_coefficients": [],
    }
    t = TreePressureTangent(d)
    q = torch.zeros(n, 3, dtype=torch.float64)
    q[0, 1] = 1
    torch.testing.assert_close(
        t.matvec(q), 2 * torch.arange(n - 1, -1, -1, dtype=torch.float64), atol=0, rtol=0
    )
    z = torch.ones(n, dtype=torch.float64)
    torch.testing.assert_close((t.matvec(q) * z).sum(), (q * t.rmatvec(z)).sum())

    def entries(value):
        if torch.is_tensor(value):
            return value.numel()
        if isinstance(value, (tuple, list)):
            return sum(entries(x) for x in value)
        return 0

    stored = sum(entries(v) for v in vars(t).values())
    assert stored < 80 * n
    with pytest.raises(ValueError, match="output"):
        t.jacobian_rows(t.node_names)


@pytest.mark.parametrize("outlet", [1.2e6, 2.75e6])
def test_physical_tree_matches_dense_derivatives_and_directional_fd(outlet):
    from test_surface_blackoil_network import DENSITIES, feeds, network

    from surface.process import ComponentStream

    n, sources = network(outlet), feeds()
    dense = n.solve(sources)
    compact = n.solve(sources, sensitivity_mode="operator")
    assert compact["node_pressures_pa"] == dense["node_pressures_pa"]
    assert "pressure_derivatives_per_kg_s" not in compact
    assert "pressure_derivatives_valve_opening" not in compact
    t = TreePressureTangent(compact["pressure_linearization"])
    rows = t.jacobian_rows(t.node_names)
    expected = torch.tensor(
        [
            [
                [
                    dense["pressure_derivatives_per_kg_s"][node][s][c]
                    for c in ("oil", "water", "gas")
                ]
                for s in t.source_names
            ]
            for node in t.node_names
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(rows, expected, rtol=1e-12, atol=1e-8)
    own = t.own_source_coefficients()
    for j, source in enumerate(t.source_names):
        torch.testing.assert_close(
            own[t.node_index[source]], rows[t.node_index[source], j], rtol=1e-12, atol=1e-8
        )
    direction = torch.tensor([[0.2, -0.1, 0.002], [0.1, 0.2, -0.001]], dtype=torch.float64)
    plus, minus = {}, {}
    eps = 1e-4
    for s, dq in zip(t.source_names, direction, strict=True):
        m = list(sources[s].mass_rates(DENSITIES).values())
        plus[s] = ComponentStream.from_mass_rates(
            *[a + eps * float(b) for a, b in zip(m, dq)], DENSITIES
        )
        minus[s] = ComponentStream.from_mass_rates(
            *[a - eps * float(b) for a, b in zip(m, dq)], DENSITIES
        )
    p1, p0 = n.solve(plus)["node_pressures_pa"], n.solve(minus)["node_pressures_pa"]
    fd = torch.tensor(
        [(p1[node] - p0[node]) / (2 * eps) for node in t.node_names], dtype=torch.float64
    )
    torch.testing.assert_close(t.matvec(direction), fd, rtol=1e-5, atol=0.003)


def test_physical_valve_column_and_receiving_boundary():
    from test_surface_blackoil_network import feeds, network

    from surface.blackoil_network import BlackOilProductionNetwork

    old = network()
    pipes = [
        replace(p, loss_coefficient=12, valve_opening=0.7) if p.name == "a" else p
        for p in old.pipes
    ]
    n = BlackOilProductionNetwork(old.model, old.nodes, pipes)
    r = n.solve(feeds(), sensitivity_mode="operator")
    t = TreePressureTangent(r["pressure_linearization"])
    zero = torch.zeros(len(t.source_names), 3, dtype=torch.float64)
    h = 1e-5
    plus = n.with_controls({"a": 0.7 + h}).solve(feeds())["node_pressures_pa"]
    minus = n.with_controls({"a": 0.7 - h}).solve(feeds())["node_pressures_pa"]
    fd = torch.tensor([(plus[x] - minus[x]) / (2 * h) for x in t.node_names], dtype=torch.float64)
    torch.testing.assert_close(
        t.apply(zero, valve_openings=torch.ones(1, dtype=torch.float64)), fd, rtol=1e-5, atol=0.003
    )
    expected = torch.tensor(
        [r["pressure_derivatives_receiving_pressure"][x] for x in t.node_names], dtype=torch.float64
    )
    torch.testing.assert_close(
        t.apply(zero, receiving_pressure=1.0), expected, rtol=1e-12, atol=1e-12
    )


def test_invalid_topology_and_receiving_law_are_rejected():
    d = spec()
    d["parent_names"][2] = "A"
    with pytest.raises(ValueError):
        TreePressureTangent(d)
    d = spec()
    d["mass_coefficients"][1] = [1, 0, 0]
    with pytest.raises(ValueError):
        TreePressureTangent(d)


def test_unrelated_large_branch_cannot_erase_small_branch_in_either_direction():
    d = {
        "model": "blackoil_tree_pressure_tangent_v1",
        "node_names": ["A", "B", "root"],
        "parent_names": ["root", "root", None],
        "source_names": ["A", "B"],
        "pressure_coefficients": [1, 1, 0],
        "mass_coefficients": [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
        "valve_names": [],
        "valve_nodes": [],
        "valve_coefficients": [],
    }
    t = TreePressureTangent(d)
    q = torch.tensor([[1e16, 0, 0], [1, 0, 0]], dtype=torch.float64)
    assert t.matvec(q)[1] == 1
    z = torch.tensor([1e16, 1, 0], dtype=torch.float64)
    assert t.rmatvec(z)[1, 0] == 1


def test_explicit_cpu_device_index_is_canonicalized():
    t = TreePressureTangent(spec(), device="cpu:0")
    assert t.matvec(torch.zeros(4, 3, dtype=torch.float64)).shape == (4,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_actions_match_cpu_and_accept_unindexed_cuda_device():
    q = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    cpu = TreePressureTangent(spec())
    gpu = TreePressureTangent(spec(), device="cuda")
    torch.testing.assert_close(gpu.matvec(q.cuda()).cpu(), cpu.matvec(q))
    z = torch.arange(4, dtype=torch.float64)
    torch.testing.assert_close(gpu.rmatvec(z.cuda()).cpu(), cpu.rmatvec(z))
