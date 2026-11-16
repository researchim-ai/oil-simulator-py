"""Structured B/D matches dense elimination without allocating those matrices."""

import pytest
import torch
from test_tree_pressure_tangent import spec

from simulator.surface_well_blocks import assemble_surface_blocks
from simulator.tree_well_system import TreeWellSystem
from surface.liquid import HydraulicConvergenceError
from surface.tree_pressure_tangent import TreePressureTangent


def fixture():
    g = torch.Generator().manual_seed(8)
    rand = lambda *shape: torch.randn(*shape, dtype=torch.float64, generator=g)
    nw, nc, nb = 7, 31, 3
    names = ["A", "B", "header", "A", "B", "header", "A"]
    cw = torch.arange(nc) % nw
    cells = torch.arange(nc) % 8
    dq = rand(nc, 3, nb) * 0.01
    cv = -rand(nc, 3).abs() * 0.001
    weights = rand(nc, 3).abs()
    rate = torch.tensor([True, False, False, True, False, False, False])
    surface = torch.tensor([False, True, False, False, True, False, True])
    data = spec()
    t = TreePressureTangent(data)
    k = t.jacobian_rows(names)[:, [t.source_names.index(n) for n in names]]
    B, D, _br = assemble_surface_blocks(k, cw, dq, cv, weights, rate, surface)
    op = TreeWellSystem(data, names, cw, cells, dq, cv, weights, rate, surface)
    return op, B, D, rand


def test_structured_actions_and_inverse_against_dense_reference():
    op, B, D, rand = fixture()
    for count in (1, 4):
        rhs = rand(7) if count == 1 else rand(7, count)
        torch.testing.assert_close(op.solve(rhs), torch.linalg.solve(D, rhs), rtol=1e-10, atol=1e-9)
        torch.testing.assert_close(op.apply_D(rhs), D @ rhs, rtol=1e-12, atol=1e-12)
    dx = rand(8, 3)
    torch.testing.assert_close(
        op.Bx(dx), torch.einsum("wcb,cb->w", B, dx[op.conn_cell]), rtol=1e-12, atol=1e-12
    )
    assert not bool(op.factor_bad)
    assert not op.iterative_fallback
    assert not any(torch.is_tensor(v) and tuple(v.shape) == (7, 7) for v in vars(op).values())


def test_large_tree_inverse_uses_linear_data_and_no_dense_linalg(monkeypatch):
    nw, nc = 512, 8192
    names = [f"p{i}" for i in range(nw)] + ["header", "root"]
    data = {
        "model": "blackoil_tree_pressure_tangent_v1",
        "node_names": names,
        "parent_names": ["header"] * nw + ["root", None],
        "source_names": names[:-2],
        "pressure_coefficients": [1.0] * (nw + 1) + [0.0],
        "mass_coefficients": [[0.001, 0.002, 0.003]] * nw + [[0.5, 0.2, 0.8], [0.0, 0.0, 0.0]],
        "valve_names": [],
        "valve_nodes": [],
        "valve_coefficients": [],
    }
    cw = torch.arange(nc) % nw
    cv = torch.full((nc, 3), -1e-5, dtype=torch.float64)
    dq = torch.ones(nc, 3, 3, dtype=torch.float64) * 1e-5
    weights = torch.ones(nc, 3, dtype=torch.float64)
    rate = torch.arange(nw) % 3 == 0
    surface = ~rate

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Dense well factorization forbidden")

    monkeypatch.setattr(torch.linalg, "lu_factor_ex", forbidden)
    monkeypatch.setattr(torch.linalg, "solve", forbidden)
    op = TreeWellSystem(
        data, names[:-2], cw, torch.arange(nc) % 1024, dq, cv, weights, rate, surface
    )
    exact = torch.sin(torch.arange(nw, dtype=torch.float64))
    torch.testing.assert_close(op.solve(op.apply_D(exact)), exact, rtol=1e-10, atol=1e-10)
    assert op.tangent is not None
    assert not op.iterative_fallback
    assert not any(
        torch.is_tensor(v) and (tuple(v.shape) == (nw, nw) or tuple(v.shape) == (nw, nc, 3))
        for v in vars(op).values()
    )


def test_bad_local_pivot_uses_certified_matrix_free_fallback():
    data = {
        "model": "blackoil_tree_pressure_tangent_v1",
        "node_names": ["child", "parent", "root"],
        "parent_names": ["parent", "root", None],
        "source_names": ["child", "parent"],
        "pressure_coefficients": [1.0, 1.0, 0.0],
        "mass_coefficients": [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "valve_names": [],
        "valve_nodes": [],
        "valve_coefficients": [],
    }
    cv = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    op = TreeWellSystem(
        data,
        ["child", "parent"],
        torch.arange(2),
        torch.arange(2),
        torch.ones(2, 3, 3, dtype=torch.float64),
        cv,
        torch.ones(2, 3, dtype=torch.float64),
        torch.zeros(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )
    assert bool(op.pivot_bad)
    D = torch.tensor([[1.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 3.0], dtype=torch.float64)
    torch.testing.assert_close(op.solve(rhs), torch.linalg.solve(D, rhs), rtol=1e-10, atol=1e-10)
    assert op.iterative_fallback
    assert not bool(op.factor_bad)


def test_singular_global_block_is_not_accepted_as_a_compatible_gmres_solution():
    data = {
        "model": "blackoil_tree_pressure_tangent_v1",
        "node_names": ["p", "root"],
        "parent_names": ["root", None],
        "source_names": ["p"],
        "pressure_coefficients": [1.0, 0.0],
        "mass_coefficients": [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "valve_names": [],
        "valve_nodes": [],
        "valve_coefficients": [],
    }
    op = TreeWellSystem(
        data,
        ["p"],
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, 3, 3, dtype=torch.float64),
        torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.ones(1, 3, dtype=torch.float64),
        torch.zeros(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
    )
    with pytest.raises(HydraulicConvergenceError):
        op.solve(torch.zeros(1, dtype=torch.float64))


def test_full_schur_action_with_approximate_local_part_is_exact():
    from simulator.full_well_schur import FullWellSchur

    op, B, D, rand = fixture()
    wells = FullWellSchur()
    wells.n_w = op.nw
    wells.conn_well, wells.conn_cell = op.conn_well, op.conn_cell
    C = rand(31, 3)
    system = {"_surface_operator": op, "Cv": C, "r_w": rand(7), "diag_pos": op.conn_cell}
    wells.factor_well_block(system)
    J = torch.zeros(8, 3, 3, dtype=torch.float64)
    wells.add_local_schur(system, J)
    x = rand(8, 3)
    bx = torch.einsum("wcb,cb->w", B, x[op.conn_cell])
    coeff = torch.linalg.solve(D, bx)
    expected = torch.zeros_like(x).index_add_(0, op.conn_cell, -C * coeff[op.conn_well, None])
    actual = torch.einsum("cij,cj->ci", J, x) + wells.schur_correction(system, x)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        wells.recover_dpw(system, x),
        torch.linalg.solve(D, -system["r_w"] - bx),
        rtol=1e-10,
        atol=1e-9,
    )
    expected_rhs = torch.zeros_like(x).index_add_(
        0, op.conn_cell, C * torch.linalg.solve(D, system["r_w"])[op.conn_well, None]
    )
    torch.testing.assert_close(wells.schur_rhs(system, 8), expected_rhs, rtol=1e-10, atol=1e-9)
    assert "B_full" not in system and "D_full" not in system and "_surface_lu" not in system


def test_bad_solve_residual_is_returned_as_failure_not_an_accepted_update(monkeypatch):
    op, *_ = fixture()
    monkeypatch.setattr(op, "apply_D", lambda x: torch.zeros_like(x))
    assert torch.isnan(op.solve(torch.ones(7, dtype=torch.float64))).all()
    assert bool(op.factor_bad)
