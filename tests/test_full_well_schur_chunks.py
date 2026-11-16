"""Bounded RHS Schur elimination, including several wells sharing one cell."""

import pytest
import torch

from simulator.full_well_schur import FullWellSchur


@pytest.mark.parametrize("nw,nconn,ncell,chunk_cells", [(7, 19, 8, 2), (192, 4096, 1024, 16)])
def test_bounded_rhs_matches_explicit_elimination(nw, nconn, ncell, chunk_cells, monkeypatch):
    generator = torch.Generator().manual_seed(27)
    rand = lambda *size: torch.randn(*size, dtype=torch.float64, generator=generator)
    w = FullWellSchur()
    w.n_w = nw
    w.conn_cell = torch.arange(nconn) % ncell
    w.conn_well = torch.arange(nconn) % nw
    w.maximum_rhs_entries = nw * 3 * chunk_cells
    raw = rand(nw, nw)
    D = torch.eye(nw, dtype=torch.float64) + raw @ raw.T / nw
    B, Cv = rand(nw, nconn, 3), rand(nconn, 3)
    sys = {"D_full": D, "B_full": B, "Cv": Cv, "r_w": rand(nw), "diag_pos": w.conn_cell}
    Bcell = D.new_zeros(nw, ncell, 3).index_add_(1, w.conn_cell, B)
    original = w._solve_D
    observed = []

    def bounded(system, rhs):
        observed.append(rhs.numel())
        assert rhs.numel() <= w.maximum_rhs_entries
        return original(system, rhs)

    monkeypatch.setattr(w, "_solve_D", bounded)
    J = D.new_zeros(ncell, 3, 3)
    w.add_local_schur(sys, J)
    assert sys["_surface_Bcell"] is None
    x = rand(ncell, 3)
    bx = torch.einsum("wcb,cb->w", Bcell, x)
    coefficient = torch.linalg.solve(D, bx)
    expected = torch.zeros_like(x).index_add_(0, w.conn_cell, -Cv * coefficient[w.conn_well, None])
    actual = torch.einsum("cij,cj->ci", J, x) + w.schur_correction(sys, x)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        w.recover_dpw(sys, x), torch.linalg.solve(D, -sys["r_w"] - bx), rtol=1e-10, atol=1e-10
    )
    rhs = torch.zeros_like(x).index_add_(
        0, w.conn_cell, Cv * torch.linalg.solve(D, sys["r_w"])[w.conn_well, None]
    )
    torch.testing.assert_close(w.schur_rhs(sys, ncell), rhs, rtol=1e-10, atol=1e-10)
    assert observed and max(observed) <= w.maximum_rhs_entries
