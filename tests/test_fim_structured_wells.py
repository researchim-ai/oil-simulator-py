"""Actual FIM with operator-only well elimination and unchanged convergence."""

import pickle

import pytest
import torch
from test_fim_blackoil_surface import config, context, solver

from simulator.tree_well_system import TreeWellSystem


def structured_solver():
    c = config()
    c["simulation"]["physical_surface_network"]["well_block_backend"] = "tree_structured"
    return solver(c)


def test_real_jacobian_schur_action_and_rhs_match_dense_reference():
    a, b = solver(), structured_solver()
    *_, fa = context(a)
    *_, fb = context(b)
    ra, ja, wa = fa()
    rb, jb, wb = fb()
    assert "B_full" not in wb and "D_full" not in wb
    torch.testing.assert_close(ra, rb, rtol=0, atol=0)
    torch.testing.assert_close(ja, jb, rtol=0, atol=0)
    a.wells.add_local_schur(wa, ja)
    b.wells.add_local_schur(wb, jb)
    x = torch.arange(a.n_cells * 3, dtype=torch.float64).reshape(-1, 3) / 10

    def multiply(s, J, w, xx):
        out = torch.zeros_like(xx)
        for row in range(s.n_cells):
            for pos in range(int(s.builder.indptr[row]), int(s.builder.indptr[row + 1])):
                out[row] += J[pos] @ xx[int(s.builder.indices[pos])]
        return out + s.wells.schur_correction(w, xx)

    torch.testing.assert_close(
        multiply(a, ja, wa, x), multiply(b, jb, wb, x), rtol=1e-10, atol=1e-9
    )
    torch.testing.assert_close(
        a.wells.schur_rhs(wa, a.n_cells), b.wells.schur_rhs(wb, b.n_cells), rtol=1e-10, atol=1e-9
    )


def test_accepted_state_and_pickle_restart():
    a, b = solver(), structured_solver()
    assert a.step(86400) and b.step(86400)
    torch.testing.assert_close(a.fluid.pressure, b.fluid.pressure, rtol=1e-9, atol=0.01)
    torch.testing.assert_close(a.fluid.s_w, b.fluid.s_w, rtol=1e-8, atol=1e-8)
    assert b.wells._surface_eye is None
    saved = pickle.loads(pickle.dumps(b))
    assert b.step(86400) and saved.step(86400)
    torch.testing.assert_close(b.fluid.pressure, saved.fluid.pressure, rtol=0, atol=0)


def test_linear_nonfinite_is_rejected_and_physical_state_restored(monkeypatch):
    s = structured_solver()
    old = s.fluid.pressure.clone()

    def fail(self, rhs):
        self.factor_bad.fill_(True)
        return torch.full_like(rhs, float("nan"))

    monkeypatch.setattr(TreeWellSystem, "solve", fail)
    assert not s.step(86400)
    assert torch.equal(old, s.fluid.pressure)


def test_memory_admission_and_backend_combination_are_explicit():
    c = config()
    c["simulation"]["physical_surface_network"].update(
        well_block_backend="tree_structured", structured_memory_budget_bytes=1
    )
    with pytest.raises(ValueError, match="budget"):
        solver(c)
    c["simulation"]["physical_surface_network"].update(
        structured_memory_budget_bytes=256 * 1024**2, sensitivity_backend="dense"
    )
    with pytest.raises(Exception, match="require"):
        solver(c)


def test_legacy_dense_well_limit_is_not_silently_removed():
    import copy

    c = config()
    first = c["wells"][0]
    c["wells"] = [copy.deepcopy(first) for _ in range(129)]
    network = c["simulation"]["physical_surface_network"]
    node = network["well_nodes"][first["name"]]
    for i, well in enumerate(c["wells"]):
        well["name"] = f"W{i}"
    network["well_nodes"] = {well["name"]: node for well in c["wells"]}
    with pytest.raises(ValueError, match="<=128"):
        solver(c)
    network["well_block_backend"] = "tree_structured"
    s = solver(c)
    assert s.wells.n_w == 129 and s.wells._surface_eye is None
