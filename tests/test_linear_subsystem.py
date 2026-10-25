"""Связная подсистема линейного решателя: изолированные ячейки с перфорацией остаются в системе."""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from simulator.manual_assembly_solver import ManualFIMSolver  # noqa: E402


def _solver(conn_cell):
    # 4 ячейки: 0–1 связаны, 2 и 3 без соседей
    s = ManualFIMSolver.__new__(ManualFIMSolver)
    s.n_cells = 4
    s.builder = types.SimpleNamespace(indptr=torch.tensor([0, 2, 4, 5, 6]), indices=torch.tensor([0, 1, 0, 1, 2, 3]))
    s.wells = types.SimpleNamespace(conn_cell=conn_cell)
    return s


def test_perforated_isolated_cell_kept_after_compdat():
    s = _solver(torch.tensor([0]))
    keep, iso = s._linear_subsystem()[:2]
    assert iso.tolist() == [2, 3] and keep.tolist() == [0, 1]
    s.wells.conn_cell = torch.tensor([0, 3])     # новая перфорация в изолированной ячейке
    keep, iso = s._linear_subsystem()[:2]
    assert iso.tolist() == [2] and keep.tolist() == [0, 1, 3]
    assert s._linear_subsystem()[0].tolist() == [0, 1, 3]
