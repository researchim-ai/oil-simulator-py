"""Переменные ячейки без нефти: вход в Sg при So ≤ SO_EMPTY, выход на Rs только при So > SO_FILLED (без дребезга)."""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.blackoil_fim import SO_EMPTY, SO_FILLED  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")


def test_water_cell_does_not_chatter_between_sg_and_rs():
    _, info = run_deck(SPE1, max_days=1)
    fim = info["solver"]
    n = fim.n_cells if hasattr(fim, "n_cells") else fim.p.numel()
    f = dict(dtype=torch.float64, device=fim.p.device) if hasattr(fim, "p") else dict(dtype=torch.float64)
    p = torch.full((n,), 2.0e7, **f)
    zero = torch.zeros((n, 3), **f)
    dbg = SimpleNamespace(enabled=False)
    ug = torch.zeros(n, dtype=torch.bool, device=p.device)

    # водоносная ячейка в Sg, So чуть выше SO_EMPTY и Sg чуть ниже нуля — остаётся в Sg
    sw = torch.full((n,), 1.0 - 10 * SO_EMPTY, **f)
    sat = torch.ones(n, dtype=torch.bool, device=p.device)
    x = torch.full((n,), -1e-9, **f)
    _, _, _, sat1, _, n_sw = fim._apply_update(p, sw, x, sat, ug, zero, dbg)
    assert bool(sat1.all()) and n_sw == 0

    # заметная нефть — обычный переход на Rs
    sw = torch.full((n,), 1.0 - 10 * SO_FILLED, **f)
    _, _, _, sat2, _, n_sw = fim._apply_update(p, sw, x, sat, ug, zero, dbg)
    assert not bool(sat2.any()) and n_sw == n

    # ячейка на Rs без нефти — переводится в Sg
    sw = torch.full((n,), 1.0, **f)
    x = torch.zeros((n,), **f)
    _, _, _, sat3, _, _ = fim._apply_update(p, sw, x, ~sat, ug, zero, dbg)
    assert bool(sat3.all())
