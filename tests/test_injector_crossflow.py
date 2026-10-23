"""Перфорации нагнетательной скважины с обратным потоком (пласт → ствол) закрыты: масса не уходит из пласта."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.blackoil_fim import GAS, OIL, WAT  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")


def test_injector_backflow_connections_are_shut():
    _, info = run_deck(SPE1, max_days=1)
    fim = info["solver"]
    fw = fim.wells
    k = next(i for i, w in enumerate(fw.wells) if w.name == "INJ")
    assert fw.wells[k].type == "injector" and (fw.wells[k].injected_phase or "").lower() == "gas"
    p = fim.fluid.pressure.reshape(-1).double()
    sw = fim.fluid.s_w.reshape(-1).double()
    v, d = fim.cell_state(p, sw, fim.x, fim.sat, fim.ug)
    m = fw.conn_well == k
    pb = lambda val: torch.full((fw.n_w,), val, dtype=torch.float64, device=p.device)

    # забой выше пласта — закачка: только газ, в пласт
    q, _, _ = fw._conn_flows(v, d, pb(1e9))
    assert bool((q[m][:, GAS] < 0).all()) and float(q[m][:, [OIL, WAT]].abs().max()) == 0.0

    # забой ниже пласта — обратный поток не допускается: расхода и производных по ячейкам нет, а производная
    # по забою остаётся (уравнение скважины на дебите не вырождается)
    q, dq, dqb = fw._conn_flows(v, d, pb(1e5))
    assert float(q[m].abs().max()) == 0.0 and float(dq[m].abs().max()) == 0.0
    assert bool((dqb[m][:, GAS] < 0).all())
