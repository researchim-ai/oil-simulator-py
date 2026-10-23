"""Переток из пласта в ствол нагнетательной скважины несёт флюиды ячейки, а не закачиваемую фазу."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.blackoil_fim import GAS, OIL, WAT  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")


def test_injector_backflow_carries_cell_fluids():
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

    # забой ниже пласта — переток в ствол: флюиды ячейки по их подвижностям, как у добывающей
    q_out, _, _ = fw._conn_flows(v, d, pb(1e5))
    assert bool((q_out[m] >= 0).all()) and float(q_out[m][:, OIL].sum()) > 0.0
    fw.is_inj[k] = False
    q_prod, _, _ = fw._conn_flows(v, d, pb(1e5))
    fw.is_inj[k] = True
    assert torch.allclose(q_out[m], q_prod[m], rtol=1e-12, atol=0.0)
