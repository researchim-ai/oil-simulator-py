"""Переток из пласта в ствол нагнетательной скважины несёт флюиды ячейки, а не закачиваемую фазу."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.blackoil_fim import GAS, OIL, WAT  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")


def test_wellbore_mixing_conserves_mass():
    from types import SimpleNamespace

    from simulator.blackoil_fim import BlackOilWells

    # скважина 0: перфорации 0,1 закачивают воду, 2 — обратный поток нефти и газа; скважина 1 — не трогается
    fake = SimpleNamespace(conn_well=torch.tensor([0, 0, 0, 1]), n_w=2, device="cpu")
    q = torch.tensor([[0.0, -3.0, 0.0], [0.0, -1.0, 0.0], [0.5, 0.2, 2.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    back = torch.tensor([False, False, True, False])
    into = torch.tensor([True, True, False, False])
    out = BlackOilWells._wellbore_mixing(fake, q, back, into)
    # обратный поток по-прежнему выходит из своей ячейки, а по скважине в сумме — только закачка воды
    assert torch.equal(out[2], q[2]) and torch.equal(out[3], q[3])
    assert torch.allclose(out[:3].sum(0), torch.tensor([0.0, -4.0, 0.0], dtype=torch.float64))
    # возврат распределён пропорционально расходу закачивающих перфораций (3:1)
    assert torch.allclose(out[0] - q[0], 3.0 * (out[1] - q[1]))


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
