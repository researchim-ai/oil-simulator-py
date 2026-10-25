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
    fake = SimpleNamespace(conn_well=torch.tensor([0, 0, 0, 1]), n_w=2, device="cpu",
                           inj_phase=torch.tensor([WAT, WAT]))
    q = torch.tensor([[0.0, -3.0, 0.0], [0.0, -1.0, 0.0], [0.5, 0.2, 2.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    back = torch.tensor([False, False, True, False])
    into = torch.tensor([True, True, False, False])
    out, _ = BlackOilWells._wellbore_mixing(fake, q, back, into)
    # обратный поток по-прежнему выходит из своей ячейки; масса закачки через перфорацию — по её расходу,
    # меняется только состав; по скважине в сумме — нетто-закачка воды (−4 + 2.7)
    assert torch.equal(out[2], q[2]) and torch.equal(out[3], q[3])
    assert torch.allclose(out[:2].sum(1), q[:2].sum(1))
    assert torch.allclose(out[:3].sum(0), torch.tensor([0.0, -1.3, 0.0], dtype=torch.float64))
    # возврат распределён пропорционально расходу закачивающих перфораций (3:1)
    assert torch.allclose(out[0] - q[0], 3.0 * (out[1] - q[1]))


def test_wellbore_mixing_jacobian_matches_finite_differences():
    from types import SimpleNamespace

    from simulator.blackoil_fim import BlackOilWells

    torch.manual_seed(0)
    n = 4   # перфорации одной скважины: 0, 1, 3 закачивают, 2 — обратный поток
    A = torch.rand(n, 3, 3, dtype=torch.float64) + 0.1
    A[[0, 1, 3]] *= -1.0
    cb = torch.rand(n, 3, dtype=torch.float64)
    x0 = torch.rand(n, 3, dtype=torch.float64) + 1.0
    back = torch.tensor([False, False, True, False])
    into = ~back
    pc, pd = torch.arange(n).repeat_interleave(n), torch.arange(n).repeat(n)
    fake = SimpleNamespace(conn_well=torch.zeros(n, dtype=torch.long), n_w=1, device="cpu",
                           inj_phase=torch.tensor([WAT]),
                           pair_c=pc, pair_d=pd, pair_pos=torch.arange(n * n), _mix=None)

    def mixed(x, pb=0.0):
        q = torch.einsum("cij,cj->ci", A, x) + cb * pb
        return BlackOilWells._wellbore_mixing(fake, q, back, into, A.clone(), cb.clone())

    _, dqb = mixed(x0)
    pos, blk = BlackOilWells._mixing_blocks(fake)
    J = torch.zeros(n, n, 3, 3, dtype=torch.float64)
    J[pc, pd] = torch.where((pc == pd).view(-1, 1, 1), A[pc], torch.zeros_like(A[pc]))
    J.view(-1, 3, 3).index_add_(0, pos, blk)
    h = 1e-6
    for d_ in range(n):
        for j in range(3):
            dx = torch.zeros_like(x0)
            dx[d_, j] = h
            fd = (mixed(x0 + dx)[0] - mixed(x0 - dx)[0]) / (2 * h)
            assert torch.allclose(J[:, d_, :, j], fd, rtol=1e-6, atol=1e-8)
    fd_pb = (mixed(x0, h)[0] - mixed(x0, -h)[0]) / (2 * h)
    assert torch.allclose(dqb, fd_pb, rtol=1e-6, atol=1e-8)


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
