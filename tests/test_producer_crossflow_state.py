"""Immobile phases still carry nonzero Rs/Rv when well fluid enters a cell."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator import blackoil_fim as bo


@pytest.mark.parametrize('rs,rv', [(1., 0.), (0., .1)])
def test_backflow_volume_uses_state_ratios_with_zero_phase_mobility(rs, rv):
    rho = torch.tensor([800., 1000., 1.], dtype=torch.float64)
    fake = SimpleNamespace(conn_cell=torch.arange(2), conn_well=torch.zeros(2, dtype=torch.long),
                           rho_sc=rho, is_inj=torch.zeros(1, dtype=torch.bool), n_w=1, device='cpu', msw={})
    v = torch.zeros((bo.N_PROPS, 2), dtype=torch.float64)
    d = torch.zeros((bo.N_PROPS, 2, 3), dtype=torch.float64)
    v[bo.I_B:bo.I_B+3] = torch.tensor([[1.], [1.], [10.]])
    v[bo.I_LT] = 1.
    v[bo.I_RS], v[bo.I_RV] = rs, rv
    v[bo.I_MOO, 0], v[bo.I_MGO, 0], v[bo.I_RS, 0] = rho[0], 2 * rho[2], 2.
    if rs:
        v[bo.I_MGG, 1] = 10 * rho[2]
        assert v[bo.I_MOO, 1] == 0
    else:
        v[bo.I_MOO, 1] = rho[0]
        assert v[bo.I_MGG, 1] == 0
    # The absent phase's mobility ratio is undefined, but its Rs/Rv is not zero.
    q0 = torch.stack([rho * torch.tensor([1., 0., 2.]),
                      -torch.stack([v[bo.I_MOO, 1], v[bo.I_MWW, 1], v[bo.I_MGG, 1]])])
    dq = torch.zeros((2, 3, 3), dtype=torch.float64)
    dqb = torch.zeros((2, 3), dtype=torch.float64)
    q, _, _ = bo.BlackOilWells._producer_crossflow(
        fake, v, d, q0, dq, dqb, torch.tensor([1., -1.]),
        torch.tensor([False, True]), torch.ones(2))
    cm = torch.tensor([1/3, 0., 2/3], dtype=torch.float64)
    volume_ratio = (cm[0] - rv * cm[2]) / (1-rs*rv) + (cm[2] - rs * cm[0]) / (10*(1-rs*rv))
    torch.testing.assert_close(q[0], q0[0])
    torch.testing.assert_close(q[1], -rho * cm / volume_ratio)
    # The negative connection injects exactly one reservoir m3/s.
    stock = -q[1] / rho
    actual_volume = (stock[0]-rv*stock[2])/(1-rs*rv) + (stock[2]-rs*stock[0])/(10*(1-rs*rv))
    assert float(actual_volume) == pytest.approx(1.)
