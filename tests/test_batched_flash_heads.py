"""Independent midpoint-density oracle for single-connection well batching."""
from types import SimpleNamespace

import pytest
import torch

from simulator import blackoil_fim as bo


@pytest.mark.parametrize('warm', [False, True])
@pytest.mark.parametrize('zero_mobility', [False, True])
def test_batch_preserves_pressure_density_and_schedule_changes(monkeypatch, warm, zero_mobility):
    # Deliberately permuted ownership; closed, stopped and injection connections
    # must not acquire the flowing producer's flashed density.
    owners = torch.tensor([2, 0, 4, 1, 3])
    cells = torch.tensor([4, 2, 0, 3, 1])
    depths = torch.tensor([100., -80., 50., 120., 90.], dtype=torch.float64)
    mix = torch.tensor([600., 650., 700., 750., 800.], dtype=torch.float64)
    old = torch.tensor([1e5, -2e5, 3e5, 4e5, 5e5], dtype=torch.float64)
    pb = torch.tensor([2e6, 3e6, 4e6, 5e6, 6e6], dtype=torch.float64)
    v = torch.zeros((bo.N_PROPS, 5), dtype=torch.float64)
    v[bo.I_P] = 8e6
    for key in (bo.I_MOO, bo.I_MWW, bo.I_MGG):
        v[key] = 0. if zero_mobility else 100.
    fake = SimpleNamespace(
        conn_cell=cells, conn_well=owners, conn_wi=torch.ones(5),
        conn_sched=torch.tensor([1., 1., 0., 1., 1.]),
        rho_sc=torch.tensor([800., 1000., 1.], dtype=torch.float64),
        n_w=5, device=torch.device('cpu'), p_bh=pb if warm else None,
        conn_head=old, conn_dz=depths,
        wells=[SimpleNamespace(type='injector' if i == 1 else 'producer', stopped=i == 3)
               for i in range(5)],
        well_conn_order=[(owners == i).nonzero().view(-1) for i in range(5)],
        pvt=SimpleNamespace(at=lambda idx: idx))
    calls = []
    def volumes(idx, p, qo, qw, qg):
        calls.append(idx.numel())
        mass = qo*800. + qw*1000. + qg
        density = 500. + 10*idx + 1e-5*p
        return mass/density, torch.zeros_like(p), torch.zeros_like(p)
    monkeypatch.setattr(bo, 'surface_to_reservoir', volumes)
    for opened in (False, True):
        fake.conn_sched[2] = float(opened)
        calls.clear()
        actual = bo.BlackOilWells._flash_heads(fake, v, mix)
        expected = []
        for i, owner in enumerate(owners.tolist()):
            h = float(mix[i])*bo.G*float(depths[i])
            if owner not in (1, 3) and float(fake.conn_sched[i]) > 0:
                base = float(pb[owner]) if warm else float(v[bo.I_P, cells[i]])-h
                h = float(old[i]) if warm else h
                for _ in range(2):
                    density = (float(mix[i]) if zero_mobility else
                               500.+10*int(cells[i])+1e-5*max(base+.5*h, 1e5))
                    h = density*bo.G*float(depths[i])
            expected.append(h)
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=1e-14, atol=1e-9)
        assert calls == [3 if opened else 2]*2
