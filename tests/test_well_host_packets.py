"""Keep control decisions, messages and weighted PVT values bitwise unchanged."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from well_packet_reference import legacy_update_controls, legacy_update_resv_averages

from simulator.blackoil_fim import BlackOilFIMSolver, BlackOilWells


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def test_control_packet_preserves_switches_and_messages(device):
    for sign in (1, -1):
        for mode in ('rate', 'bhp_limit', 'stop'):
            for exhausted in (0, 3):
                state = SimpleNamespace(
                    n_w=4, wells=[SimpleNamespace(name=f'W{i}') for i in range(4)],
                    network=None, _Q=None, switches=0, well_switches=[exhausted]*4,
                    step_targets=[('rate', 1.0)]*3+[('bhp', 2e6)], step_mode=[mode]*4,
                    bhp_limit=lambda w: 2e6, has_thp=lambda i: False, _sgn=lambda i, sign=sign: sign)
                before, after = copy.deepcopy(state), copy.deepcopy(state)
                p=torch.tensor([2e6-1, 2e6, 2e6+1, 2e6],dtype=torch.float64,device=device)
                q=torch.tensor([1-2e-9,1,1+2e-9,1],dtype=torch.float64,device=device)
                a=legacy_update_controls(before,p,q)
                b=BlackOilWells.update_controls(after,p,q)
                assert a==b
                for key in ('step_mode','switches','well_switches','pbh_override'):
                    assert getattr(before,key)==getattr(after,key)


def test_resv_packet_preserves_exact_reductions(device):
    gen=torch.Generator().manual_seed(72)
    for all_water in (False,True):
        for wet in (False,True):
            for has_pvt in (False,True):
                t=lambda a: a.to(device=device,dtype=torch.float64)
                p=t(1e7+1e6*torch.rand(103,generator=gen,dtype=torch.float64))
                sw=t(torch.ones(103) if all_water else torch.rand(103,generator=gen,dtype=torch.float64))
                pv=t(torch.rand(103,generator=gen,dtype=torch.float64))
                state=SimpleNamespace(_pv=lambda p, pv=pv:pv,wells=SimpleNamespace(),wet=wet,
                    fluid=SimpleNamespace(rs=t(torch.rand(103,generator=gen)),rv=t(torch.rand(103,generator=gen))),
                    pvt=SimpleNamespace(has_pvt=has_pvt))
                before,after=copy.deepcopy(state),copy.deepcopy(state)
                legacy_update_resv_averages(before,p,sw)
                BlackOilFIMSolver._update_resv_averages(after,p,sw)
                assert vars(before.wells)==vars(after.wells)
