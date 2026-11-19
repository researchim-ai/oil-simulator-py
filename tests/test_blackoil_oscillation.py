"""A sole unconverged component must not bypass Newton cycle damping."""
from types import SimpleNamespace
import pytest
from simulator.blackoil_fim import BlackOilFIMSolver


def detects(history):
    return BlackOilFIMSolver._oscillating(SimpleNamespace(relax_rel_tol=.2),history)


@pytest.mark.parametrize('component',[0,1,2])
def test_single_unconverged_component_cycle(component):
    history=[[.1,.2,.3] for _ in range(3)]
    for row,value in zip(history,[1087.686848,6667.,1087.686848]):row[component]=value
    assert detects(history)


@pytest.mark.parametrize('history',[
    [[.1,.2,.3],[.2,.3,.4],[.1,.2,.3]],  # everything already converged
    [[.1,.2,1000],[.1,.2,500],[.1,.2,250]],  # monotone decrease
    [[.1,.2,100],[.1,.2,101],[.1,.2,100]],  # insignificant wobble
    [[.1,.2,100],[.1,.2,500]],  # insufficient history
    [[10,50,100],[5,20,600],[2,10,100]],  # retain multi-component rule
])
def test_do_not_damp_converged_monotone_or_unconfirmed_history(history):
    assert not detects(history)


def test_two_component_cycle_retains_existing_behavior():
    assert detects([[10,.2,100],[30,.1,600],[10,.05,100]])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_last_permitted_correction_is_checked_without_an_extra_solve(monkeypatch,device):
    import copy
    import torch
    from test_blackoil_fim import BASE, _objects
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    cfg=copy.deepcopy(BASE)
    cfg['simulation']['newton_relaxed_max_pv_fraction']=0.
    def build():
        res,fluid,wells=_objects(cfg,torch.device(device))
        return BlackOilFIMSolver(res,fluid,wells,cfg['simulation'])
    reference=build()
    assert reference.step(3600.,current_time=0.)
    required=reference.last_newton_iters
    assert required>0
    limited=build();limited.max_newton=required
    original=limited.solve_linear;calls=[]
    def solve(*a,**kw):
        calls.append(1)
        return original(*a,**kw)
    monkeypatch.setattr(limited,'solve_linear',solve)
    assert limited.step(3600.,current_time=0.)
    assert len(calls)==required==limited.last_newton_iters
    for name in ['pressure','s_w','s_g','s_o','rs','rv']:
        torch.testing.assert_close(getattr(limited.fluid,name),getattr(reference.fluid,name),rtol=0,atol=0)
    insufficient=build();insufficient.max_newton=required-1
    before={name:getattr(insufficient.fluid,name).clone() for name in ['pressure','s_w','s_g','rs']}
    assert not insufficient.step(3600.,current_time=0.)
    for name,value in before.items():
        torch.testing.assert_close(getattr(insufficient.fluid,name),value,rtol=0,atol=0)
