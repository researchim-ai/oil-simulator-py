from types import SimpleNamespace
import pytest
import torch
from simulator.adaptive_amg import AdaptiveAMG
from test_fim_facility_coupling import build


def dummy():
    return SimpleNamespace(builder=SimpleNamespace(indptr=torch.tensor([0,1]),indices=torch.tensor([0])),
        wells=SimpleNamespace(conn_cell=torch.tensor([0]),conn_well=torch.tensor([0])),_amg=object(),lin_accept=.1)


def record(p,s,rebuilt=True,iters=4,accepted=True):
    p.record(s,dict(amg_rebuilt=rebuilt,amg_levels=[1],iters=iters,converged=True,true_rel_res=1e-5),accepted)


def test_bounded_consecutive_reuse_and_quality():
    s=dummy();p=AdaptiveAMG()
    assert not p.reuse(s,10,0,0)
    record(p,s)
    for t in [10,20,30]:
        assert p.reuse(s,10,t,0)
        record(p,s,False)
    assert not p.reuse(s,10,40,0)
    record(p,s)
    assert p.reuse(s,10,50,0)
    record(p,s,False,40)
    assert p.refresh and not p.reuse(s,10,50,1)


def test_rejected_or_changed_step_and_structure_force_fresh():
    for kind in ['retry','jump','dt','structure','hierarchy','failure']:
        s=dummy();p=AdaptiveAMG();p.reuse(s,10,0,0);record(p,s)
        t,dt=10,10
        if kind=='retry':t=0
        if kind=='jump':t=100
        if kind=='dt':dt=30
        if kind=='structure':s.wells.conn_cell.add_(1)
        if kind=='hierarchy':s._amg=object()
        if kind=='failure':record(p,s,False,accepted=False)
        assert not p.reuse(s,dt,t,0),kind


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_step_retry_after_modified_matrix(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    a,b=build(device=device),build(device=device)
    b.solver.amg_rebuild='adaptive'
    original=b.solver.solve_linear
    injected=[]
    def solve(J,R,wsys,**kwargs):
        answer=original(J,R,wsys,**kwargs)
        if not kwargs['rebuild_amg'] and not injected:
            injected.append(True)
            return None
        return answer
    monkeypatch.setattr(b.solver,'solve_linear',solve)
    for t in [0.,60.]:
        assert a.solver.step(60.,current_time=t)
        assert b.solver.step(60.,current_time=t)
    assert injected and b.solver._adaptive_amg_policy.retries==1
    torch.testing.assert_close(a.solver.fluid.pressure,b.solver.fluid.pressure,rtol=1e-7,atol=1e-3)
    torch.testing.assert_close(a.solver.fluid.s_w,b.solver.fluid.s_w,rtol=0,atol=1e-7)
    # Process-local policy must not survive checkpoint serialization.
    assert b.solver.__getstate__()['_adaptive_amg_policy'] is None
