"""Finite junction transport and vector support reactions."""
import copy
import math

import pytest
import torch

from test_surface_pr_multifluid_pipe import setup, initial
from surface.pr_multifluid_network import PRMultifluidNetwork
from surface.pr_multifluid_pipe import PRMultifluidPipeState


def network(*, dense=128, device='cpu', three=False):
    source = setup(three=three).sources[0]
    if device != 'cpu':
        source = type(source).from_dict(source.to_dict(), device=device)
    faces = []
    for a, pipe, b, direction in [('A', 'x', 'B', (1.,0.,0.)), ('A','y','C',(0.,1.,0.)), ('C','d','B',(1/math.sqrt(2),-1/math.sqrt(2),0.))]:
        for left, right in [(a,pipe),(pipe,b)]:
            faces.append(dict(source=left,target=right,area_m2=0.01,direction=direction))
    return PRMultifluidNetwork([source]*6, [0.001]*6, ['A','x','B','y','C','d'], faces, ['A','B','C'], dense_limit=dense)


def start(model, pulse=True):
    # Build valid finite stocks with common P/T and zero junction velocity.
    state = initial(model, velocity=0.)
    if not pulse:
        return state
    cells = []
    for i, cell in enumerate(state.cells):
        cells.append([dict(pressure_pa=p.pressure_pa+(1000 if i == 0 else 0), temperature_k=p.temperature_k,
                           mole_fractions=[1. if j == k else 0. for j in range(3)], volume_m3=p.volume_m3,
                           velocity_m_s=0.) for k,p in enumerate(cell.phases)])
    return model.initialize(cells)


def test_branched_finite_junction_transport_and_restart():
    model = network()
    state = start(model)
    end, report = model.step(state, 1e-6)
    assert max(map(abs,report['conservation_residuals'])) < 1e-7
    assert any(abs(p.momentum_kg_m_s)>1e-12 for i,c in enumerate(end.cells) if i not in model.supported_cells for p in c.phases)
    assert all(p.momentum_kg_m_s == 0 for i in model.supported_cells for p in end.cells[i].phases)
    assert end.cells[0].phases[0].component_moles != state.cells[0].phases[0].component_moles
    restored = PRMultifluidNetwork.from_dict(model.to_dict())
    resumed = PRMultifluidPipeState.from_dict(end.to_dict())
    a,_ = model.step(end,1e-6)
    b,_ = restored.step(resumed,1e-6)
    assert a.to_dict() == b.to_dict()
    impulse = report['wall_impulse_kg_m_s']
    assert abs(impulse[0])+abs(impulse[1]) > 1e-12
    assert impulse[2] == 0


def test_uniform_rest_is_exact_fixed_point():
    model=network()
    state=start(model,False)
    end,report=model.transport_step(state,1e-5)
    for a,b in zip(state.cells,end.cells):
        for p,q in zip(a.phases,b.phases):
            assert q.pressure_pa == pytest.approx(p.pressure_pa,abs=1e-7)
            assert q.momentum_kg_m_s == pytest.approx(0.,abs=1e-12)
    assert max(map(abs,report['conservation_residuals'])) < 1e-7


def test_global_sparse_dense_and_off_branch_jacobian():
    dense,sparse=network(),network(dense=1)
    old=start(dense)
    other=PRMultifluidPipeState.from_dict({**old.to_dict(),'model_id':sparse.model_id})
    a,_=dense.transport_step(old,1e-6)
    b,_=sparse.transport_step(other,1e-6)
    for x,y in zip(a.cells,b.cells):
        for p,q in zip(x.phases,y.phases):
            assert q.pressure_pa == pytest.approx(p.pressure_pa,abs=1e-5)
    y,_,_,evaluate=dense._problem(old,1e-6)
    jac=torch.func.jacfwd(lambda v:evaluate(v)[0])(y)
    rows,cols=evaluate.sparsity
    mask=torch.zeros_like(jac,dtype=torch.bool)
    mask[rows,cols]=True
    assert float(jac[~mask].abs().max()) == 0
    width=y.numel()//6
    assert float(jac[:width,3*width:4*width].abs().max()) > 0


def test_geometry_rejects_unsupported_turn_and_disconnected_cells():
    model=network()
    raw=copy.deepcopy(model.to_dict())
    raw['faces'][1]['direction']=(0.,1.,0.)
    with pytest.raises(ValueError,match='Turning'):
        PRMultifluidNetwork.from_dict(raw)
    raw=copy.deepcopy(model.to_dict())
    raw['supported_nodes']=('A','B')
    with pytest.raises(ValueError,match='Turning'):
        PRMultifluidNetwork.from_dict(raw)


def test_independent_scalar_pr_and_gauss_face_balances():
    import numpy as np
    from reference_pr_multifluid_network import reconstruct
    model=network()
    state=start(model)
    dt=1e-6
    end,report=model.transport_step(state,dt)
    old=reconstruct(model.to_dict(),state.to_dict(),model.fluid)
    new=reconstruct(model.to_dict(),end.to_dict(),model.fluid)
    assert np.max(np.abs(dt*new['force']-report['wall_impulse_kg_m_s'])) < 1e-10
    for i,(a,b) in enumerate(zip(old['packets'],new['packets'])):
        residual=(b['conserved']-a['conserved'])*model.volumes[i]+dt*new['phase_rate'][i]
        assert np.max(np.abs(residual[:,:-2])) < 1e-9
        assert np.max(np.abs(residual[:,-1])) < 1e-7
        if i not in model.supported_cells:
            assert np.max(np.abs(residual[:,-2])) < 1e-9
        else:
            assert np.max(np.abs(b['velocity'])) == 0
        alpha=b['alpha'][1:]-a['alpha'][1:]+dt*new['alpha_rate'][i]
        assert np.max(np.abs(alpha)) < 1e-10


@pytest.mark.parametrize("three", [False, True])
def test_adaptive_runner_and_network_checkpoint(three):
    from run_pr_multifluid_pipe import run
    model=network(three=three)
    state=start(model)
    command=dict(duration_s=2e-7, maximum_step_s=1e-7)
    config=dict(model=model.to_dict(), initial=dict(state=state.to_dict()), schedule=[command])
    result=run(config)
    assert result['state']['time_s'] == pytest.approx(2e-7, abs=1e-20)
    assert max(map(abs,result['conservation_residuals'])) < 1e-7
    continued=run(dict(model=model.to_dict(),initial=dict(state=result['state']),schedule=[command]))
    assert continued['state']['time_s'] == pytest.approx(4e-7,abs=1e-20)
