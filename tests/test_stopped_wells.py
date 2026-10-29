"""A stopped well redistributes fluid internally; SHUT disconnects it."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator.fluid import Fluid
from simulator.reservoir import Reservoir
from simulator.well import WellManager
from simulator.simulation import Simulator


def _model(kind='producer', shut=False, device=None):
    dev = torch.device('cpu') if device is None else torch.device(device)
    res = Reservoir(dict(dimensions=[2, 1, 1], grid_size=[30., 30., 5.],
                         porosity=.2, permeability=0.), device=dev)
    fluid = Fluid(dict(pressure=18., s_w=.3, s_g=.1,
                       pvt_path=str(Path(__file__).resolve().parents[1] / 'configs/pvt/pvt_synthetic.json'),
                       rho_oil=850., rho_water=1000., rho_gas=.9, c_rock=1e-5), res, device=dev)
    fluid.pressure[:, 0, 0] = torch.tensor([13.7e6, 19.3e6])
    fluid.s_w[:, 0, 0] = torch.tensor([.52, .37])
    wm = WellManager([dict(name='S', type=kind, i=0, j=0, k=0, radius=.1,
                           perforations=[[0,0,0], [1,0,0]], well_index=1e-12,
                           control_type='shut' if shut else 'rate', control_value=0.,
                           stopped=not shut, start_mode=None if shut else 'stop',
                           surface_phase='liquid', injected_phase='water')], res)
    sim = Simulator(res, fluid, wm, dict(solver_type='fully_implicit', fim_model='blackoil',
                    newton_max_iter=25, newton_tol_cnv=1e-7, newton_tol_mb=1e-10,
                    newton_relaxed_max_pv_fraction=0., max_time_step_attempts=8), device=dev)
    return sim


@pytest.mark.parametrize('kind', ['producer', 'injector'])
def test_stopped_well_has_zero_surface_rates_and_conserves_each_component(kind):
    sim = _model(kind)
    s, f = sim.fim_solver, sim.fluid
    def mass():
        return s.masses(f.pressure.reshape(-1), f.s_w.reshape(-1), s.x, s.sat, s.ug).sum(1)
    before = mass().clone()
    oldp = f.pressure.clone()
    assert sim.run_step(.01 * 86400)
    torch.testing.assert_close(mass(), before, rtol=1e-9, atol=1e-5)
    w = s.wells.wells[0]
    assert w.last_mode == 'stop'
    assert max(abs(x) for x in w.last_surface_rates.values()) < 1e-5
    assert max(abs(x) for x in w.last_net_connection_rates.values()) < 1e-5
    assert max(w.last_surface_prod.values()) > .01
    assert float((f.pressure-oldp).abs().max()) > 1e3


def test_shut_well_cannot_redistribute_fluid():
    sim = _model(shut=True)
    oldp, oldsw = sim.fluid.pressure.clone(), sim.fluid.s_w.clone()
    assert sim.run_step(.01 * 86400)
    torch.testing.assert_close(sim.fluid.pressure, oldp, rtol=0, atol=1e-8)
    torch.testing.assert_close(sim.fluid.s_w, oldsw, rtol=0, atol=1e-12)


def test_stopped_well_cannot_hide_opposite_surface_component_rates():
    from simulator.blackoil_fim import I_B, N_PROPS
    s = _model().fim_solver
    v = torch.zeros((N_PROPS, 2), dtype=torch.float64)
    v[I_B:I_B+3] = 1.
    ws = dict(is_rate=torch.tensor([True]), is_stopped=torch.tensor([True]),
              target=torch.zeros(1), r_w=torch.zeros(1),
              component_rates=torch.tensor([[1., 0., -1.]], dtype=torch.float64))
    ok, _ = s.convergence(torch.zeros(6, dtype=torch.float64), 1., torch.ones(2),
                           ws, torch.tensor([1e7]), v)
    assert not ok


@pytest.mark.parametrize('msw', [False, True])
def test_producer_mixing_full_jacobian_and_well_blocks_match_finite_differences(msw):
    sim = _model()
    s = sim.fim_solver
    p = s.fluid.pressure.reshape(-1).double()
    sw = s.fluid.s_w.reshape(-1).double()
    x, sat, ug = s.x.clone(), s.sat.clone(), s.ug.clone()
    old = s.masses(p,sw,x,sat,ug)
    state = s.cell_state(p,sw,x,sat,ug)
    s.wells.begin_step(*state, 0.)
    if msw:
        # Upper connection receives fluid from the producing lower segment.
        s.wells.msw = {0: {'sub': torch.tensor([[1.,1.],[0.,1.]],dtype=torch.float64)}}
    pb = torch.tensor([16.7e6],dtype=torch.float64)
    _, J, wells = s.assemble_bo(p,sw,x,sat,old,864.,pb,ug=ug)
    rows = torch.repeat_interleave(torch.arange(2),s.builder.indptr[1:]-s.builder.indptr[:-1])
    cols = s.builder.indices
    assert bool(J[(rows==0)&(cols==1)].abs().max()>0)
    for c in range(2):
        for var, h in enumerate([10.,1e-6,1e-6]):
            values=[];wr=[]
            for sign in [1.,-1.]:
                z=[p.clone(),sw.clone(),x.clone()];z[var][c]+=sign*h
                R,_,ws=s.assemble_bo(*z,sat,old,864.,pb,ug=ug)
                values.append(R.view(2,3));wr.append(ws['r_w'])
            fd=(values[0]-values[1])/(2*h)
            an=torch.zeros_like(fd);m=cols==c;an[rows[m]]=J[m][:,:,var]
            scale=fd.abs().max().clamp_min(1e-30)
            assert float((fd-an).abs().max()/scale)<1e-5
            fdw=(wr[0]-wr[1])/(2*h)
            torch.testing.assert_close(fdw[0],wells['Bv'][c,var],rtol=1e-5,atol=1e-10)
    vals=[];wr=[]
    for sign in [1.,-1.]:
        R,_,ws=s.assemble_bo(p,sw,x,sat,old,864.,pb+sign*10.,ug=ug)
        vals.append(R.view(2,3));wr.append(ws['r_w'])
    torch.testing.assert_close((vals[0]-vals[1])/20.,wells['Cv'],rtol=1e-5,atol=1e-12)
    torch.testing.assert_close((wr[0]-wr[1])/20.,wells['D'],rtol=1e-5,atol=1e-12)
