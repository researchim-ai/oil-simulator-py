"""RESV conversion uses one current hydrocarbon PV measure for P, Rs and Rv."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator.blackoil_fim import BlackOilFIMSolver


@pytest.mark.parametrize('all_water', [False, True])
def test_resv_averages_share_hydrocarbon_pore_volume(all_water):
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    p, sw = t([100., 200.]), t([1., 1.] if all_water else [.2, .8])
    state = SimpleNamespace(_pv=lambda p: t([2., 1.]), wells=SimpleNamespace(),
                            fluid=SimpleNamespace(rs=t([10., 100.]), rv=t([.001, .0001])),
                            pvt=SimpleNamespace(has_pvt=True), wet=True)
    BlackOilFIMSolver._update_resv_averages(state, p, sw)
    # Current PV is [2,1], hydrocarbon PV is [1.6,.2]. An entirely
    # water-filled field falls back to full PV, as RateConverter does.
    if all_water:
        expected = (400./3, 40., .0007)
    else:
        expected = (200./1.8, 20., .0009)
    assert (state.wells.p_avg, state.wells.rs_avg, state.wells.rv_avg) == pytest.approx(expected)


@pytest.mark.parametrize('gas_ratio', [0., 20., 160.])
def test_produced_resv_uses_stream_composition_and_full_gradient(gas_ratio):
    from test_stopped_wells import _model
    wells = _model().fim_solver.wells
    wells.p_avg, wells.rs_avg, wells.rv_avg = 18e6, 80., 0.
    q = torch.tensor([.02, .005, .02 * gas_ratio], dtype=torch.float64)
    mass = q * wells.rho_sc
    value, gradient = wells._producer_resv_rate(0, mass)
    p = torch.tensor([18e6], dtype=torch.float64)
    rs = torch.tensor([min(gas_ratio, 80.)], dtype=torch.float64)
    bo = wells.pvt.oil(p, rs, rs >= wells.pvt.rs_sat(p))[0]
    bw, bg = wells.pvt.water(p)[0], wells.pvt.gas(p)[0]
    expected = q[0] / bo + q[1] / bw + (q[2] - rs * q[0]) / bg
    torch.testing.assert_close(value, expected[0], rtol=1e-12, atol=1e-14)
    # Away from the nonnegative-GOR kink, derivatives include changing oil PVT.
    if gas_ratio > 0:
        for component in range(3):
            h = 1e-6 * max(float(mass[component]), 1e-3)
            delta = torch.zeros_like(mass); delta[component] = h
            fd = (wells._producer_resv_rate(0, mass + delta)[0]
                  - wells._producer_resv_rate(0, mass - delta)[0]) / (2*h)
            torch.testing.assert_close(gradient[component], fd, rtol=1e-6, atol=1e-10)
    zero, dz = wells._producer_resv_rate(0, torch.zeros_like(mass))
    assert zero == 0 and bool(torch.isfinite(dz).all())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_resv_control_row_full_jacobian_matches_finite_differences(device, tmp_path):
    from test_stopped_wells import _model
    s = _model(device=device).fim_solver
    w = s.wells.wells[0]
    w.stopped, w.control_type, w.control_value = False, 'resv', 1000.
    s.wells.mode = ['rate']
    s.wells.p_avg, s.wells.rs_avg, s.wells.rv_avg = 18e6, 80., 0.
    p, sw = s.fluid.pressure.reshape(-1).double(), s.fluid.s_w.reshape(-1).double()
    x, sat, ug = torch.full_like(p, 20.), torch.zeros_like(s.sat), s.ug.clone()
    old = s.masses(p, sw, x, sat, ug)
    s.wells.begin_step(*s.cell_state(p, sw, x, sat, ug), 0.)
    pb = p.new_tensor([12e6])
    _, _, ws = s.assemble_bo(p, sw, x, sat, old, 864., pb, ug=ug)
    assert bool(ws['is_rate'][0])
    for cell in range(2):
        for var, h in enumerate([10., 1e-6, 1e-5]):
            values = []
            for sign in [1., -1.]:
                z = [p.clone(), sw.clone(), x.clone()]; z[var][cell] += sign*h
                values.append(s.assemble_bo(*z, sat, old, 864., pb, ug=ug)[2]['r_w'][0])
            fd = (values[0]-values[1])/(2*h)
            torch.testing.assert_close(ws['Bv'][cell, var], fd, rtol=1e-5, atol=1e-10)
    plus = s.assemble_bo(p, sw, x, sat, old, 864., pb+10., ug=ug)[2]['r_w'][0]
    minus = s.assemble_bo(p, sw, x, sat, old, 864., pb-10., ug=ug)[2]['r_w'][0]
    torch.testing.assert_close(ws['D'][0], (plus-minus)/20., rtol=1e-5, atol=1e-12)
    # AD-produced coefficients must remain plain tensors in a restart file.
    checkpoint = tmp_path/'resv.pt'
    torch.save(s, checkpoint)
    restored = torch.load(checkpoint, map_location=device, weights_only=False)
    after = restored.assemble_bo(p, sw, x, sat, old, 864., pb, ug=ug)[2]
    torch.testing.assert_close(after['r_w'], ws['r_w'], rtol=1e-12, atol=1e-14)


def test_gas_rich_resv_uses_actual_condensate_ratio(tmp_path):
    from test_stopped_wells import _model
    from test_wet_gas import _wet_deck
    from simulator.eclipse_deck import EclipseCase
    wells = _model().fim_solver.wells
    _, fluid, _, _ = EclipseCase(_wet_deck(tmp_path, 4e-5)).build()
    wells.pvt = fluid.blackoil_pvt
    p = torch.tensor([18e6], dtype=torch.float64)
    wells.p_avg = float(p[0])
    wells.rs_avg = float(.7*wells.pvt.rs_sat(p)[0])
    wells.rv_avg = float(.8*wells.pvt.rv_sat(p)[0])
    rv = .5*wells.rv_avg
    q = p.new_tensor([rv, .005, 1.])
    mass = q*wells.rho_sc
    value, gradient = wells._producer_resv_rate(0, mass)
    bg = wells.pvt.gas(p, p.new_tensor([rv]), torch.tensor([False]))[0]
    expected = q[2]/bg + q[1]/wells.pvt.water(p)[0]
    torch.testing.assert_close(value, expected[0], rtol=1e-10, atol=1e-14)
    for component in range(3):
        h = 1e-6*max(float(mass[component]), 1e-3)
        delta = torch.zeros_like(mass); delta[component] = h
        fd = (wells._producer_resv_rate(0, mass+delta)[0]
              - wells._producer_resv_rate(0, mass-delta)[0])/(2*h)
        torch.testing.assert_close(gradient[component], fd, rtol=1e-5, atol=1e-9)
