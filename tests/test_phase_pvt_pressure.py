"""PVT must use phase pressure; phase switching must preserve nonlinear mass."""
import copy
from types import SimpleNamespace

import pytest
import torch

from test_blackoil_fim import BASE, _objects
from test_wet_gas import _wet_deck
from simulator.blackoil_fim import BlackOilFIMSolver, I_B, I_RV
from simulator.eclipse_deck import EclipseCase
from simulator.simulation import Simulator


class CapillaryModel:
    def __call__(self, sw, sg):
        return sw.square(), (1-sw-sg).square(), sg.square()

    def pc(self, sw, sg):
        return 2e6 * sw, 1e7 * sg


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_water_and_dry_gas_pvt_use_their_phase_pressures(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    dev = torch.device(device)
    cfg = copy.deepcopy(BASE)
    cfg['reservoir']['dimensions'] = [1, 1, 1]
    cfg['wells'] = []
    res, fluid, wells = _objects(cfg, dev)
    s = BlackOilFIMSolver(res, fluid, wells, cfg['simulation'])
    s.kr = CapillaryModel()
    p, sw, sg = (torch.tensor([v], dtype=torch.float64, device=dev) for v in (18e6, .4, .2))
    sat = torch.tensor([True], device=dev)
    values, derivatives = s.cell_state(p, sw, sg, sat)
    if dev.type == 'cuda':
        assert s._graphs, 'Phase-pressure PVT must retain CUDA graph execution'
    bw = s.pvt.water(p - 2e6*sw)[0]
    bg = s.pvt.gas(p + 1e7*sg)[0]
    torch.testing.assert_close(values[I_B+1], bw, rtol=1e-12, atol=0.)
    torch.testing.assert_close(values[I_B+2], bg, rtol=1e-12, atol=0.)
    # Capillary pressure couples saturation to density, beyond relative permeability.
    h = 1e-6
    water_fd = (s.pvt.water(p-2e6*(sw+h))[0] - s.pvt.water(p-2e6*(sw-h))[0])/(2*h)
    gas_fd = (s.pvt.gas(p+1e7*(sg+h))[0] - s.pvt.gas(p+1e7*(sg-h))[0])/(2*h)
    torch.testing.assert_close(derivatives[I_B+1, :, 1], water_fd, rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(derivatives[I_B+2, :, 2], gas_fd, rtol=1e-6, atol=1e-10)


def test_wet_gas_rv_and_bg_use_gas_pressure(tmp_path):
    case = EclipseCase(_wet_deck(tmp_path, 4e-5))
    res, fluid, wells, params = case.build()
    s = Simulator(res, fluid, wells, params, device=torch.device('cpu')).fim_solver
    s.kr = CapillaryModel()
    p = torch.full((s.n_cells,), 20e6, dtype=torch.float64)
    sw = torch.full_like(p, .3)
    sg = torch.full_like(p, .2)
    sat = torch.ones_like(p, dtype=torch.bool)
    ug = torch.zeros_like(sat)
    values = s.cell_values(p, sw, sg, sat, ug)
    pg = p + 1e7*sg
    rv = s.pvt.rv_sat(pg)
    bg = s.pvt.gas(pg, rv, sat)[0]
    torch.testing.assert_close(values[I_RV], rv, rtol=1e-12, atol=0.)
    torch.testing.assert_close(values[I_B+2], bg, rtol=1e-12, atol=0.)


def test_switch_to_sg_conserves_nonlinear_gas_mass():
    # A compressible gas at saturation-dependent capillary pressure yields
    # nonlinear gas mass. A single linear interpolation misses the root.
    sw = torch.tensor([.3], dtype=torch.float64)
    p = torch.tensor([1e7], dtype=torch.float64)
    star = torch.tensor([.3], dtype=torch.float64)
    target = 1-sw + 4*star + 2*star.square()
    old_x = target / (1-sw)
    old_sat = torch.tensor([False])
    new_sat = ~old_sat

    def masses(p, sw, x, sat, ug):
        gas = torch.where(sat, 1-sw + 4*x + 2*x.square(), (1-sw)*x)
        return torch.stack((torch.zeros_like(gas), sw, gas))

    fake = SimpleNamespace(masses=masses)
    switched = BlackOilFIMSolver._conservative_switch(
        fake, p, sw, old_x, old_sat, None, new_sat, None, new_sat)
    torch.testing.assert_close(switched, star, rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(masses(p, sw, switched, new_sat, None)[2], target,
                               rtol=1e-12, atol=1e-13)
