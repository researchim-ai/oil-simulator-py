import copy
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator
from simulator.manual_assembly_solver import ManualFIMSolver

BASE = {
    "simulation": {"solver_type": "fully_implicit", "newton_max_iter": 20, "max_time_step_attempts": 4,
                   "dt_reduction_factor": 2.0, "dt_increase_factor": 1.0},
    "reservoir": {"dimensions": [8, 6, 3], "grid_size": [20.0, 20.0, 4.0],
                  "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 0.2},
    "fluid": {"pressure": 20.0, "s_w": 0.2, "mu_oil": 1.0, "mu_water": 0.5,
              "rho_oil": 850.0, "rho_water": 1000.0, "c_oil": 1e-5, "c_water": 1e-5, "c_rock": 1e-5,
              "s_wr": 0.2, "s_or": 0.2, "pc_scale": 0.0},
    "wells": [
        {"name": "I", "type": "injector", "i": 0, "j": 0, "k": 0, "radius": 0.1,
         "control_type": "rate", "control_value": 30.0, "injected_phase": "water",
         "perforations": [[0, 0, 0], [0, 0, 1], [0, 0, 2]]},
        {"name": "P", "type": "producer", "i": 7, "j": 5, "k": 0, "radius": 0.1,
         "control_type": "rate", "control_value": 30.0, "surface_phase": "liquid",
         "perforations": [[7, 5, 0], [7, 5, 1]]},
    ],
}


def _objects(cfg, dev=torch.device('cpu')):
    res = Reservoir(cfg['reservoir'], device=dev)
    fluid = Fluid(cfg['fluid'], res, device=dev)
    wells = WellManager(cfg['wells'], res)
    return res, fluid, wells


def test_report_tail_never_exceeds_maximum_time_step():
    cfg = copy.deepcopy(BASE)
    cfg['wells'] = []
    cfg['reservoir']['dimensions'] = [1, 1, 1]
    cfg['simulation'].update(initial_time_step_days=1., max_time_step_days=1.)
    res, fluid, wm = _objects(cfg)
    sim = Simulator(res, fluid, wm, cfg['simulation'], device=torch.device('cpu'))
    accepted = [0.]
    sim.substep_hook = accepted.append
    for _ in range(2):
        assert sim.run_step(2.2 * 86400.)
    assert np.max(np.diff(accepted)) <= 1. + 1e-12
    assert abs(accepted[-1] - 4.4) < 1e-12


def test_well_schur_blocks_match_finite_differences():
    cfg = copy.deepcopy(BASE)
    res, fluid, wm = _objects(cfg)
    s = ManualFIMSolver(res, fluid, wm, cfg['simulation'])
    dt = 86400.0
    gen = torch.Generator().manual_seed(3)
    p = fluid.pressure.double().reshape(-1) + 3e5 * torch.rand(s.n_cells, generator=gen, dtype=torch.float64)
    sw = (0.2 + 0.4 * torch.rand(s.n_cells, generator=gen, dtype=torch.float64))
    m_o, m_w = s.calc_masses(p, sw)
    s.wells.begin_step(p, sw, s.cell_props(p, sw), 0.0)
    p_bh = torch.tensor([21e6, 18e6], dtype=torch.float64)

    def full(pp, ss, pb):
        R, _ = s.assemble_system(pp, ss, m_o, m_w, dt, p_bh=pb)
        return R.clone(), s._last_well_sys["r_w"].clone()

    R0, _ = s.assemble_system(p, sw, m_o, m_w, dt, p_bh=p_bh)
    sys0 = {k: v.clone() for k, v in s._last_well_sys.items()}
    assert bool(sys0["is_rate"].all())

    for w in range(2):
        h = 10.0
        pb_p, pb_m = p_bh.clone(), p_bh.clone()
        pb_p[w] += h
        pb_m[w] -= h
        (Rp, rwp), (Rm, rwm) = full(p, sw, pb_p), full(p, sw, pb_m)
        dR = ((Rp - Rm) / (2 * h)).view(-1, 2)
        m = s.wells.conn_well == w
        cells = s.wells.conn_cell[m]
        assert torch.allclose(dR[cells], sys0["Cv"][m], rtol=1e-6, atol=0)
        assert abs(((rwp - rwm) / (2 * h))[w] - sys0["D"][w]) <= 1e-6 * abs(sys0["D"][w])

    for ci, c in enumerate(s.wells.conn_cell.tolist()):
        w = int(s.wells.conn_well[ci])
        for var, h in ((0, 10.0), (1, 1e-6)):
            pp, sp_, pm, sm = p.clone(), sw.clone(), p.clone(), sw.clone()
            (pp if var == 0 else sp_)[c] += h
            (pm if var == 0 else sm)[c] -= h
            fd = (full(pp, sp_, p_bh)[1] - full(pm, sm, p_bh)[1]) / (2 * h)
            assert abs(fd[w] - sys0["Bv"][ci, var]) <= 1e-6 * abs(fd[w]) + 1e-30


def _run(cfg, solver_type, dt_days, days):
    cfg = copy.deepcopy(cfg)
    cfg['simulation']['solver_type'] = solver_type
    res, fluid, wm = _objects(cfg)
    sim = Simulator(res, fluid, wm, cfg['simulation'], device=torch.device('cpu'))
    for _ in range(int(round(days / dt_days))):
        assert sim.run_step(dt_days * 86400.0)
    return sim, fluid


def test_fim_matches_impes():
    cfg = copy.deepcopy(BASE)
    cfg['reservoir']['dimensions'] = [12, 12, 1]
    cfg['reservoir']['k_vertical_fraction'] = 1.0
    cfg['wells'] = [
        {"name": "I", "type": "injector", "i": 2, "j": 6, "k": 0, "radius": 0.1,
         "control_type": "rate", "control_value": 20.0, "injected_phase": "water"},
        {"name": "P", "type": "producer", "i": 9, "j": 6, "k": 0, "radius": 0.1,
         "control_type": "rate", "control_value": 20.0},
    ]
    _, f_impes = _run(cfg, 'impes', 0.02, 4.0)
    _, f_fim = _run(cfg, 'fully_implicit', 0.25, 4.0)
    p_i, p_f = f_impes.pressure.double().numpy(), f_fim.pressure.double().numpy()
    dp = np.abs(p_i - p_f)
    drawdown = p_i.max() - p_i.min()
    assert drawdown > 0.2e6
    assert dp.max() < 0.05 * drawdown
    assert np.abs(f_impes.s_w.double().numpy() - f_fim.s_w.double().numpy()).max() < 0.02


def test_producer_switches_to_bhp_limit_and_stays_physical():
    cfg = copy.deepcopy(BASE)
    cfg['wells'][1]['control_value'] = 200.0
    cfg['wells'][1]['bhp_min'] = 15.0
    sim, fluid = _run(cfg, 'fully_implicit', 1.0, 6.0)
    prod = sim.fim_solver.wells.wells[1]
    assert prod.last_mode == 'bhp_limit'
    assert abs(prod.bhp_mpa - 15.0) < 1e-6
    assert prod.last_surface_rates['liquid'] < 200.0
    assert float(fluid.pressure.min()) > 0


def test_multiperforation_rate_target_met():
    sim, fluid = _run(BASE, 'fully_implicit', 1.0, 3.0)
    inj, prod = sim.fim_solver.wells.wells
    assert len(inj.connection_wi) == 3 and len(prod.connection_wi) == 2
    assert abs(inj.last_surface_rates['water'] + 30.0) < 30.0 * 1e-4
    assert abs(prod.last_surface_rates['liquid'] - 30.0) < 30.0 * 1e-4
