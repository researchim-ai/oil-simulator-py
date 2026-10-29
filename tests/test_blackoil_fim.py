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
from simulator.blackoil_fim import BlackOilFIMSolver, I_ACC

PVT = os.path.join(os.path.dirname(__file__), '..', 'configs', 'pvt', 'pvt_synthetic.json')

BASE = {
    "simulation": {"solver_type": "fully_implicit", "fim_model": "blackoil", "newton_max_iter": 25,
                   "max_time_step_attempts": 6, "dt_reduction_factor": 2.0, "dt_increase_factor": 1.0},
    "reservoir": {"dimensions": [6, 5, 3], "grid_size": [30.0, 30.0, 5.0],
                  "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 0.2},
    "fluid": {"pressure": 18.0, "s_w": 0.25, "s_g": 0.0, "pvt_path": PVT, "rho_oil": 850.0,
              "rho_water": 1000.0, "rho_gas": 0.9, "c_rock": 1e-5,
              "relative_permeability": {"nw": 2.0, "no": 2.0, "ng": 2.0, "sw_cr": 0.2, "so_r": 0.15,
                                        "sg_cr": 0.02}},
    "wells": [
        {"name": "GI", "type": "injector", "i": 0, "j": 0, "k": 0, "radius": 0.1,
         "control_type": "rate", "control_value": 2000.0, "injected_phase": "gas", "bhp_max": 40.0},
        {"name": "P", "type": "producer", "i": 5, "j": 4, "k": 0, "radius": 0.1,
         "control_type": "rate", "control_value": 20.0, "surface_phase": "oil", "bhp_min": 5.0,
         "perforations": [[5, 4, 0], [5, 4, 1], [5, 4, 2]]},
    ],
}


def _objects(cfg, dev=torch.device('cpu')):
    res = Reservoir(cfg['reservoir'], device=dev)
    fluid = Fluid(cfg['fluid'], res, device=dev)
    wells = WellManager(cfg['wells'], res)
    return res, fluid, wells


def _mixed_state(s, seed=1):
    """Половина ячеек с газом, половина недонасыщенных; значения вдали от изломов ОФП."""
    n = s.n_cells
    g = torch.Generator().manual_seed(seed)
    p = 15e6 + 4e6 * torch.rand(n, generator=g, dtype=torch.float64)
    sw = 0.3 + 0.25 * torch.rand(n, generator=g, dtype=torch.float64)
    sat = torch.rand(n, generator=g) < 0.5
    sg = 0.05 + 0.15 * torch.rand(n, generator=g, dtype=torch.float64)
    rs = s.pvt.rs_sat(p) * (0.6 + 0.3 * torch.rand(n, generator=g, dtype=torch.float64))
    return p, sw, torch.where(sat, sg, rs), sat


def test_drsdt_does_not_create_components_in_closed_cell():
    cfg = copy.deepcopy(BASE)
    cfg["reservoir"]["dimensions"] = [1, 1, 1]
    cfg["wells"] = []
    cfg["fluid"]["s_g"] = 0.15
    cfg["simulation"].update(newton_tol_cnv=1e-8, newton_tol_mb=1e-10,
                             newton_relaxed_max_pv_fraction=0.0)
    res, fluid, wm = _objects(cfg)
    solver = BlackOilFIMSolver(res, fluid, wm, cfg["simulation"])
    p = fluid.pressure.reshape(-1).double()
    sw = fluid.s_w.reshape(-1).double()
    # Свободный газ с ещё не полностью растворённым Rs допустим при DRSDT.
    rs = solver.rs_sat_eff(p) * 0.5
    fluid.rs = rs.view_as(fluid.pressure).clone()
    solver._rs_cap = rs.clone()
    solver.drsdt = float(rs[0]) * 0.01 / 86400
    initial = solver.masses(p, sw, solver.x, solver.sat, solver.ug).sum(1)
    for dt in (86400.0, 43200.0):
        assert solver.step(dt)
        mass = solver.masses(fluid.pressure.reshape(-1).double(), fluid.s_w.reshape(-1).double(),
                             solver.x, solver.sat, solver.ug).sum(1)
        assert torch.allclose(mass, initial, rtol=1e-8, atol=1e-6), (mass - initial) / initial
    assert bool((fluid.rs.reshape(-1) > rs).all())
    assert bool((fluid.rs.reshape(-1) <= rs + solver.drsdt * 129600.0 + 1e-10).all())


def test_blackoil_jacobian_matches_finite_differences():
    cfg = copy.deepcopy(BASE)
    cfg['fluid']['capillary_pressure'] = {"pc_scale": 2e4, "pc_exponent": 1.5}
    cfg['fluid']['capillary_pressure_og'] = {"pc_scale": 1e4, "pc_exponent": 1.2}
    res, fluid, wm = _objects(cfg)
    s = BlackOilFIMSolver(res, fluid, wm, cfg['simulation'])
    p, sw, x, sat = _mixed_state(s)
    assert 0 < int(sat.sum()) < s.n_cells
    acc_old = s.cell_values(p * 0.99, sw, x, sat)[I_ACC:I_ACC + 3]
    dt = 86400.0
    st = s.cell_state(p, sw, x, sat)
    s.wells.begin_step(st[0], st[1], 0.0)
    p_bh = torch.tensor([22e6, 14e6], dtype=torch.float64)
    R0, J, wsys = s.assemble_bo(p, sw, x, sat, acc_old, dt, p_bh)

    rows = torch.repeat_interleave(torch.arange(s.n_cells), s.builder.indptr[1:] - s.builder.indptr[:-1])
    cols = s.builder.indices
    worst = 0.0
    cells = list(range(0, s.n_cells, 7)) + s.wells.conn_cell.tolist()
    for c in cells:
        for var in range(3):
            base = (p, sw, x)[var][c].item()
            h = max(1.0, 1e-7 * abs(base)) if var == 0 else (1e-7 if var == 1 else (1e-7 if sat[c] else 1e-6))
            vals = []
            for sgn in (1.0, -1.0):
                st_ = [p.clone(), sw.clone(), x.clone()]
                st_[var][c] += sgn * h
                Rk, _, _ = s.assemble_bo(*st_, sat, acc_old, dt, p_bh)
                vals.append(Rk)
            fd = ((vals[0] - vals[1]) / (2 * h)).view(-1, 3)
            an = torch.zeros_like(fd)
            m = cols == c
            an[rows[m]] = J[m][:, :, var]
            err = ((fd - an).abs().max() / fd.abs().max().clamp_min(1e-300)).item()
            worst = max(worst, err)
    assert worst < 1e-5, worst

    # блоки скважин
    for w in range(2):
        h = 10.0
        pb_p, pb_m = p_bh.clone(), p_bh.clone()
        pb_p[w] += h
        pb_m[w] -= h
        Rp, _, sp = s.assemble_bo(p, sw, x, sat, acc_old, dt, pb_p)
        Rm, _, sm = s.assemble_bo(p, sw, x, sat, acc_old, dt, pb_m)
        dR = ((Rp - Rm) / (2 * h)).view(-1, 3)
        m = s.wells.conn_well == w
        assert torch.allclose(dR[s.wells.conn_cell[m]], wsys["Cv"][m], rtol=1e-6, atol=1e-30)
        dD = ((sp["r_w"] - sm["r_w"]) / (2 * h))[w]
        assert abs(dD - wsys["D"][w]) <= 1e-6 * abs(wsys["D"][w])
    for ci, c in enumerate(s.wells.conn_cell.tolist()):
        w = int(s.wells.conn_well[ci])
        for var in range(3):
            h = 10.0 if var == 0 else 1e-7
            st_p = [p.clone(), sw.clone(), x.clone()]
            st_m = [p.clone(), sw.clone(), x.clone()]
            st_p[var][c] += h
            st_m[var][c] -= h
            rp = s.assemble_bo(*st_p, sat, acc_old, dt, p_bh)[2]["r_w"]
            rm = s.assemble_bo(*st_m, sat, acc_old, dt, p_bh)[2]["r_w"]
            fd = ((rp - rm) / (2 * h))[w]
            assert abs(fd - wsys["Bv"][ci, var]) <= 1e-5 * abs(fd) + 1e-20


def _run(cfg, dt_days, days):
    cfg = copy.deepcopy(cfg)
    res, fluid, wm = _objects(cfg)
    sim = Simulator(res, fluid, wm, cfg['simulation'], device=torch.device('cpu'))
    for _ in range(int(round(days / dt_days))):
        assert sim.run_step(dt_days * 86400.0)
    return sim, fluid


def test_blackoil_reduces_to_oil_water_without_gas():
    cfg = copy.deepcopy(BASE)
    cfg['fluid'].pop('pvt_path')
    cfg['fluid'].update({"mu_oil": 1.0, "mu_water": 0.5, "c_oil": 1e-5, "c_water": 1e-5})
    cfg['reservoir']['dimensions'] = [10, 10, 1]
    cfg['wells'] = [
        {"name": "I", "type": "injector", "i": 1, "j": 1, "k": 0, "radius": 0.1,
         "control_type": "rate", "control_value": 30.0, "injected_phase": "water"},
        {"name": "P", "type": "producer", "i": 8, "j": 8, "k": 0, "radius": 0.1,
         "control_type": "rate", "control_value": 30.0},
    ]
    cfg_ow = copy.deepcopy(cfg)
    cfg_ow['simulation']['fim_model'] = 'oil_water'
    sim_bo, f_bo = _run(cfg, 1.0, 10.0)
    sim_ow, f_ow = _run(cfg_ow, 1.0, 10.0)
    assert isinstance(sim_bo.fim_solver, BlackOilFIMSolver)
    assert not isinstance(sim_ow.fim_solver, BlackOilFIMSolver)
    dp = (f_bo.pressure.double() - f_ow.pressure.double()).abs().max().item()
    dsw = (f_bo.s_w.double() - f_ow.s_w.double()).abs().max().item()
    assert dp < 2e3, dp
    assert dsw < 1e-4, dsw
    assert float(f_bo.s_g.abs().max()) < 1e-12


def _component_masses(sim):
    s = sim.fim_solver
    f = sim.fluid
    p = f.pressure.double().reshape(-1)
    sw = f.s_w.double().reshape(-1)
    return s.cell_values(p, sw, s.x, s.sat)[I_ACC:I_ACC + 3].sum(dim=1)


def test_blackoil_gas_injection_conserves_mass_and_meets_rates():
    cfg = copy.deepcopy(BASE)
    cfg['simulation'].update({"initial_time_step_days": 0.5, "max_time_step_days": 0.5})
    res, fluid, wm = _objects(cfg)
    sim = Simulator(res, fluid, wm, cfg['simulation'], device=torch.device('cpu'))
    rho = sim.fim_solver.rho_sc
    m0 = _component_masses(sim)
    out = torch.zeros(3, dtype=torch.float64)
    for _ in range(12):
        assert sim.run_step(0.5 * 86400.0)
        for w in sim.fim_solver.wells.wells:
            r = w.last_surface_rates
            out += torch.tensor([r['oil'], r['water'], r['gas']], dtype=torch.float64) * rho * 0.5
    m1 = _component_masses(sim)
    err = ((m1 - m0) + out).abs() / m0
    assert (err < 1e-6).all(), err
    gi, prod = sim.fim_solver.wells.wells
    assert abs(gi.last_surface_rates['gas'] + 2000.0) < 2000.0 * 1e-4
    assert abs(prod.last_surface_rates['oil'] - 20.0) < 20.0 * 1e-4
    assert float(fluid.s_g.max()) > 0.01
    sg, sw, so = fluid.s_g.double(), fluid.s_w.double(), fluid.s_o.double()
    assert float(sg.min()) >= 0 and float(so.min()) >= -1e-12
    assert torch.allclose(sg + sw + so, torch.ones_like(sg))


def test_depletion_below_bubble_point_liberates_gas():
    cfg = copy.deepcopy(BASE)
    cfg['fluid']['pressure'] = 21.0
    cfg['wells'] = [
        {"name": "P", "type": "producer", "i": 2, "j": 2, "k": 1, "radius": 0.1,
         "control_type": "bhp", "control_value": 12.0},
    ]
    res, fluid, wm = _objects(cfg)
    sim = Simulator(res, fluid, wm, cfg['simulation'], device=torch.device('cpu'))
    s = sim.fim_solver
    assert int(s.sat.sum()) == 0
    for _ in range(10):
        assert sim.run_step(2.0 * 86400.0)
    assert int(s.sat.sum()) > 0
    assert float(fluid.s_g.max()) > 1e-3
    p = fluid.pressure.double().reshape(-1)
    rs = fluid.rs.reshape(-1)
    assert bool((rs <= s.pvt.rs_sat(p) * (1 + 1e-9)).all())
    sat = s.sat
    assert torch.allclose(rs[sat], s.pvt.rs_sat(p)[sat])
    prod = s.wells.wells[0]
    gor = prod.last_surface_rates['gas'] / prod.last_surface_rates['oil']
    assert gor > s.pvt.rs_sat(p).min().item() * 0.5
