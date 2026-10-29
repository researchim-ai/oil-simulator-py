import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.manual_assembly_solver import ManualFIMSolver


def _build(dims=(5, 4, 3), tol=1e-8):
    cfg = {
        "simulation": {"newton_max_iter": 15, "newton_tol_mass": tol},
        "reservoir": {"dimensions": list(dims), "grid_size": [10.0, 10.0, 2.0],
                      "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 0.3},
        "fluid": {"pressure": 20.0, "s_w": 0.2, "mu_oil": 1.0, "mu_water": 0.5,
                  "rho_oil": 850.0, "rho_water": 1000.0, "c_oil": 1e-5, "c_water": 1e-5,
                  "s_wr": 0.2, "s_or": 0.2, "pc_scale": 0.0},
        "wells": [
            {"name": "I", "type": "injector", "i": 0, "j": 1, "k": 2, "radius": 0.1,
             "control_type": "rate", "control_value": 20.0, "injected_phase": "water"},
            {"name": "P", "type": "producer", "i": 4, "j": 3, "k": 0, "radius": 0.1,
             "control_type": "rate", "control_value": 20.0},
        ],
    }
    dev = torch.device('cpu')
    res = Reservoir(cfg['reservoir'], device=dev)
    fluid = Fluid(cfg['fluid'], res, device=dev)
    wells = WellManager(cfg['wells'], res)
    return ManualFIMSolver(res, fluid, wells, cfg['simulation']), fluid


def test_jacobian_matches_finite_differences():
    solver, fluid = _build()
    n = solver.n_cells
    dt = 86400.0
    t = 30 * 86400.0

    p0 = fluid.pressure.double().clone()
    sw0 = fluid.s_w.double().clone()
    m_o, m_w = solver.calc_masses(p0, sw0)
    gen = torch.Generator().manual_seed(0)
    p = p0 + 2e5 * torch.rand(p0.shape, generator=gen, dtype=torch.float64)
    sw = (sw0 + 0.3 * torch.rand(sw0.shape, generator=gen, dtype=torch.float64)).clamp(0, 1)

    _, J = solver.assemble_system(p, sw, m_o, m_w, dt, current_time=t)
    J_dense = sp.bsr_matrix(
        (J.numpy(), solver.builder.indices.numpy(), solver.builder.indptr.numpy()),
        shape=(2 * n, 2 * n),
    ).toarray()

    for cell in range(n):
        for var, h in ((0, 10.0), (1, 1e-6)):
            fields = []
            for sign in (1.0, -1.0):
                pp, ss = p.clone().view(-1), sw.clone().view(-1)
                (pp if var == 0 else ss)[cell] += sign * h
                R, _ = solver.assemble_system(pp.view_as(p), ss.view_as(sw), m_o, m_w, dt, current_time=t)
                fields.append(R.numpy())
            fd = (fields[0] - fields[1]) / (2 * h)
            col = J_dense[:, 2 * cell + var]
            assert np.abs(fd - col).max() <= 1e-6 * np.abs(fd).max(), (cell, var)


def test_newton_converges_quadratically(monkeypatch):
    def forbidden_cuda_call():
        raise AssertionError("CPU solve must not query a CUDA stream")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", forbidden_cuda_call)
    solver, fluid = _build(tol=1e-8)
    p_init = fluid.pressure.clone()
    for step in range(3):
        assert solver.step(86400.0, current_time=step * 86400.0)
    assert not torch.allclose(fluid.pressure, p_init)
