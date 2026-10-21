import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


def _simulator(tmp_path, level, newton_max_iter):
    cfg = {
        "simulation": {"solver_type": "fully_implicit", "time_step_days": 1.0, "newton_max_iter": newton_max_iter,
                       "newton_tol_mass": 1e-8, "max_time_step_attempts": 2, "dt_reduction_factor": 2.0,
                       "debug": {"enabled": True, "level": level, "output_dir": str(tmp_path),
                                 "run_id": "run", "console": False}},
        "reservoir": {"dimensions": [6, 5, 2], "grid_size": [10.0, 10.0, 2.0],
                      "porosity": 0.2, "permeability": 100.0, "k_vertical_fraction": 0.3},
        "fluid": {"pressure": 20.0, "s_w": 0.2, "mu_oil": 1.0, "mu_water": 0.5,
                  "rho_oil": 850.0, "rho_water": 1000.0, "c_oil": 1e-5, "c_water": 1e-5,
                  "s_wr": 0.2, "s_or": 0.2, "pc_scale": 0.0},
        "wells": [
            {"name": "I", "type": "injector", "i": 0, "j": 0, "k": 1, "radius": 0.1,
             "control_type": "rate", "control_value": 20.0, "injected_phase": "water"},
            {"name": "P", "type": "producer", "i": 5, "j": 4, "k": 0, "radius": 0.1,
             "control_type": "rate", "control_value": 20.0},
        ],
    }
    dev = torch.device('cpu')
    res = Reservoir(cfg['reservoir'], device=dev)
    fluid = Fluid(cfg['fluid'], res, device=dev)
    wells = WellManager(cfg['wells'], res)
    return Simulator(res, fluid, wells, cfg['simulation'], device=dev)


def _events(tmp_path):
    with open(tmp_path / "run" / "events.jsonl") as f:
        return [json.loads(line) for line in f]


def test_debug_log_collects_all_stages(tmp_path):
    sim = _simulator(tmp_path, level=3, newton_max_iter=15)
    assert sim.run_step(86400.0)
    sim.fim_solver.debug.close()

    ev = _events(tmp_path)
    iters = [e for e in ev if e["event"] == "newton_iter" and "linear" in e]
    assert iters
    it = iters[0]
    for key in ("res_inf", "mb_err_rel", "worst", "jac", "linear", "update", "line_search", "wells", "fd_check", "time"):
        assert key in it, key
    assert it["linear"]["iters"] > 0 and it["linear"]["history"]
    assert it["fd_check"]["worst_rel_err"] < 1e-5
    assert {w["name"] for w in it["wells"]} == {"I", "P"}
    assert [e for e in ev if e["event"] == "step_end"][-1]["converged"]
    assert (tmp_path / "run" / "summary.json").exists()


def test_failure_dumps_system_and_logs_dt_chop(tmp_path):
    sim = _simulator(tmp_path, level=1, newton_max_iter=1)
    assert not sim.run_step(86400.0)
    sim.fim_solver.debug.close()

    ev = _events(tmp_path)
    ends = [e for e in ev if e["event"] == "step_end"]
    assert ends and all(not e["converged"] and e["reason"] == "max_newton_iter" for e in ends)
    assert any(e["event"] == "dt_chop" for e in ev)
    dumps = [e["path"] for e in ev if e["event"] == "dump"]
    assert dumps
    d = np.load(dumps[0])
    n = 6 * 5 * 2
    assert d["R"].shape == (2 * n,) and d["J_vals"].shape[1:] == (2, 2) and d["indptr"].shape == (n + 1,)
