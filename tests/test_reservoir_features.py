import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.simulation import Simulator


class EmptyWellManager:
    def __init__(self):
        self.wells = []
        self.device = torch.device("cpu")

    def get_wells(self):
        return self.wells


def test_reservoir_stochastic_porosity_reproducible():
    config = {
        "dimensions": [6, 6, 4],
        "permeability": 150.0,
        "stochastic": {
            "seed": 123,
            "porosity": {
                "mean": 0.22,
                "std": 0.03,
                "corr_length": 1.0,
            }
        }
    }
    res1 = Reservoir(config, device=torch.device("cpu"))
    res2 = Reservoir(config, device=torch.device("cpu"))
    assert torch.allclose(res1.porosity, res2.porosity)
    assert res1.porosity.std() > 0.0


def test_reservoir_variable_spacing():
    config = {
        "dimensions": [2, 2, 2],
        "permeability": 100.0,
        "grid": {
            "dx": [5.0, 10.0],
            "dy": [8.0, 12.0],
            "dz": [3.0, 4.0],
        }
    }
    reservoir = Reservoir(config, device=torch.device("cpu"))
    assert torch.allclose(reservoir.dx_vector.cpu(), torch.tensor([5.0, 10.0], dtype=torch.float64))
    assert torch.allclose(reservoir.dy_vector.cpu(), torch.tensor([8.0, 12.0], dtype=torch.float64))
    assert torch.allclose(reservoir.dz_vector.cpu(), torch.tensor([3.0, 4.0], dtype=torch.float64))
    assert torch.allclose(reservoir.dx_face.cpu(), torch.tensor([7.5], dtype=torch.float64))
    assert reservoir.cell_volume.shape == (2, 2, 2)
    volumes = reservoir.cell_volume.cpu().numpy()
    expected = np.array([[[5.0*8.0*3.0, 5.0*8.0*4.0],
                          [5.0*12.0*3.0, 5.0*12.0*4.0]],
                         [[10.0*8.0*3.0, 10.0*8.0*4.0],
                          [10.0*12.0*3.0, 10.0*12.0*4.0]]])
    assert np.allclose(volumes, expected)


def test_capillary_flow_influences_saturation():
    reservoir_cfg = {
        "dimensions": [2, 1, 1],
        "grid_size": [10.0, 10.0, 10.0],
        "permeability": 100.0,
    }
    fluid_cfg_pc = {
        "pressure": 10.0,
        "s_w": 0.2,
        "s_g": 0.0,
        "capillary_pressure": {"pc_scale": 5e4, "pc_exponent": 1.5},
    }
    fluid_cfg_nopc = {
        "pressure": 10.0,
        "s_w": 0.2,
        "s_g": 0.0,
        "capillary_pressure": {"pc_scale": 0.0},
    }
    res = Reservoir(reservoir_cfg, device=torch.device("cpu"))
    well_mgr = EmptyWellManager()
    sim_params = {
        "solver_type": "impes",
        "time_step_days": 0.0001,
        "use_cuda": False,
        "use_capillary_potentials": True,
    }

    fluid_pc = Fluid(fluid_cfg_pc, res, device=torch.device("cpu"))
    fluid_pc.s_w[0, 0, 0] = 0.2
    fluid_pc.s_w[1, 0, 0] = 0.8
    fluid_pc.s_o = 1.0 - fluid_pc.s_w

    sim_pc = Simulator(res, fluid_pc, well_mgr, sim_params, device=torch.device("cpu"))
    P_new = fluid_pc.pressure.clone()
    sw_before = fluid_pc.s_w.clone()
    sim_pc._impes_saturation_step(P_new, dt=3600.0)
    sw_after_pc = fluid_pc.s_w.clone()

    res2 = Reservoir(reservoir_cfg, device=torch.device("cpu"))
    fluid_nopc = Fluid(fluid_cfg_nopc, res2, device=torch.device("cpu"))
    fluid_nopc.s_w[0, 0, 0] = 0.2
    fluid_nopc.s_w[1, 0, 0] = 0.8
    fluid_nopc.s_o = 1.0 - fluid_nopc.s_w

    sim_nopc = Simulator(res2, fluid_nopc, well_mgr, sim_params, device=torch.device("cpu"))
    sim_nopc._impes_saturation_step(P_new, dt=3600.0)
    sw_after_nopc = fluid_nopc.s_w.clone()

    delta_pc = (sw_after_pc - sw_before).abs().max()
    delta_nopc = (sw_after_nopc - sw_before).abs().max()
    assert delta_pc > delta_nopc + 1e-8

