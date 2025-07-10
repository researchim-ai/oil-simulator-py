import torch
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from simulator.fluid import Fluid
from types import SimpleNamespace

def test_capillary_hysteresis_land():
    # Mock reservoir with dimensions attribute
    reservoir = SimpleNamespace(dimensions=(3,3,1))
    device = torch.device('cpu')

    cfg = {
        'pressure': 20.0,
        's_w': 0.2,
        'mu_oil': 1.0,
        'mu_water': 0.5,
        'capillary_pressure': { 'pc_scale': 0.1e6, 'pc_exponent': 1.5 }
    }

    fluid = Fluid(cfg, reservoir, device)
    pc_initial = fluid.get_capillary_pressure(fluid.s_w)

    # Increase Sw (imbibition) to 0.6 and update hysteresis
    fluid.s_w += 0.4
    fluid.update_hysteresis()

    # Decrease Sw back to 0.3 (imbibition branch active)
    fluid.s_w -= 0.3
    pc_after = fluid.get_capillary_pressure(fluid.s_w)

    # Pc with hysteresis should be lower (imbibition curve below drainage)
    assert torch.all(pc_after < pc_initial + 1e-6) 