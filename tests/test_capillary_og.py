import torch, sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from simulator.fluid import Fluid
from types import SimpleNamespace


def _setup_fluid(pc_scale=0.2e6, pc_exponent=1.3):
    reservoir = SimpleNamespace(dimensions=(1, 1, 1))
    cfg = {
        'pressure': 20.0,
        's_w': 0.2,
        's_g': 0.1,
        'mu_oil': 1.0,
        'mu_water': 0.5,
        'mu_gas': 0.05,
        'capillary_pressure': {
            'pc_og_scale': pc_scale,
            'pc_og_exponent': pc_exponent
        }
    }
    return Fluid(cfg, reservoir, torch.device('cpu'))


def test_pc_og_monotonic():
    fluid = _setup_fluid()
    s_g_vals = torch.linspace(0.0, 0.9, 10)
    pcs = fluid.get_capillary_pressure_og(s_g_vals)
    # Проверка монотонного убывания
    assert torch.all(pcs[:-1] >= pcs[1:] - 1e-6)


def test_pc_og_derivative_sign():
    fluid = _setup_fluid()
    s_g = torch.tensor(0.4)
    dpc = fluid.get_capillary_pressure_og_derivative(s_g)
    assert dpc <= 0.0


def test_pc_og_land_hysteresis():
    fluid = _setup_fluid()
    pc_initial = fluid.get_capillary_pressure_og(fluid.s_g)

    # Увеличиваем Sg (drainage) и обновляем hysteresis
    fluid.s_g += 0.3
    fluid.update_hysteresis()

    # Снижаем Sg – теперь кривые imbibition
    fluid.s_g -= 0.2
    pc_after = fluid.get_capillary_pressure_og(fluid.s_g)

    # Имбибиционная кривая должна быть ниже соответствующей дренажной
    s_norm = fluid.s_g / (1.0 - fluid.sw_cr - fluid.so_r)
    pc_drain = fluid.pc_og_scale * (1.0 - s_norm + 1e-6) ** (-fluid.pc_og_exponent)
    assert pc_after < pc_drain + 1e-6 