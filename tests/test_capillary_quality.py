import os, sys, torch

# Ensure src is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from simulator.fluid import Fluid
from types import SimpleNamespace


def _setup_fluid(pc_scale=0.1e6, pc_exponent=1.5):
    reservoir = SimpleNamespace(dimensions=(1, 1, 1))
    cfg = {
        'pressure': 20.0,
        's_w': 0.3,  # initial value will be overwritten in test loop
        'mu_oil': 1.0,
        'mu_water': 0.5,
        'capillary_pressure': { 'pc_scale': pc_scale, 'pc_exponent': pc_exponent }
    }
    return Fluid(cfg, reservoir, torch.device('cpu'))


def finite_difference(fluid, sw_tensor, eps=1e-5):
    fluid.s_w = sw_tensor + eps
    pc_plus = fluid.get_capillary_pressure(fluid.s_w).clone()
    fluid.s_w = sw_tensor - eps
    pc_minus = fluid.get_capillary_pressure(fluid.s_w).clone()
    return (pc_plus - pc_minus) / (2 * eps)


def test_capillary_land_analytic():
    """Checks Land formula value and derivative for random Sw points."""
    torch.manual_seed(0)
    fluid = _setup_fluid()
    exp = fluid.pc_exponent
    scale = fluid.pc_scale

    for _ in range(10):
        # Sample Sw in physical range and set sw_max higher (imbibition)
        Sw = torch.rand(1).item()*0.5 + 0.25  # 0.25-0.75
        sw_max = min(Sw + 0.1, 0.9)

        fluid.s_w = torch.tensor(Sw)
        fluid.sw_max = torch.tensor(sw_max)

        # Direct model output
        pc_model = fluid.get_capillary_pressure(fluid.s_w)

        # Drainage curve
        s_norm = fluid._get_normalized_saturation(fluid.s_w)
        pc_drain = scale * (1.0 - s_norm + 1e-6) ** (-exp)
        pc_expected = pc_drain * (1 - sw_max) / (1 - Sw)

        # Relative error should be <5e-6 (accounts for epsilon in formula)
        rel_err = torch.abs((pc_model - pc_expected) / pc_expected)
        assert rel_err < 5e-6, f"Value mismatch: rel_err={rel_err}"

        # Derivative sign check
        dpc_analytic = fluid.get_capillary_pressure_derivative(fluid.s_w)
        assert dpc_analytic <= 0.0, "dPc/dSw should be ≤ 0" 