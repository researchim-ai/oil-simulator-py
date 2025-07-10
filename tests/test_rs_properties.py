import os, sys, torch
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from simulator.fluid import Fluid


def _fluid():
    cfg = {
        'pressure': 20.0,
        's_w': 0.2,
        's_g': 0.0,
        'mu_oil': 1.0,
        'mu_water': 0.5,
        'mu_gas': 0.05,
        'pbubble': 20.0,
        'rs_bubble': 120.0
    }
    res = SimpleNamespace(dimensions=(1, 1, 1))
    return Fluid(cfg, res, torch.device('cpu'))


def test_rs_linear_model():
    fluid = _fluid()
    pb = fluid.pbubble
    rs_b = fluid.rs_bubble
    p = torch.tensor([0.0, pb/2, pb])
    rs = fluid.calc_rs(p)
    assert torch.allclose(rs, torch.tensor([0.0, rs_b/2, rs_b]))
    # above bubble unchanged
    p2 = torch.tensor([pb*1.5])
    rs2 = fluid.calc_rs(p2)
    assert rs2 == rs_b


def test_drs_dp_consistency():
    fluid = _fluid()
    p = torch.linspace(1e6, fluid.pbubble*0.99, 5)
    drs_dp_analytic = fluid.calc_drs_dp(p)
    eps = 1000.0
    drs_dp_fd = (fluid.calc_rs(p+eps) - fluid.calc_rs(p-eps)) / (2*eps)
    assert torch.allclose(drs_dp_analytic, drs_dp_fd, rtol=1e-3)


def test_total_gas_mass():
    fluid = _fluid()
    por = torch.tensor(0.2)
    p = torch.tensor(15e6)  # below bubble
    s_o = torch.tensor(0.7)
    s_g = torch.tensor(0.1)
    m = fluid.total_gas_mass(s_o, s_g, p, por)
    # Should be positive and grow with s_g
    p_high = torch.tensor(18e6)
    m2 = fluid.total_gas_mass(s_o, s_g+0.05, p_high, por)
    assert m2 > m 