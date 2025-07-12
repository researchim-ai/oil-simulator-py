import torch
import numpy as np
from simulator.fluid import Fluid
from simulator.reservoir import Reservoir


def test_pvt_derivatives_linear():
    # Линейная таблица: Bo = 1.2 - 0.01*(P-10 МПа)
    p_grid = np.array([10.0, 20.0])  # МПа
    bo_tab = np.array([1.20, 1.10])
    bg_tab = np.array([0.90, 1.00])
    bw_tab = np.array([1.05, 1.00])
    rs_tab = np.array([100.0, 200.0])

    cfg = {
        'pvt': {
            'pressure': p_grid.tolist(),
            'bo': bo_tab.tolist(),
            'bg': bg_tab.tolist(),
            'bw': bw_tab.tolist(),
            'rs': rs_tab.tolist(),
            'rho_oil': [800, 800],
            'rho_water': [1000, 1000],
            'rho_gas': [150, 150],
            'mu_oil': [1, 1],
            'mu_water': [0.5, 0.5],
            'mu_gas': [0.05, 0.05],
            'rv': [0, 0]
        }
    }

    res_cfg = {
        'dimensions': [1, 1, 1],
        'grid_size': [10.0, 10.0, 10.0]
    }
    res = Reservoir(res_cfg)
    fluid = Fluid(cfg, res)

    P_test = torch.tensor([[15.0]]) * 1e6  # 15 МПа → Па

    # Аналитические производные (линейная интерполяция => piecewise const slope)
    dBo_exp = (bo_tab[1] - bo_tab[0]) / ( (p_grid[1] - p_grid[0]) * 1e6 )
    dBg_exp = (bg_tab[1] - bg_tab[0]) / ( (p_grid[1] - p_grid[0]) * 1e6 )
    dBw_exp = (bw_tab[1] - bw_tab[0]) / ( (p_grid[1] - p_grid[0]) * 1e6 )
    dRs_exp = (rs_tab[1] - rs_tab[0]) / ( (p_grid[1] - p_grid[0]) * 1e6 )

    dtype = fluid.calc_dbo_dp(P_test).dtype
    assert torch.allclose(fluid.calc_dbo_dp(P_test), torch.tensor([[dBo_exp]], dtype=dtype), rtol=0, atol=1e-12)
    assert torch.allclose(fluid.calc_dbg_dp(P_test), torch.tensor([[dBg_exp]], dtype=dtype), rtol=0, atol=1e-12)
    assert torch.allclose(fluid.calc_dbw_dp(P_test), torch.tensor([[dBw_exp]], dtype=dtype), rtol=0, atol=1e-12)
    assert torch.allclose(fluid.calc_drs_dp(P_test), torch.tensor([[dRs_exp]], dtype=dtype), rtol=0, atol=1e-12) 