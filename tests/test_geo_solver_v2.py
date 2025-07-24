# tests/test_geo_solver_v2.py
import os, sys, torch
sys.path.append('src')

from simulator.reservoir import Reservoir
from solver.geo_solver_v2 import GeoSolverV2

def test_geo_solver_residual_drop():
    os.environ['OIL_DEBUG'] = '0'
    torch.manual_seed(0)

    res = Reservoir({
        'dimensions': [16, 16, 16],
        'grid_size':  [10, 10, 5],
        'permeability': 100,
        'k_vertical_fraction': 0.1,
        'porosity': 0.2,
        'c_rock': 1e-5
    })
    solver = GeoSolverV2(res, cycles_per_call=1, pre_smooth=3, post_smooth=3, debug=False)

    assert hasattr(solver, 'Dinv') and hasattr(solver, 'D')

    n = res.nx * res.ny * res.nz
    rhs_phys = torch.randn(n, dtype=torch.float64, device=solver.device)

    rhs_hat = solver.Dinv * rhs_phys

    x_hat = torch.zeros_like(rhs_hat)
    for _ in range(5):
        x_hat = solver._v_cycle(0, x_hat, rhs_hat)

    r_hat = rhs_hat - solver._apply_A(0, x_hat)
    factor_hat = (r_hat.norm() / rhs_hat.norm()).item()

    # r_phys = rhs_phys - A_phys * delta_phys, где A_phys = D * A_hat * D, delta_phys = D * x_hat
    delta_phys   = solver.Dinv * x_hat
    A_phys_delta = solver.D * solver._apply_A(0, x_hat)   # D * (A_hat x_hat)
    r_phys       = rhs_phys - A_phys_delta
    factor_phys  = (r_phys.norm() / rhs_phys.norm()).item()
    torch.testing.assert_close(solver.D * r_hat, r_phys, rtol=1e-6, atol=1e-9)

    # Проверки
    assert factor_hat < 0.08, f"Too weak reduction in hat space: {factor_hat:.3e}"
    # Резидуалы должны совпасть после обратного скейла
    torch.testing.assert_close(solver.D * r_hat, r_phys, rtol=1e-6, atol=1e-9)
    # А факторы могут отличаться чуть-чуть (норма изменилась)
    assert abs(factor_phys - factor_hat) < 5e-3, "Residual drop mismatch due to scaling"



def test_geo_solver_solve_api():
    """Проверяем, что public solve() даёт малый резидуал в физическом пространстве."""
    os.environ['OIL_DEBUG'] = '0'
    torch.manual_seed(1)

    res = Reservoir({
        'dimensions': [16, 16, 16],
        'grid_size':  [10, 10, 5],
        'permeability': 100,
        'k_vertical_fraction': 0.1,
        'porosity': 0.2,
        'c_rock': 1e-5
    })
    solver = GeoSolverV2(res, cycles_per_call=1, pre_smooth=3, post_smooth=3, debug=False)

    n = res.nx * res.ny * res.nz
    rhs_phys = torch.randn(n, dtype=torch.float64, device=solver.device)

    # Решение
    delta_phys = torch.from_numpy(solver.solve(rhs_phys)).to(device=solver.device, dtype=torch.float64)

    # r_hat = rhs_hat - A_hat * delta_hat
    rhs_hat   = solver.Dinv * rhs_phys
    delta_hat = solver.Dinv * delta_phys
    r_hat     = rhs_hat - solver._apply_A(0, delta_hat)

    # назад в физическое
    r_phys = solver.D * r_hat
    factor_phys = (r_phys.norm() / rhs_phys.norm()).item()

    assert factor_phys < 1e-6, f"solve() residual too large: {factor_phys:.3e}"
