import torch
from solver.geom_amg import mg_solve, _apply_poisson

def test_geom_multigrid_poisson():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nx, ny, nz = 32, 32, 32
    x_true = torch.randn(nz, ny, nx, device=device)
    b = _apply_poisson(x_true, 1.0, 1.0, 1.0)

    x_sol, res = mg_solve(b, cycles=5, device=device)

    # Проверяем, что невязка упала на 4 порядка и совпало решение в среднем
    with torch.no_grad():
        final_res = torch.norm(_apply_poisson(x_sol, 1.0, 1.0, 1.0) - b) / torch.norm(b)
        err = torch.norm(x_sol - x_true) / torch.norm(x_true)
        assert final_res.item() < 1e-4, f"residual too high {final_res}"
        assert err.item() < 1e-2, f"solution error {err}" 