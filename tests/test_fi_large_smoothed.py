import json, pytest, os, torch, time

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.simulation import Simulator

CONFIG = os.path.join('configs', 'large_3d.json')
DT_SEC = 86400.0  # один шаг – сутки

@pytest.mark.parametrize("smoother", ["jacobi", "l1gs", "chebyshev"])
def test_fully_implicit_large_3d_converges(smoother):
    """Проверяем, что 1 шаг FI сходится на крупной сетке при разных AMG-сглаживателях."""
    with open(CONFIG) as f:
        cfg = json.load(f)

    res = Reservoir.from_config(cfg)
    fluid = Fluid.from_config(cfg)

    sim_params = {
        "solver_type": "fully_implicit",
        "jacobian": "jfnk",
        "backend": "geo",           # используем Geo-AMG
        "smoother": smoother,
        "newton_max_iter": 15,
        "verbose": False,
    }

    sim = Simulator(res, fluid, well_manager=None, sim_params=sim_params)

    t0 = time.time()
    ok = sim.run_step(DT_SEC)
    elapsed = time.time() - t0

    assert ok, f"Newton failed to converge for smoother={smoother}"

    # Диагностическая информация (печатается только при падении теста)
    niters = getattr(sim.fi_solver, "last_newton_iters", 99)
    giters = getattr(sim.fi_solver, "last_gmres_iters", 9999)

    assert niters <= 15, f"Too many Newton iters ({niters}) for smoother={smoother}"
    assert giters <= 1500, f"Too many GMRES iters ({giters}) for smoother={smoother}"
    assert elapsed < 60, f"Step took too long ({elapsed:.1f}s) for smoother={smoother}" 