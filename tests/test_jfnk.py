import torch
import math

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager  # wells not used but required by Simulator
from simulator.simulation import Simulator
from solver.jfnk import FullyImplicitSolver

# -----------------------------------------------------------------------------
# Helpers to build minimal configs
# -----------------------------------------------------------------------------

def make_reservoir(nx, ny, nz=1):
    return Reservoir({
        "dimensions": [nx, ny, nz],
        "grid_size": [10.0, 10.0, 10.0],
        "porosity": 0.2,
        "permeability": 100.0,
        "k_vertical_fraction": 0.1,
    }, device=torch.device("cpu"))


def make_fluid(reservoir):
    return Fluid({}, reservoir, device=torch.device("cpu"))


# -----------------------------------------------------------------------------
# 1. Directional derivative check
# -----------------------------------------------------------------------------

def test_jfnk_directional_derivative():
    torch.manual_seed(0)
    res = make_reservoir(5, 5)
    fluid = make_fluid(res)
    wells = WellManager([], res)  # empty well list
    sim = Simulator(res, fluid, wells, {"jacobian": "jfnk"})
    solver = FullyImplicitSolver(sim)

    N = res.nx * res.ny * res.nz
    x0 = torch.cat([fluid.pressure.view(-1), fluid.s_w.view(-1)])

    v = torch.randn_like(x0)
    dt = 1000.0  # s

    # Jv via solver
    Jv = solver._Jv(x0, v, dt)

    # Finite-difference approximation with small eps
    eps = 1e-6
    num = (sim._fi_residual_vec(x0 + eps * v, dt) - sim._fi_residual_vec(x0, dt)) / eps

    rel_err = (Jv - num).norm() / (Jv.norm() + 1e-12)
    assert rel_err < 1e-2, f"Directional derivative mismatch: {rel_err:.2e}"


# -----------------------------------------------------------------------------
# 2. Convergence on a trivial 1×1 case
# -----------------------------------------------------------------------------

def test_jfnk_convergence_single_cell():
    res = make_reservoir(1, 1)
    fluid = make_fluid(res)
    wells = WellManager([], res)
    sim = Simulator(res, fluid, wells, {"jacobian": "jfnk"})

    converged = sim.run_step(dt=1000.0)
    assert converged, "JFNK failed to converge on 1×1 reservoir"


# -----------------------------------------------------------------------------
# 3. Consistency with explicit Jacobian (debug small grid)
# -----------------------------------------------------------------------------


def finite_diff_jacobian(sim, x, dt, eps=1e-6):
    """Build full Jacobian via finite differences (very small problems)."""
    n = x.numel()
    F0 = sim._fi_residual_vec(x, dt)
    J_cols = []
    for i in range(n):
        ei = torch.zeros_like(x)
        ei[i] = eps
        col = (sim._fi_residual_vec(x + ei, dt) - F0) / eps
        J_cols.append(col.view(-1, 1))
    return torch.cat(J_cols, dim=1)  # shape (m, n)


def test_jfnk_vs_explicit_jacobian():
    res = make_reservoir(4, 4)
    fluid = make_fluid(res)
    wells = WellManager([], res)
    sim = Simulator(res, fluid, wells, {"jacobian": "jfnk"})
    solver = FullyImplicitSolver(sim)

    dt = 1000.0
    x = torch.cat([fluid.pressure.view(-1), fluid.s_w.view(-1)])

    J_exp = finite_diff_jacobian(sim, x, dt)

    # Test 10 random vectors
    torch.manual_seed(1)
    for _ in range(10):
        v = torch.randn_like(x)
        Jv_exp = J_exp @ v
        Jv_free = solver._Jv(x, v, dt)
        rel_err = (Jv_exp - Jv_free).norm() / (Jv_exp.norm() + 1e-12)
        assert rel_err < 1e-2, f"Jv mismatch vs explicit Jacobian: {rel_err:.2e}" 