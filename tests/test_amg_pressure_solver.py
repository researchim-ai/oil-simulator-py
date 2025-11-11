import os
import sys
import json
import torch
import numpy as np

# Добавляем src в путь для импорта компонентов
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator
from solver.classical_amg import ClassicalAMG, _apply_reference_fix


def _run_one_step_with_solver(config_path: str, solver_name: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Параметры
    sim_cfg = config.get('simulation', {}) or {}
    sim_cfg['solver_type'] = 'impes'
    sim_cfg['total_time_days'] = 1.0
    sim_cfg['time_step_days'] = 1.0
    # Включаем решатель давления
    sim_cfg['pressure_solver'] = solver_name
    if solver_name in {'amg', 'gmres_amg', 'cg_amg'}:
        sim_cfg['amg'] = {
            'tol': 1e-6,
            'max_cycles': 20,
            'theta': 0.25,
            'max_levels': 10,
            'coarsest_size': 200,
            'device': 'auto'
        }
    # CPU для устойчивых тестов
    device = torch.device('cpu')
    reservoir = Reservoir(config=config['reservoir'], device=device)
    well_manager = WellManager(config['wells'], reservoir)
    fluid = Fluid(config=config['fluid'], reservoir=reservoir, device=device)
    sim = Simulator(reservoir, fluid, well_manager, sim_cfg, device=device)
    # Один шаг
    dt = sim_cfg['time_step_days'] * 86400.0
    ok = sim.run_step(dt=dt)
    assert ok, f"IMPES шаг не сошёлся для '{solver_name}'"
    # Возвращаем плоское давление для сравнения
    return fluid.pressure.clone().detach().cpu().numpy().reshape(-1)


def test_amg_matches_cg_pressure_field():
    cfg = "configs/gas_injection_3phase.json"
    p_cg = _run_one_step_with_solver(cfg, 'cg')
    p_amg = _run_one_step_with_solver(cfg, 'gmres_amg')
    # Сравнение полей давления
    denom = max(np.linalg.norm(p_cg), 1e-12)
    rel = np.linalg.norm(p_amg - p_cg) / denom
    # AMG и CG должны совпасть до разумной относительной точности
    assert rel < 5e-4, f"AMG vs CG расхождение слишком большое: rel={rel:.3e}"


def _build_2d_laplacian(n: int) -> torch.Tensor:
    rows = []
    cols = []
    vals = []

    def idx(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            p = idx(i, j)
            diag = 0.0
            if i > 0:
                rows.append(p); cols.append(idx(i - 1, j)); vals.append(-1.0); diag += 1.0
            if i < n - 1:
                rows.append(p); cols.append(idx(i + 1, j)); vals.append(-1.0); diag += 1.0
            if j > 0:
                rows.append(p); cols.append(idx(i, j - 1)); vals.append(-1.0); diag += 1.0
            if j < n - 1:
                rows.append(p); cols.append(idx(i, j + 1)); vals.append(-1.0); diag += 1.0
            rows.append(p); cols.append(p); vals.append(diag)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float64)
    size = (n * n, n * n)
    return torch.sparse_coo_tensor(indices, values, size=size).coalesce().to_sparse_csr()


def test_classical_amg_quality_on_laplacian():
    A = _build_2d_laplacian(12)
    A_fixed = _apply_reference_fix(A, anchor_idx=0)
    amg = ClassicalAMG(A_fixed, theta=0.25, max_levels=10, coarsest_size=40, anchor_idx=None)

    vectors = []
    const = torch.ones(A_fixed.size(0), dtype=torch.float64)
    vectors.append(const / (const.norm() + 1e-30))
    grad = torch.linspace(0, 1, A_fixed.size(0), dtype=torch.float64)
    vectors.append(grad / (grad.norm() + 1e-30))
    rand = torch.randn(A_fixed.size(0), dtype=torch.float64)
    vectors.append(rand / (rand.norm() + 1e-30))

    for v in vectors:
        z = amg.apply(v.to(amg.device), cycles=3)
        Az = torch.sparse.mm(A_fixed.to(amg.device), z.unsqueeze(1)).squeeze(1)
        rel = (Az - v.to(amg.device)).norm() / (v.norm() + 1e-30)
        assert rel < 5e-2, f"AMG self-check не проходит (rel={rel:.3e})"


