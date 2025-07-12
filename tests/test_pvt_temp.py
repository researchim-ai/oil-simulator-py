import torch, math, sys, os
sys.path.append('src')
from simulator.fluid import Fluid

class DummyReservoir:
    def __init__(self):
        self.dimensions = (1, 1, 1)
        self.grid_size = (1.0, 1.0, 1.0)
        self.pressure_ref = 1e5
        # проницаемости нужны только для Geo-AMG, здесь заглушки
        self.permeability_x = torch.ones(1,1,1)
        self.permeability_y = torch.ones(1,1,1)
        self.permeability_z = torch.ones(1,1,1)

def build_pvt():
    # Pressure grid  (MPa)
    p = [10.0, 30.0]
    # Temperature grid (°C)
    t = [40.0, 80.0]
    # Bo table shape (nT, nP)
    #   T=40°C: Bo = 1.20 @10 MPa -> 1.00 @30 MPa (linear w.r.t P)
    #   T=80°C: Bo = 1.40 -> 1.10
    bo = [[1.20, 1.00],
          [1.40, 1.10]]
    # Для простоты остальные таблицы пустые
    return {
        'pressure': p,
        'temperature': t,
        'bo': bo
    }

def test_bilinear_bo():
    cfg = {
        'temperature': 60.0,   # между 40 и 80
        'pvt': build_pvt()
    }
    fluid = Fluid(cfg, DummyReservoir(), device='cpu')

    # Точка (P=20 МПа, T=60°C) – должна быть средняя по обоим направлениям
    p = torch.tensor(20e6)  # Па
    Bo = fluid.calc_bo(p)
    # Ручная билинейная: сначала интерполяция по T
    #   для 10 MPa: 1.20 -> 1.40 => 1.30 (середина)
    #   для 30 MPa: 1.00 -> 1.10 => 1.05
    # затем по P (середина): 1.30 -> 1.05 => 1.175
    expected = 1.175
    assert math.isclose(Bo.item(), expected, rel_tol=1e-6), f"Bo {Bo.item()} vs {expected}"

    # Проверяем dBo/dP аналитическую (должна быть -0.025 1/MPa = -2.5e-8 1/Pa)
    dBo = fluid.calc_dbo_dp(p)
    expected_slope = (1.05 - 1.30) / (20e6)  # ΔBo / ΔP same middle difference
    assert math.isclose(dBo.item(), expected_slope, rel_tol=1e-6), f"dBo {dBo.item()} vs {expected_slope}"

    # Finite difference sanity
    eps = 1e5  # 0.1 MPa
    Bo_plus = fluid.calc_bo(p + eps)
    Bo_minus = fluid.calc_bo(p - eps)
    fd = (Bo_plus - Bo_minus) / (2*eps)
    assert math.isclose(dBo.item(), fd.item(), rel_tol=1e-3) 