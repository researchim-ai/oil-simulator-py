"""Analytic standard-well mixing must agree with the general AD reference."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator import blackoil_fim as bo


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('n', [2, 5, 16])
@pytest.mark.parametrize('regime', ['normal', 'pvt_floor', 'zero_supply'])
def test_standard_mixing_matches_full_automatic_jacobian(monkeypatch, device, n, regime):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    dev = torch.device(device)
    generator = torch.Generator().manual_seed(912 + n)

    def rand(*shape):
        return torch.rand(shape, generator=generator, dtype=torch.float64).to(dev)

    rho = torch.tensor([800., 1000., 1.], dtype=torch.float64, device=dev)
    fake = SimpleNamespace(conn_cell=torch.arange(n, device=dev),
                           conn_well=torch.zeros(n, dtype=torch.long, device=dev),
                           rho_sc=rho, msw={}, device=dev)
    v = torch.zeros((bo.N_PROPS, n), dtype=torch.float64, device=dev)
    d = torch.zeros((bo.N_PROPS, n, 3), dtype=torch.float64, device=dev)
    v[bo.I_B:bo.I_B + 3] = torch.stack((.8 + rand(n), .9 + rand(n), 50. + 30. * rand(n)))
    v[bo.I_RS], v[bo.I_RV], v[bo.I_LT] = 20. + 10. * rand(n), .001 * rand(n), .5 + rand(n)
    rows = [bo.I_B, bo.I_B + 1, bo.I_B + 2, bo.I_RS, bo.I_RV, bo.I_LT]
    d[rows] = .001 * (rand(6, n, 3) - .5)
    d[bo.I_P, :, 0] = 1.
    draw = torch.where(torch.arange(n, device=dev) < n // 2, 1. + rand(n), -1. - rand(n))
    if regime == 'pvt_floor':
        v[bo.I_B + 1] = 1e-14
    elif regime == 'zero_supply':
        draw = -draw.abs()
    mobility = (.1 + rand(n, 3)) * rho
    raw = mobility * draw[:, None]
    raw_jac = rand(n, 3, 3) - .5
    raw_bhp = -mobility
    wi = .1 + rand(n)
    back = draw < 0

    def evaluate():
        q, dq, db = bo.BlackOilWells._producer_crossflow(
            fake, v, d, raw, raw_jac, raw_bhp, draw, back, wi)
        return q, dq, db, fake._producer_mix[0][2].clone()

    analytic = evaluate()
    monkeypatch.setattr(bo.BlackOilWells, '_standard_crossflow_blocks', staticmethod(lambda *args: None))
    automatic = evaluate()
    for a, b in zip(analytic, automatic):
        torch.testing.assert_close(a, b, rtol=5e-11, atol=1e-10)
