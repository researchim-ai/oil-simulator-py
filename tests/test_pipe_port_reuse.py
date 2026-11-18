"""Reused port flashes retain the independently evaluated public reports."""
import pytest
import torch
from test_surface_blackoil_properties import model

from surface.blackoil_pipe import BlackOilPipe


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('rates', [[.85, 2., .06], [.85, 2., .144], [0., 2., 0.]])
@pytest.mark.parametrize('inverse', [False, True])
def test_port_report_is_exact_to_independent_flash(device, rates, inverse):
    m = model().to(device)
    pipe = BlackOilPipe('riser', 300, .08, 30, 1e-5)
    run = pipe.required_inlet if inverse else pipe.run
    result = run(2.72e6 if inverse else 3e6, rates, m)
    for port in ('inlet', 'outlet'):
        assert result[port+'_state'] == m.state(result[port+'_pressure_pa'], rates)
    assert abs(result['mechanical_balance_error_w']) < 1e-5
