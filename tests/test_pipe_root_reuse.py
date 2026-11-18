"""Accepted Newton tape gives the same implicit Jacobian as a fresh integral."""
import pytest
import torch
from test_surface_blackoil_properties import model

from surface.blackoil_pipe import BlackOilPipe
from surface.blackoil_pipe_batch import PipeBatch


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_reused_root_jacobian_exact_to_fresh_full_input_tape(device, monkeypatch):
    m = model().to(device)
    pipes = [BlackOilPipe('phase', 300, .08, 30, 1e-5),
             BlackOilPipe('gas', 100, .15, 5, 2e-5, 2),
             BlackOilPipe('water', 80, .03, 0, 1e-5)]
    boundaries = [2.72e6, 1.1e6, 2.4e6]
    rates = [[.85, 2., .06], [.85, 1., .144], [0., .001, 0.]]
    batch = PipeBatch(pipes, m)
    checked = batch.checked
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return checked(*args, **kwargs)
    monkeypatch.setattr(batch, 'checked', counted)
    rows = batch.solve(boundaries, rates)
    assert len(calls) == rows[0]['pressure_evaluations']
    x = torch.tensor([[row['inlet_pressure_pa'], b, *q, p.loss_coefficient]
                      for row,b,q,p in zip(rows,boundaries,rates,pipes)],
                     dtype=torch.float64, device=device, requires_grad=True)
    integral,order,error = checked(x[:,1],x[:,0],x[:,2:5],x[:,5])
    gradient = torch.autograd.grad(integral[:,0].sum(),x)[0]
    expected = (-gradient[:,1:]/gradient[:,:1]).detach().cpu().tolist()
    for row,derivative,err in zip(rows,expected,error.detach().cpu().tolist()):
        assert list(row['pressure_derivatives'].values()) == derivative
        assert row['quadrature_order'] == order
        assert row['quadrature_length_error_m'] == err
