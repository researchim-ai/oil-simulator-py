"""Temporal velocity uses conserved momentum, never an extra phase solve."""
from dataclasses import replace

import pytest
import torch

from test_surface_pr_equilibrium_transport import pipe


@pytest.mark.parametrize('temperature', [320., 196.2])
@pytest.mark.parametrize('direction', [-1., 1.])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_velocity_defect_matches_full_decode_without_repeating_eos(monkeypatch, temperature, direction, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model, old = pipe(temperature=temperature, velocity=direction)
    if device == 'cuda':
        model = type(model).from_dict(model.to_dict(), device=device)
    changed = replace(old, nodes=tuple(
        (name, replace(cell, momentum_kg_m_s=cell.momentum_kg_m_s * 1.1))
        for name, cell in old.nodes))
    expected = max(float((model._state_data(i, a)['u'] - model._state_data(i, b)['u']).abs())
                   for i, ((_, a), (_, b)) in enumerate(zip(old.nodes, changed.nodes)))
    expected /= model.velocity_tolerance_m_s

    def forbidden(*args, **kwargs):
        raise AssertionError('A temporal velocity defect must not repeat the EOS')

    monkeypatch.setattr(model, '_state_data', forbidden)
    assert model._error(old, changed) == expected
    assert model._error(old, old) == 0.
