"""VFP interpolation/extrapolation and its pressure/rate derivatives."""
import numpy as np
import pytest
import torch
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator.vfp import VFPProd


def _table(device, singleton):
    table = VFPProd.__new__(VFPProd)
    table.flo_type, table.wfr_type, table.gfr_type = 'LIQ', 'WCT', 'GOR'
    axes = [np.array(a, dtype=float) for a in
            ([1e5, 3e5, 7e5], [.1, .4, .8], [1., 4., 8.], [0., 2., 5.], [.1, .4, 1.])]
    for k in singleton:
        axes[k] = axes[k][1:2]
    for name, axis in zip(('thp', 'wfr', 'gfr', 'alq', 'flo'), axes):
        setattr(table, name, torch.tensor(axis, dtype=torch.float64))
    values = np.random.default_rng(73).uniform(1e6, 8e6, tuple(map(len, axes)))
    table.table = torch.tensor(values, dtype=torch.float64)
    return table, RegularGridInterpolator(axes, values, bounds_error=False, fill_value=None)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('shape', [(), (5,), (2, 3)])
@pytest.mark.parametrize('singleton', [(), (1, 3), (0, 1, 2, 3, 4)])
def test_vfp_values_and_derivatives_match_independent_interpolation(device, shape, singleton):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    tab, reference = _table(device, singleton)
    values = [.17, .09, .8, 9e5, 6.]
    args = [torch.full(shape, v, dtype=torch.float64, device=device, requires_grad=True)
            for v in values]

    def expected(xs):
        qo, qw, qg, thp, alq = xs
        coords = np.stack(np.broadcast_arrays(thp, qw / (qo + qw), qg / qo, alq, qo + qw), axis=-1)
        return reference(coords.reshape(-1, 5)).reshape(shape)

    result = tab.bhp(*args)
    numpy_args = [x.detach().cpu().numpy() for x in args]
    np.testing.assert_allclose(result.detach().cpu(), expected(numpy_args), rtol=2e-13, atol=1e-7)
    gradients = torch.autograd.grad(result.sum(), args)
    for k, (value, gradient) in enumerate(zip(values, gradients)):
        h = max(abs(value), 1.) * 1e-5
        plus, minus = list(numpy_args), list(numpy_args)
        plus[k], minus[k] = plus[k] + h, minus[k] - h
        fd = (expected(plus) - expected(minus)) / (2 * h)
        np.testing.assert_allclose(gradient.detach().cpu(), fd, rtol=2e-7, atol=2e-3)
    # Repeat with cached corner offsets and different interpolation cells.
    more = [a.detach() * .6 for a in args]
    np.testing.assert_allclose(tab.bhp(*more).cpu(), expected([a.cpu().numpy() for a in more]),
                               rtol=2e-13, atol=1e-7)
