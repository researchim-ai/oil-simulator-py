"""Vanishing circular apertures retain real area and exact endpoint Jacobians."""

import math

import numpy as np
import pytest
import torch

from surface.phase_geometry import pickup_fractions
from surface.three_phase_properties import disk_area_moment, disk_band_area


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_small_upper_cap_is_not_erased_by_subtracting_almost_full_disks(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    d, height = 0.01, 0.01
    top = height + d / 2
    interface = torch.tensor(top - 1e-14, dtype=torch.float64, device=device)
    eps = top - float(interface.cpu())

    def fractions(x):
        return pickup_fractions(
            {
                "phase_bottoms_m": torch.stack((x * 0, x)),
                "phase_tops_m": torch.stack((x, x * 0 + 0.02)),
            },
            height,
            d,
        )

    actual = fractions(interface)
    expected = 4 / 3 * math.sqrt(d) * eps**1.5 / (math.pi * d * d / 4)
    assert float(actual[1].cpu()) > 0
    assert float(actual[1].cpu()) == pytest.approx(expected, rel=1e-10)
    forward = torch.func.jacfwd(fractions)(interface)
    backward = torch.func.jacrev(fractions)(interface)
    torch.testing.assert_close(forward, backward, rtol=1e-12, atol=1e-16)
    slope = 2 * math.sqrt(eps * (d - eps)) / (math.pi * d * d / 4)
    assert float(forward[1].cpu()) == pytest.approx(-slope, rel=1e-10)
    assert float(fractions(interface.new_tensor(top))[1].cpu()) == 0.0


def test_band_area_matches_regular_analytic_geometry_and_vmap_jacobian():
    d = 0.01
    lower = torch.tensor([0.0, 0.001, 0.004, 0.007], dtype=torch.float64)
    upper = torch.tensor([0.001, 0.003, 0.006, 0.01], dtype=torch.float64)
    actual = disk_band_area(lower, upper, d)
    expected = disk_area_moment(upper, d)[0] - disk_area_moment(lower, d)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-20)
    jac = torch.func.jacfwd(lambda z: disk_band_area(z, upper, d))(lower)
    np.testing.assert_allclose(
        jac.numpy(),
        np.diag(-2 * np.sqrt(lower.numpy() * (d - lower.numpy()))),
        rtol=1e-12,
        atol=1e-20,
    )
