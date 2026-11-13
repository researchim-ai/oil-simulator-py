"""Constrained Gibbs curvature must retain traces and reject saddles."""

import pytest
import torch

from surface.equilibrium_stability_batch import stability_candidates

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class RegularSolution:
    components = (None, None)

    def __init__(self, interaction):
        self.interaction = interaction

    def tensor_homogeneous(self, p, t, x):
        return {
            "valid": torch.ones_like(p, dtype=torch.bool),
            "log_fugacity_coefficients": self.interaction
            * torch.stack((x[1].square(), x[0].square())),
        }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trace_first", [True, False])
def test_ideal_trace_minima_have_finite_positive_scaled_curvature(device, trace_first):
    dtype = torch.float64
    f = RegularSolution(0.0)
    z = torch.tensor([1e-30, 1.0] if trace_first else [1.0, 1e-30], device=device, dtype=dtype)
    active = torch.arange(2, device=device)
    point = (z[:-1].log() - z[-1].log()).reshape(1, 1)
    args = (
        f,
        torch.tensor([1e6], device=device, dtype=dtype),
        torch.tensor([350.0], device=device, dtype=dtype),
        z.log()[None],
        active,
        point,
    )
    scaled = stability_candidates(*args)
    legacy = stability_candidates(*args, _scaled_curvature=False)
    assert bool(scaled[-1].all())
    assert not bool(legacy[-1].any())
    torch.testing.assert_close(scaled[1][0], z, rtol=2e-14, atol=0.0)
    assert abs(float(scaled[0][0])) < 1e-12


@pytest.mark.parametrize("device", DEVICES)
def test_stationary_regular_solution_saddle_is_not_a_minimum(device):
    f = RegularSolution(3.0)
    p = torch.tensor([1e6], device=device, dtype=torch.float64)
    t = torch.full_like(p, 350.0)
    z = torch.full((2,), 0.5, device=device, dtype=torch.float64)
    mu = z.log() + f.tensor_homogeneous(p[0], t[0], z)["log_fugacity_coefficients"]
    point = torch.zeros((1, 1), device=device, dtype=torch.float64)
    for scaled in (False, True):
        value, _, _, accepted = stability_candidates(
            f, p, t, mu[None], torch.arange(2, device=device), point, _scaled_curvature=scaled
        )
        assert not bool(accepted.any())
        assert abs(float(value[0])) < 1e-12


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("offset", [1.0, 1e4])
def test_trace_stationarity_polish_is_invariant_to_reference_offset(device, offset):
    f = RegularSolution(0.0)
    z = torch.tensor([1.0, 1e-30], device=device, dtype=torch.float64)
    mu = z.log() + offset
    point = (z[0].log() - z[1].log() + 5).reshape(1, 1)
    p = torch.tensor([1e6], device=device, dtype=torch.float64)
    t = torch.full_like(p, 350.0)
    value, x, _, accepted = stability_candidates(
        f, p, t, mu[None], torch.arange(2, device=device), point
    )
    assert bool(accepted.all())
    torch.testing.assert_close(x[0], z, rtol=2e-10, atol=0.0)
    assert float(value[0]) == pytest.approx(-offset, abs=1e-9)
