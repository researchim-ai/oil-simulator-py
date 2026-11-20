"""PR analytic local tangents against independent forward AD and finite differences."""

import json
from pathlib import Path

import pytest
import torch

from surface.chemical_kernel import _jacobian
from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_chemical_derivatives import directional, logit_jacobian, tangent_hessian


def mixture(device="cpu"):
    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/surface_pr_integrated_facility_liquid_mixture.json"
        ).read_text()
    )
    return EquilibriumFluid.from_dict(cfg["model"]["plant"]["transport"]["fluid"], device=device)


@pytest.mark.parametrize("support", [[0, 1], [1, 2], [0, 1, 2]])
def test_analytic_stationarity_matches_forward_ad_across_roots_and_traces(support):
    f = mixture()
    active = torch.tensor(support)
    p = torch.linspace(1e5, 5e6, 31, dtype=torch.float64)
    t = torch.linspace(280.0, 460.0, 31, dtype=torch.float64)
    v = torch.linspace(-75.0, 75.0, 31 * (len(support) - 1), dtype=torch.float64).reshape(31, -1)
    mu = p.new_zeros((31, len(support)))
    with torch.no_grad():
        expected = _jacobian(f, p, t, mu, active, v)
        actual = logit_jacobian(f, p, t, mu, active, v)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_general_direction_with_asymmetric_interaction_matches_ad():
    f = mixture()
    f.kij[0, 1] = 0.07
    p, t, x = map(f.tensor, (1.4e6, 330.0, [0.2, 0.3, 0.5]))
    direction = f.tensor([[0.1, -0.2], [0.3, 0.1], [-0.4, 0.1]])
    expected = torch.func.jacfwd(
        lambda z: f.tensor_homogeneous(p, t, x + direction @ z)["log_fugacity_coefficients"]
    )(f.tensor([0.0, 0.0]))
    actual, _ = directional(f, p, t, x, direction)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("point", [[0.2, -0.5], [-70.0, -30.0], [70.0, 0.5]])
def test_tangent_gibbs_curvature_matches_forward_ad_away_from_stationarity(point):
    f = mixture()
    active = torch.arange(3)
    w = f.tensor([*point, 0.0]).softmax(-1)
    root = w.sqrt()
    pivot = w.argmax()
    direction = root + torch.nn.functional.one_hot(pivot, 3)
    house = (
        torch.eye(3, dtype=torch.float64)
        - 2 * direction[:, None] * direction[None, :] / direction.square().sum()
    )
    basis = house[:, torch.arange(3) != pivot]
    p, t, mu = map(f.tensor, (1e6, 340.0, [0.2, -0.3, 0.5]))

    def gradient(v):
        rr = root + basis @ v
        total = rr.square().sum()
        x = rr.square() / total
        q = f.tensor_homogeneous(p, t, x)
        delta = x.log() + q["log_fugacity_coefficients"] - mu
        potential = (x * delta).sum()
        return 2 / total * (basis.T @ (rr * (delta - potential)))

    expected = torch.func.jacfwd(gradient)(f.tensor([0.0, 0.0]))
    actual = tangent_hessian(f, p, t, mu, active, root, basis)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)


def test_composition_direction_matches_finite_differences():
    f = mixture()
    p, t, x = map(f.tensor, (1.2e6, 350.0, [0.25, 0.25, 0.5]))
    dx = f.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    actual, _ = directional(f, p, t, x, dx)
    h = 1e-5
    expected = torch.stack(
        [
            (
                f.tensor_homogeneous(p, t, x + h * d)["log_fugacity_coefficients"]
                - f.tensor_homogeneous(p, t, x - h * d)["log_fugacity_coefficients"]
            )
            / (2 * h)
            for d in dx.T
        ],
        -1,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
