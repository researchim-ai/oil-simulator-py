"""Fugacity SSI is only a predictor; existing physical acceptance is retained."""

import pytest
import torch
from test_surface_stability_curvature import RegularSolution

from surface.equilibrium_ssi_predictor import refine_candidates, successive_points
from surface.equilibrium_stability_batch import stability_candidates, stability_objective

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def inputs(device, z, offsets):
    z = torch.tensor(z, device=device, dtype=torch.float64)
    v = (z[0].log() - z[1].log() + torch.tensor(offsets, device=device)).reshape(-1, 1)
    p = torch.full((len(v),), 1e6, device=device, dtype=torch.float64)
    return p, p * 0 + 350, z.log().expand(len(v), -1), torch.arange(2, device=device), v


@pytest.mark.parametrize("device", DEVICES)
def test_ideal_trace_ssi_satisfies_unweighted_chemical_potentials(device):
    f = RegularSolution(0.0)
    p, t, mu, active, v = inputs(device, [1.0, 1e-30], [-80, 0, 40])
    original = v.clone()
    out = successive_points(f, p, t, mu, active, v)
    before = stability_objective(f, active)(v, p, t, mu)[0]
    value, grad, residual, *_ = stability_objective(f, active)(out, p, t, mu)
    assert bool((value <= before + 1e-14).all())
    assert float(residual.abs().max().cpu()) < 1e-9
    assert float(grad.abs().max().cpu()) < 2e-9
    torch.testing.assert_close(v, original, rtol=0, atol=0)
    accepted = stability_candidates(f, p, t, mu, active, out)
    assert bool(accepted[-1].all())


@pytest.mark.parametrize("device", DEVICES)
def test_ssi_does_not_certify_stationary_saddle(device):
    f = RegularSolution(3.0)
    f._fugacity_ssi_predictor = True
    p, t, mu, active, v = inputs(device, [0.5, 0.5], [0])
    z = v.new_tensor([0.5, 0.5])
    mu = mu + f.tensor_homogeneous(p[0], t[0], z)["log_fugacity_coefficients"]
    assert not bool(stability_candidates(f, p, t, mu, active, v)[-1].any())


@pytest.mark.parametrize("device", DEVICES)
def test_failed_predictor_leaves_original_candidates_for_fallback(device, monkeypatch):
    import surface.equilibrium_ssi_predictor as ssi

    f = RegularSolution(0.0)
    p, t, mu, active, v = inputs(device, [0.7, 0.3], [0, 40])
    value, x, q, accepted = stability_candidates(f, p, t, mu, active, v)
    accepted = accepted.clone()
    accepted[1] = False
    original = value, x, q, accepted
    monkeypatch.setattr(ssi, "successive_points", lambda f, p, t, mu, active, v: v)
    assert refine_candidates(f, p, t, mu, active, v, original) is original
    assert not bool(original[-1][1])


@pytest.mark.parametrize("device", DEVICES)
def test_refinement_keeps_already_accepted_rows_unchanged(device):
    f = RegularSolution(0.0)
    p, t, mu, active, v = inputs(device, [0.7, 0.3], [0, 40])
    value, x, q, accepted = stability_candidates(f, p, t, mu, active, v)
    accepted = accepted.clone()
    accepted[1] = False
    original = value, x, q, accepted
    result = refine_candidates(f, p, t, mu, active, v, original)
    assert bool(result[-1].all()) and not bool(original[-1][1])
    for before, after in ((value, result[0]), (x, result[1]), (accepted, result[-1])):
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    for key in q:
        torch.testing.assert_close(q[key][0], result[2][key][0], rtol=0, atol=0)


def test_pr_minimum_matches_reference_across_feed_compositions():
    from test_surface_pr_analytic_chemical import mixture

    from surface.equilibrium_minimization_batch import minimize_stability_candidates

    f = mixture()
    active = torch.arange(3)
    p = f.tensor([1e5, 1e6, 3e6, 5e6])
    t = f.tensor([300, 350, 400, 450])
    for z in ([0.8, 0.15, 0.05], [0.99, 0.009999, 1e-6]):
        z = f.tensor(z)
        q = f.tensor_homogeneous(p, t, z)
        mu = z.log() + q["log_fugacity_coefficients"]
        starts = f.tensor([[-8, -4], [-1, 3], [1, 1], [7, -5]])
        with torch.no_grad():
            f._fugacity_ssi_predictor = False
            reference = minimize_stability_candidates(f, p, t, mu, active, starts)
            f._fugacity_ssi_predictor = True
            actual = minimize_stability_candidates(f, p, t, mu, active, starts)
        assert bool((actual[-1] | ~reference[-1]).all())
        both = actual[-1] & reference[-1]
        # Each accepted seed must remain a physically admissible minimum;
        # different solver paths may find different minima from that seed.
        assert bool(actual[2]["valid"][actual[-1]].all())
        if bool(both.any()):
            assert float(actual[0][both].min()) <= float(reference[0][both].min()) + 2e-8
