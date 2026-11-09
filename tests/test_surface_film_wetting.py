"""Dry-patch force, contact-angle hysteresis and integration with annular source."""

import math
from copy import deepcopy

import pytest
import torch

from surface.annular_film import film_mobility
from surface.film_wetting import FilmWettingCriterion
from surface.liquid import PropertyModelError
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_material_wetting_diagnostic_cuda_matches_cpu():
    criterion = FilmWettingCriterion(0.02, 20.0, 80.0)
    alpha = torch.tensor([1e-5, 0.02, 0.02], dtype=torch.float64)
    velocity = alpha.new_tensor([-1.0, -0.2, -1.0])
    cpu = criterion.diagnose(alpha, 0.1, 0.001, alpha * 0 + 600, velocity, alpha * 0)
    gpu = criterion.diagnose(
        alpha.cuda(), 0.1, 0.001, (alpha * 0 + 600).cuda(), velocity.cuda(), (alpha * 0).cuda()
    )
    for key in cpu:
        torch.testing.assert_close(cpu[key], gpu[key].cpu(), rtol=1e-10, atol=1e-12)


def test_thin_gravity_limit_recovers_hartley_murgatroyd_without_empirical_fit():
    rho, mu, sigma, theta, g = 600.0, 0.001, 0.02, 45.0, 9.80665
    delta = (15 * mu**2 * sigma * (1 - math.cos(math.radians(theta))) / (rho**3 * g * g)) ** 0.2
    criterion = FilmWettingCriterion(sigma, theta, theta)
    errors = []
    for diameter in (1.0, 10.0):
        alpha = torch.tensor(1 - (1 - 2 * delta / diameter) ** 2, dtype=torch.float64)
        w, _ = film_mobility(alpha, diameter)
        mean = -w * rho * g / mu
        result = criterion.diagnose(alpha, diameter, mu, alpha.new_tensor(rho), mean, alpha * 0)
        assert bool(result["profile_valid"])
        expected = sigma * (1 - math.cos(math.radians(theta)))
        errors.append(abs(float(result["momentum_force_n_m"]) / expected - 1))
    assert errors[1] < errors[0] / 5
    assert errors[1] < 0.001


@pytest.mark.parametrize(
    "receding,advancing", [(45.0, 20.0), (-1.0, 20.0), (10.0, 180.0), (float("nan"), 30.0)]
)
def test_material_angles_are_required_ordered_and_finite(receding, advancing):
    with pytest.raises(ValueError):
        FilmWettingCriterion(0.02, receding, advancing)


def test_hysteresis_is_a_real_indeterminate_band_not_automatic_dryout():
    c = FilmWettingCriterion(0.02, 20.0, 80.0)
    alpha = torch.tensor([0.02] * 3, dtype=torch.float64)
    out = c.diagnose(
        alpha, 0.1, 0.001, alpha * 0 + 600.0, alpha.new_tensor([-0.001, -0.2, -1.0]), alpha * 0
    )
    assert out["dry_patch_supported"].tolist() == [True, False, False]
    assert out["history_dependent"].tolist() == [False, True, False]
    assert out["full_wetting_supported"].tolist() == [False, False, True]


def test_counterflow_profile_and_thick_curvature_are_not_certified():
    c = FilmWettingCriterion(0.02, 0.0, 5.0)
    a = torch.tensor([0.02, 0.02, 0.3], dtype=torch.float64)
    out = c.diagnose(
        a,
        0.1,
        0.001,
        a * 0 + 600.0,
        a.new_tensor([0.0, -0.1, -1.0]),
        a.new_tensor([1.0, 100.0, 0.0]),
    )
    assert not bool(out["profile_valid"].any())
    assert not bool(out["full_wetting_supported"].any())


def test_optional_wetting_guard_is_source_bound_independently_audited_and_atomic():
    from check_pr_active_surface import friction_audit
    from test_surface_pr_active_annular_friction import model_state

    base, old = model_state()
    raw = base.to_dict()
    raw["wall_friction"]["wetting"] = FilmWettingCriterion(0.02, 0.0, 5.0, True).to_dict()
    pipe = PRActiveMultifluidPipe.from_dict(raw)
    old = pipe.initialize([{n: p for n, p in c} for c in old.cells])
    new, r = pipe.friction.apply(pipe, old, 0.001)
    friction_audit(pipe, old, new, r)
    assert all(v["full_wetting_supported"] for v in r["wetting"])
    bad = deepcopy(r)
    bad["wetting"][0]["momentum_force_n_m"] *= 2
    with pytest.raises(ValueError, match="audit differs"):
        friction_audit(pipe, old, new, bad)
    raw["wall_friction"]["wetting"] = FilmWettingCriterion(0.02, 80.0, 90.0, True).to_dict()
    dry = PRActiveMultifluidPipe.from_dict(raw)
    state = dry.initialize([{n: p for n, p in c} for c in old.cells])
    snapshot = state.to_dict()
    with pytest.raises(PropertyModelError, match="wetting"):
        dry.friction.apply(dry, state, 0.001)
    assert state.to_dict() == snapshot
