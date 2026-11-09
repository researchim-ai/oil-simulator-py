"""Independent radial Stokes solution, force balance and failure envelope."""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad, solve_bvp

from surface.annular_film import countercurrent_film, film_mobility
from surface.liquid import HydraulicConvergenceError


@pytest.mark.parametrize("alpha", [0.00001, 0.003, 0.03, 0.3])
def test_mobility_against_independent_radial_stokes_boundary_value_problem(alpha):
    radius = 0.085 / 2
    inner = radius * math.sqrt(1 - alpha)
    mu, gradient, shear = 0.001002, 9000.0, 0.4
    # r-domain BVP, no analytic profile or runtime quadrature reused.
    gap = radius - inner
    mesh = np.linspace(0.0, 1.0, 40)
    solution = solve_bvp(
        lambda z, y: np.vstack((y[1], gap**2 * gradient / mu - gap * y[1] / (inner + gap * z))),
        lambda ya, yb: np.array([ya[1] + gap * shear / mu, yb[0]]),
        mesh,
        np.zeros((2, len(mesh))),
        tol=1e-9,
        max_nodes=2000,
    )
    assert solution.success
    mean = quad(
        lambda r: 2 * r * solution.sol((r - inner) / gap)[0] / (radius**2 - inner**2),
        inner,
        radius,
        epsabs=1e-12,
    )[0]
    w, h = film_mobility(torch.tensor(alpha, dtype=torch.float64), radius * 2)
    assert float((-w * gradient + h * shear) / mu) == pytest.approx(mean, abs=1e-10, rel=1e-9)


def test_countercurrent_equal_opposite_shear_and_both_phase_force_balances():
    jg = torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64)
    r = countercurrent_film(jg, 0.06, 1.2)
    a = r["liquid_holdup"]
    area, g = math.pi * 0.085**2 / 4, 9.80665
    gas = -area * (1 - a) * (r["pressure_gradient_pa_m"] + 1.2 * g) + r["gas_interfacial_force_n_m"]
    liquid = (
        -area * a * (r["pressure_gradient_pa_m"] + 998.2 * g)
        + r["liquid_interfacial_force_n_m"]
        + r["liquid_wall_force_n_m"]
    )
    torch.testing.assert_close(gas, torch.zeros_like(a), atol=1e-14, rtol=0)
    torch.testing.assert_close(liquid, torch.zeros_like(a), atol=1e-14, rtol=0)
    assert torch.equal(r["gas_interfacial_force_n_m"], -r["liquid_interfacial_force_n_m"])
    assert bool((r["liquid_velocity_m_s"] < 0).all())
    assert bool((r["liquid_wall_force_n_m"] > 0).all())
    assert bool((r["liquid_velocity_residual_m_s"].abs() < 1e-8).all())


def test_film_domain_and_blocked_roots_do_not_silently_extrapolate():
    with pytest.raises(ValueError, match="laminar domain"):
        countercurrent_film(7.0, 0.2, 1.2)
    with pytest.raises(ValueError, match="countercurrent"):
        countercurrent_film(-7.0, 0.06, 1.2)
    with pytest.raises(HydraulicConvergenceError, match="No steady"):
        countercurrent_film(1000.0, 0.06, 1.2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_film_keeps_outputs_on_device_and_matches_cpu():
    jg = torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64)
    cpu = countercurrent_film(jg, 0.06, 1.2)
    gpu = countercurrent_film(jg.cuda(), 0.06, 1.2)
    for key in cpu:
        assert gpu[key].is_cuda
        torch.testing.assert_close(gpu[key].cpu(), cpu[key], atol=1e-10, rtol=1e-10)


def test_external_challenge_retains_failure_and_input_only_exclusions():
    from validate_annular_film_experiment import ROOT, validate

    result = validate(ROOT / "tests/data/surface/hannover_countercurrent")
    assert result["fit_performed"] is False
    assert result["accepted_as_validated_for_annular_countercurrent"] is False
    assert result["matched_wet_windows"] == 16
    assert len(result["excluded_by_laminar_domain"]) == 4
    assert all(v["mean_absolute_error_pa"] > 4 for v in result["roughness_sensitivity"].values())
