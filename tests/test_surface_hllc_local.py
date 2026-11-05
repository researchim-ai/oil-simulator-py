"""Supersonic and contact invariants; EOS acoustic identity vs independent FD."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from surface.hllc_local import hllc_local


def ideal(rho, p, u, gamma=1.4):
    rho, p, u = (torch.tensor(x, dtype=torch.float64) for x in (rho, p, u))
    e = p / (gamma - 1) + 0.5 * rho * u * u
    state = torch.stack((rho, rho * u, e))
    flux = torch.stack((rho * u, rho * u * u + p, (e + p) * u))
    return (state, flux, p, u, rho, rho * (p / rho**gamma).log()), torch.sqrt(gamma * p / rho)


@pytest.mark.parametrize("velocity", [3.0, -3.0])
@pytest.mark.parametrize("estimate", ["davis", "pvrs"])
def test_supersonic_uniform_direction_uses_physical_upwind_flux(velocity, estimate):
    left, a = ideal(1.0, 1.0, velocity)
    right, b = ideal(0.9, 0.8, velocity)
    flux, entropy, _, _, ok, fallback = hllc_local(left, right, a, b, 10.0, estimate=estimate)
    expected = left if velocity > 0 else right
    assert bool(ok) and not bool(fallback)
    torch.testing.assert_close(flux, expected[1], rtol=1e-14, atol=1e-14)
    assert float(entropy) == pytest.approx(float(expected[-1] * expected[3]))


@pytest.mark.parametrize("velocity", [0.0, 0.3, -0.3])
@pytest.mark.parametrize("estimate", ["davis", "pvrs"])
def test_exact_contact_discontinuity_is_preserved(velocity, estimate):
    left, a = ideal(1.0, 1.0, velocity)
    right, b = ideal(0.1, 1.0, velocity)
    flux, _, contact, pressure, ok, fallback = hllc_local(left, right, a, b, 10.0, estimate=estimate)
    assert bool(ok) and not bool(fallback)
    torch.testing.assert_close(flux, left[1] if velocity >= 0 else right[1], rtol=1e-13, atol=1e-13)
    assert float(contact) == pytest.approx(velocity, abs=1e-14)
    assert float(pressure) == pytest.approx(1.0, abs=1e-14)


def test_analytic_pr_sound_matches_pressure_entropy_finite_differences():
    from test_surface_compositional_separator import binary

    f = binary()
    x = f.tensor([0.5, 0.5])
    p = 1e6
    t = 320.0
    q = f.homogeneous(p, t, x)

    def values(p, t):
        z = f.homogeneous(p, t, x)
        return torch.stack((z["density_kg_m3"], z["molar_entropy_j_mol_k"]))

    dp = (values(p + 10, t) - values(p - 10, t)) / 20
    dt = (values(p, t + 0.001) - values(p, t - 0.001)) / 0.002
    denominator = dp[0] - dt[0] * dp[1] / dt[1]
    expected = torch.rsqrt(denominator)
    torch.testing.assert_close(q["frozen_sound_speed_m_s"], expected, rtol=1e-7, atol=1e-7)


def test_pure_component_entropy_tangent_is_finite_on_fixed_support():
    from test_surface_compositional_separator import binary

    f = binary()

    def entropy(y):
        x = torch.stack((y[2], y[2] * 0))
        return f.tensor_homogeneous(y[0], y[1], x)["molar_entropy_j_mol_k"]

    y = f.tensor([1000.0, 400.0, 1.0])
    direction = f.tensor([10.0, 0.1, 0.0])
    tangent = torch.func.jvp(entropy, (y,), (direction,))[1]
    finite_difference = (entropy(y + 1e-3 * direction) - entropy(y - 1e-3 * direction)) / 2e-3
    assert bool(torch.isfinite(torch.func.jacfwd(entropy)(y)).all())
    torch.testing.assert_close(tangent, finite_difference, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("estimate", ["davis", "pvrs"])
def test_sod_shock_tube_converges_to_exact_solution_and_conserves(estimate):
    import math

    import numpy as np

    gamma = 1.4
    pstar = 0.30313017805064685
    ustar = 0.9274526200489499
    aleft = math.sqrt(gamma)
    aright = math.sqrt(gamma * 0.1 / 0.125)
    astar = aleft * pstar ** ((gamma - 1) / (2 * gamma))
    shock = aright * math.sqrt((gamma + 1) / (2 * gamma) * pstar / 0.1 + (gamma - 1) / (2 * gamma))
    rho_l = pstar ** (1 / gamma)
    ratio = pstar / 0.1
    beta = (gamma - 1) / (gamma + 1)
    rho_r = 0.125 * (ratio + beta) / (beta * ratio + 1)
    errors = []
    for count in [80, 160, 320]:
        dx = 1 / count
        x = (torch.arange(count, dtype=torch.float64) + 0.5) * dx
        rho = torch.where(x < 0.5, torch.ones_like(x), torch.full_like(x, 0.125))
        p = torch.where(x < 0.5, torch.ones_like(x), torch.full_like(x, 0.1))
        state = torch.stack((rho, torch.zeros_like(rho), p / (gamma - 1)), -1)
        initial = state.sum(0) * dx
        exchange = torch.zeros(3, dtype=torch.float64)
        time = 0.0

        def face_state(u):
            density = u[:, 0]
            v = u[:, 1] / density
            pp = (gamma - 1) * (u[:, 2] - 0.5 * density * v * v)
            assert bool((density > 0).all() & (pp > 0).all())
            sound = torch.sqrt(gamma * pp / density)
            flux = torch.stack((density * v, density * v * v + pp, (u[:, 2] + pp) * v), -1)
            entropy = density * (pp / density**gamma).log() / (gamma - 1)
            return (u, flux, pp, v, density, entropy), sound

        while time < 0.2:
            data, sound = face_state(state)
            dt = min(0.35 * dx / float((data[3].abs() + sound).max()), 0.2 - time)
            extended = torch.cat((state[:1], state, state[-1:]))
            q, a = face_state(extended)
            flux, _, _, _, ok, _ = hllc_local(
                tuple(v[:-1] for v in q), tuple(v[1:] for v in q), a[:-1], a[1:], 8.0,
                estimate=estimate,
            )
            assert bool(ok.all())
            exchange += dt * (flux[0] - flux[-1])
            state = state - dt / dx * (flux[1:] - flux[:-1])
            time += dt
        xi = (x.numpy() - 0.5) / 0.2
        exact = np.where(xi < 0, 1.0, 0.125)
        fan = (xi >= -aleft) & (xi < ustar - astar)
        af = 2 / (gamma + 1) * (aleft - (gamma - 1) * xi[fan] / 2)
        exact[fan] = (af / aleft) ** (2 / (gamma - 1))
        exact[(xi >= ustar - astar) & (xi < ustar)] = rho_l
        exact[(xi >= ustar) & (xi < shock)] = rho_r
        errors.append(float(np.abs(state[:, 0].numpy() - exact).mean()))
        final = state.sum(0) * dx
        torch.testing.assert_close(final - initial, exchange, rtol=1e-11, atol=1e-11)
        # Outflow boundaries have a small numerical tail before the physical
        # wave arrives. Account for that transfer instead of declaring it zero.
        assert abs(float(final[0] - initial[0])) < 1e-7
        assert abs(float(final[2] - initial[2])) < 1e-6
        assert float(final[1]) == pytest.approx(0.9 * 0.2, abs=1e-7)
    assert errors[1] < 0.85 * errors[0]
    assert errors[2] < 0.85 * errors[1]
    assert errors[-1] < 0.015
