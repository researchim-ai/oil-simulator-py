"""Entropy identity, dissipation sign, caloric consistency and wall conservation."""

import sys
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import ThermalHomogeneousPipe
from surface.thermal_flux import (
    entropy_conservative_flux,
    entropy_density,
    entropy_potential,
    entropy_stable_flux,
    entropy_variables,
    entropy_viscosity_sensor,
    logarithmic_mean,
)


def pipe(**kwargs):
    return ThermalHomogeneousPipe(10, 0.1, 4, 850, 500, 300, 1, 0, 0, **kwargs)


@pytest.mark.parametrize(
    "a,b",
    [(1.0, 1.0), (1.0, 1 + 2e-12), (1e-100, 1e100), (1e100, 1.001e100), (1e308, 1.000000001e308)],
)
def test_logarithmic_mean_against_high_precision(a, b):
    with localcontext() as ctx:
        ctx.prec = 80
        x, y = Decimal.from_float(a), Decimal.from_float(b)
        expected = float(x if x == y else (y - x) / (y.ln() - x.ln()))
    actual = logarithmic_mean(a, b)
    assert actual == pytest.approx(expected, rel=3e-15)
    assert logarithmic_mean(b, a) == pytest.approx(actual, rel=1e-15)
    assert min(a, b) <= actual <= max(a, b)


def test_entropy_variables_are_derivatives_of_physical_entropy():
    p = pipe()
    u = p.primitive(1e6, 0.9, 20, 320)
    analytic = entropy_variables(p, u)
    for k in range(4):
        h = 1e-6 * max(abs(u[k]), 1)
        left, right = u.copy(), u.copy()
        left[k] -= h
        right[k] += h
        numeric = -(entropy_density(p, right) - entropy_density(p, left)) / (2 * h)
        assert analytic[k] == pytest.approx(numeric, rel=2e-7, abs=1e-7)


def test_tadmor_identity_and_diffusion_inequality_for_variable_composition():
    p = pipe()
    rng = np.random.default_rng(20261102)

    def states():
        return p.primitive(
            np.exp(rng.uniform(np.log(1e5), np.log(1e7), 128)),
            rng.uniform(0.02, 0.99, 128),
            rng.uniform(-100, 100, 128),
            rng.uniform(200, 550, 128),
        )

    left, right = states(), states()
    delta = entropy_variables(p, right) - entropy_variables(p, left)
    potential = entropy_potential(p, right) - entropy_potential(p, left)
    central = entropy_conservative_flux(p, left, right)
    scale = np.maximum(1, abs(potential) + np.sum(abs(delta * central), axis=-1))
    error = np.sum(delta * central, axis=-1) - potential
    assert np.max(abs(error) / scale) < 2e-12
    dissipative = entropy_stable_flux(p, left, right)
    production = potential - np.sum(delta * dissipative, axis=-1)
    assert np.min(production / scale) >= -2e-12
    assert np.min(np.sum(delta * (right - left), axis=-1)) >= 0


def test_temperature_regularization_is_entropy_dissipating_and_only_changes_energy():
    p = pipe()
    left = p.primitive(1e6, 0.9, 2, 300)
    right = p.primitive(1.1e6, 0.85, -1, 340)
    no_heat = entropy_stable_flux(replace(p, thermal_entropy_dissipation=0), left, right)
    with_heat = entropy_stable_flux(p, left, right)
    change = with_heat - no_heat
    assert change[:3] == pytest.approx([0, 0, 0], abs=1e-14)
    assert change[3] < 0
    delta = entropy_variables(p, right) - entropy_variables(p, left)
    assert np.dot(delta, change) < 0


def test_consistent_flux_equal_states_and_frozen_mass_ratio():
    p = pipe()
    u = p.primitive(1e6, 0.98, 12, 310)
    assert p._interface_flux(u, u) == pytest.approx(p._flux(u), rel=2e-14, abs=1e-8)
    pressures = np.array([1e6, 2e6])
    temperatures = np.array([280.0, 350.0])
    ratio = 0.3 / 0.7
    alpha = 850 / (850 + ratio * pressures / (500 * temperatures))
    states = p.primitive(pressures, alpha, [3, 7], temperatures)
    flux = p._interface_flux(states[0], states[1])
    assert flux[0] / flux[1] == pytest.approx(ratio, rel=2e-13)


def test_reflecting_wall_has_exact_zero_component_and_energy_flux():
    p = pipe()
    u = p.primitive(1e6, 0.98, 50, 300)
    reflected = u.copy()
    reflected[2] *= -1
    flux = p._interface_flux(u, reflected)
    assert flux[[0, 1, 3]] == pytest.approx([0, 0, 0], abs=1e-12)


def test_material_contact_stays_in_pressure_temperature_equilibrium():
    p = pipe()
    initial = p.initialize(1e6, [0.2, 0.2, 0.8, 0.8])
    result = p.advance(initial, 0.005, initial_dt_s=0.001)
    assert result["properties"]["pressure_pa"] == pytest.approx([1e6] * 4, abs=0.001)
    assert result["properties"]["temperature_k"] == pytest.approx([300] * 4, abs=1e-7)
    assert all(s["entropy_production_j_k"] >= -1e-7 for s in result["steps"])
    assert sum(result["state"]["total_energy_j"]) == pytest.approx(
        sum(initial.total_energy_j), abs=1e-5
    )


def test_legacy_rusanov_and_numerical_options_are_explicit():
    p = pipe(numerical_flux="rusanov")
    l = p.primitive(1e6, 0.9, 2, 300)
    r = p.primitive(1.1e6, 0.85, -1, 340)
    speed = max(p._wave_speed(l), p._wave_speed(r))
    expected = 0.5 * (p._flux(l) + p._flux(r)) - 0.5 * speed * (r - l)
    assert p._interface_flux(l, r) == pytest.approx(expected, rel=1e-14)
    for kw in (
        {"numerical_flux": "unknown"},
        {"thermal_entropy_dissipation": -1},
        {"thermal_entropy_dissipation": float("nan")},
        {"thermal_entropy_dissipation": float("inf")},
    ):
        with pytest.raises(ValueError):
            pipe(**kw)
    with pytest.raises(TypeError):
        pipe(adaptive_entropy_dissipation=1)


def test_curvature_sensor_is_local_bounded_and_detects_material_contacts():
    p = replace(pipe(), cells=16)
    uniform = p.primitive(np.full(16, 1e6), np.full(16, 0.9), np.zeros(16), np.full(16, 300.0))
    assert entropy_viscosity_sensor(p, uniform, None, None) == pytest.approx(np.zeros(15))
    contact = uniform.copy()
    contact[8:] = p.primitive(1e6, 0.2, 0, 300)
    factor = entropy_viscosity_sensor(p, contact, None, None)
    assert np.isfinite(factor).all() and np.min(factor) >= 0 and np.max(factor) <= 1
    assert factor[7] > 0.99
    changed = contact.copy()
    changed[3] = p.primitive(2e6, 0.8, 10, 350)
    modified = entropy_viscosity_sensor(p, changed, None, None)
    # Only faces whose four-cell stencil contains cell 3 can change.
    assert modified[5:] == pytest.approx(factor[5:], abs=0)
    assert modified[:1] == pytest.approx(factor[:1], abs=0)


def test_reduced_dissipation_preserves_entropy_sign_and_zero_limit():
    p = pipe()
    left = p.primitive(1e6, 0.9, 2, 300)
    right = p.primitive(1.1e6, 0.85, -1, 340)
    delta = entropy_variables(p, right) - entropy_variables(p, left)
    potential = entropy_potential(p, right) - entropy_potential(p, left)
    for factor in (0, 0.1, 0.5, 1):
        flux = entropy_stable_flux(p, left, right, factor)
        assert potential - np.dot(delta, flux) >= -1e-8
    assert entropy_stable_flux(p, left, right, 0) == pytest.approx(
        entropy_conservative_flux(p, left, right)
    )


def test_sensor_is_computed_once_per_attempt_not_inside_newton(monkeypatch):
    import surface.thermal_pipe as implementation

    calls = []
    original = implementation.entropy_viscosity_sensor

    def measured(pipe, state, left, right):
        calls.append(state.copy())
        return original(pipe, state, left, right)

    monkeypatch.setattr(implementation, "entropy_viscosity_sensor", measured)
    p = pipe()
    state = p.initialize(1e6, 0.9, 1)
    result = p.advance(state, 1e-4, initial_dt_s=1e-4)
    assert len(calls) == len(result["steps"]) + result["failed_steps"]
    assert len(calls) == 1
    assert result["steps"][0]["nonlinear_evaluations"] > 1
    calls.clear()
    constant = replace(p, adaptive_entropy_dissipation=False)
    constant.advance(constant.initialize(1e6, 0.9, 1), 1e-4, initial_dt_s=1e-4)
    assert not calls
