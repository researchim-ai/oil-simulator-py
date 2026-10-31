"""Independent limiting solutions for the explicitly no-slip pipe closure."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.gas import GasState, IsothermalGasPipe
from surface.liquid import HydraulicConvergenceError, PropertyModelError
from surface.multiphase import HomogeneousFluid, HomogeneousPipe


def test_water_laminar_gravity_and_friction_exact():
    f = HomogeneousFluid()
    p = HomogeneousPipe("rise", 100, 0.1, 20)
    r = p.run(2e6, 300, 0, 0.01, 0, f)
    q = 0.01 / f.water_density_kg_m3
    exact = (
        2e6
        - f.water_density_kg_m3 * 9.80665 * 20
        - 128 * f.water_viscosity_pa_s * 100 * q / (math.pi * 0.1**4)
    )
    assert r["outlet_pressure_pa"] == pytest.approx(exact, abs=1e-7)
    assert abs(r["mechanical_balance_error_w"]) < 1e-8


def test_pure_gas_matches_independent_integral_solution():
    f = HomogeneousFluid()
    p = HomogeneousPipe("gas", 2000, 0.08, roughness_m=1e-5)
    r = p.run(6e6, 300, 0, 0, 0.4, f)
    ref = IsothermalGasPipe("ref", 2000, 0.08, 1e-5).run(GasState(0.4, 6e6, 300), f.gas)
    assert r["outlet_pressure_pa"] == pytest.approx(ref["outlet"]["pressure_pa"], abs=0.01)
    assert abs(r["mechanical_balance_error_w"]) < 1e-5


def test_three_phase_expansion_and_energy_closure():
    f = HomogeneousFluid()
    r = HomogeneousPipe("mixture", 1200, 0.08, 300, 1e-5).run(8e6, 320, 1, 1, 0.1, f)
    assert 0 < r["outlet_pressure_pa"] < 8e6
    assert r["outlet_gas_volume_fraction"] > r["inlet_gas_volume_fraction"]
    assert r["outlet_velocity_m_s"] > r["inlet_velocity_m_s"]
    assert r["phase_mass_rates_kg_s"] == {"oil": 1, "water": 1, "gas": 0.1}
    assert r["friction_dissipation_w"] > 0
    assert abs(r["mechanical_balance_error_w"]) < 1e-4


def test_liquid_mixture_laminar_closure_exact():
    f = HomogeneousFluid()
    oil, water = 0.02, 0.03
    q = oil / f.oil_density_kg_m3 + water / f.water_density_kg_m3
    rho = (oil + water) / q
    mu = (
        oil / f.oil_density_kg_m3 * f.oil_viscosity_pa_s
        + water / f.water_density_kg_m3 * f.water_viscosity_pa_s
    ) / q
    r = HomogeneousPipe("ow", 500, 0.1, -100).run(2e6, 300, oil, water, 0, f)
    exact = 2e6 + rho * 9.80665 * 100 - 128 * mu * 500 * q / (math.pi * 0.1**4)
    assert r["outlet_pressure_pa"] == pytest.approx(exact, abs=1e-6)


def test_domain_failures_are_explicit():
    f = HomogeneousFluid()
    p = HomogeneousPipe("bad", 1000, 0.01)
    with pytest.raises(PropertyModelError):
        p.run(1e6, 300, 0, 0, 0, f)
    with pytest.raises(PropertyModelError):
        p.run(1e6, 300, 0, 0, 10, f)
    with pytest.raises(HydraulicConvergenceError):
        p.run(1e6, 300, 0, 0, 0.01, f)


def test_three_phase_pressure_against_integral_quadrature():
    """Solve the separated momentum integral instead of marching pressure in space."""
    from types import SimpleNamespace

    from scipy.integrate import quad
    from scipy.optimize import brentq

    f = HomogeneousFluid()
    pipe = HomogeneousPipe("mixed_reference", 1200, 0.08, 300, 1e-5)
    r = pipe.run(8e6, 320, 1, 1, 0.1, f)
    mass = 2.1
    flux = mass / (math.pi * 0.08**2 / 4)
    a = (1 / 850 + 1 / 1000) / mass
    b = 0.1 / mass * 500 * 320
    edge = IsothermalGasPipe("reference", 1200, 0.08, 1e-5)

    def ds_dp(p):
        v = a + b / p
        mu = ((1 / 850 * 0.005 + 1 / 1000 * 0.001) / mass + b / p * 1e-5) / v
        friction = edge.friction_factor(mass, SimpleNamespace(viscosity_pa_s=mu))
        return (1 - flux**2 * b / p**2) / (friction * flux**2 * v / (0.16) + 9.80665 * 0.25 / v)

    expected = brentq(lambda p: quad(ds_dp, p, 8e6, epsabs=1e-8)[0] - 1200, 4e6, 8e6, xtol=1e-6)
    assert r["outlet_pressure_pa"] == pytest.approx(expected, abs=0.001)
