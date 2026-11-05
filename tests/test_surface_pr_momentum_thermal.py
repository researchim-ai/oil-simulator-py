"""Thermal Euler pipe checked against independent fixed-volume PR balances."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.optimize import root
from test_surface_pr_momentum_pipe import Reference, model

from surface.liquid import HydraulicConvergenceError, PropertyModelError
from surface.pr_momentum_thermal import PRThermalMomentumPipe, PRThermalMomentumState
from surface.vessel_wall import VesselWall


def thermal(**options):
    return PRThermalMomentumPipe(model(**options), VesselWall(100.0, 20.0, 5.0, 290.0, 320.0))


def test_independent_uniform_fixed_volume_energy_and_drag():
    m = thermal(dynamic_viscosity_pa_s=0.01)
    p, w = m.pipe, m.wall
    s = m.initialize(p.initialize(1e6, 320.0, [0.5, 0.5], 2.0), 350.0)
    dt = 0.005
    out, report = m._step(s, dt, (None, None))
    summary = p.describe(out.pipe)
    rho = p.describe(s.pipe)["density_kg_m3"][0]
    v = 2 / (1 + dt * 32 * 0.01 / (rho * p.diameter_m**2))
    ref = Reference(p.fluid, np.arange(2))
    n = sum(s.pipe.component_moles[0])
    mass = rho * p.volume
    capacity, ua, ambient = (
        w.heat_capacity_j_k / p.cells,
        w.fluid_ua_w_k / p.cells,
        w.ambient_ua_w_k / p.cells,
    )

    def residual(z):
        pressure, temperature, wall = z
        q = ref.state(pressure, temperature, [0.5, 0.5], "stable")
        heat = dt * ua * (wall - temperature)
        return [
            (n * q["vm"] - p.volume) / p.volume,
            (n * q["u"] + 0.5 * mass * v * v - s.pipe.total_energy_j[0] - heat) / 100.0,
            (capacity * (wall - 350.0) + heat - dt * ambient * (290.0 - wall)) / 100.0,
        ]

    solved = root(residual, [1e6, 320.0, 350.0])
    assert solved.success
    assert summary["temperature_k"] == pytest.approx([solved.x[1]] * p.cells, abs=1e-6)
    assert summary["velocity_m_s"] == pytest.approx([v] * p.cells, abs=1e-8)
    assert out.wall_temperature_k == pytest.approx([solved.x[2]] * p.cells, abs=1e-7)
    assert report["entropy_production_j_k"] > 0
    combined = (
        sum(out.pipe.total_energy_j)
        - sum(s.pipe.total_energy_j)
        + capacity * sum(t - 350.0 for t in out.wall_temperature_k)
    )
    assert combined == pytest.approx(sum(report["ambient_heat_j"]), abs=1e-7)


def test_zero_ua_exactly_matches_adiabatic_pipe():
    p = model(dynamic_viscosity_pa_s=0.01)
    m = PRThermalMomentumPipe(p, VesselWall(100.0, 0.0))
    initial = p.initialize(1e6, 320.0, [0.5, 0.5], 2.0)
    s = m.initialize(initial, [330.0, 340.0, 350.0, 360.0])
    expected, _ = p._step(initial, 0.001, (None, None))
    out, _ = m._step(s, 0.001, (None, None))
    assert out.pipe.solution_variables == pytest.approx(expected.solution_variables, abs=1e-12)
    assert out.wall_temperature_k == s.wall_temperature_k


def test_wall_distribution_restart_exact_endpoint_and_atomic_failure():
    m = thermal(initial_step_s=1e-4, maximum_step_s=1e-3)
    s = m.initialize(m.pipe.initialize(1e6, 320.0, [0.5, 0.5]), [310.0, 320.0, 330.0, 340.0])
    original = s.to_dict()
    report = m.advance(s, 0.0002)
    out = PRThermalMomentumState.from_dict(json.loads(json.dumps(report["state"])))
    restored = PRThermalMomentumPipe.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(out)
    assert out.time_s == 0.0002
    assert abs(report["combined_energy_residual_j"]) < 1e-7
    assert s.to_dict() == original
    assert out.wall_temperature_k[0] > 310.0
    assert out.wall_temperature_k[-1] < 340.0
    a = m.advance(out, 0.0001)
    b = restored.advance(out, 0.0001)
    assert a["state"] == b["state"]
    with pytest.raises(PropertyModelError):
        restored.inspect(replace(out, wall_temperature_k=(300.0,)))
    failed = thermal(maximum_attempts=1, initial_step_s=1e-5)
    fs = failed.initialize(failed.pipe.initialize(1e6, 320.0, [0.5, 0.5]), 350.0)
    before = fs.to_dict()
    with pytest.raises(HydraulicConvergenceError):
        failed.advance(fs, 0.001)
    assert fs.to_dict() == before


def test_thermal_jacobian_directional_difference():
    m = thermal()
    s = m.initialize(m.pipe.initialize(1e6, 320.0, [0.5, 0.5], 2.0), 350.0)
    y, _, _, evaluate = m._problem(s, 0.01, (None, None))
    direction = torch.linspace(-0.2, 0.2, len(y), dtype=y.dtype, device=y.device)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    eps = 1e-6
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-6
