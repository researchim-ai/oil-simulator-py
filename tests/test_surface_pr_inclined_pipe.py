"""Well-balanced inclined PR pipe: all states reconstructed in joint Newton."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_surface_pr_momentum_pipe import fluid, model

from surface.pr_inclined_pipe import PRInclinedMomentumPipe
from surface.pr_momentum_pipe import PRMomentumPipe, PRMomentumState
from surface.pr_momentum_thermal import PRThermalMomentumPipe
from surface.vessel_wall import VesselWall


def inclined(**options):
    return PRInclinedMomentumPipe(
        fluid(),
        **{
            "length_m": 2.0,
            "diameter_m": 0.05,
            "cells": 4,
            "wave_speed_bound_m_s": 500.0,
            "maximum_velocity_m_s": 50.0,
            "elevation_change_m": 1.0,
            "initial_step_s": 1e-5,
            **options,
        },
    )


@pytest.mark.parametrize(
    "pressure,temperature,x",
    [(1e6, 320.0, [0.5, 0.5]), (2e6, 350.0, [0.7, 0.3]), (1e6, 320.0, [1.0, 0.0])],
)
def test_hydrostatic_family_remains_at_rest(pressure, temperature, x):
    p = inclined()
    s = p.initialize_equilibrium(pressure, temperature, x)
    initial = s.to_dict()
    out, r = p._step(s, 1e-3, (None, None))
    assert p.describe(out)["velocity_m_s"] == pytest.approx([0.0] * p.cells, abs=1e-8)
    assert np.max(np.abs(np.array(out.component_moles) - s.component_moles)) < 1e-10
    assert out.total_energy_j == pytest.approx(s.total_energy_j, abs=1e-7)
    assert r["body_force_impulse_kg_m_s"] < 0
    assert max(map(abs, r["conservation_residuals"])) < 1e-8
    assert s.to_dict() == initial


def test_uniform_pressure_accelerates_downhill_and_preserves_u_k_pe():
    p = inclined()
    s = p.initialize(1e6, 320.0, [0.5, 0.5])
    out, r = p._step(s, 1e-4, (None, None))
    assert all(v < 0 for v in p.describe(out)["velocity_m_s"])
    assert sum(out.total_energy_j) == pytest.approx(sum(s.total_energy_j), abs=1e-7)
    assert r["entropy_production_j_k"] >= -1e-8


def test_horizontal_limit_matches_existing_euler_pipe():
    p = inclined(elevation_change_m=0.0, dynamic_viscosity_pa_s=0.01)
    h = model(length_m=2.0, boundary_mode="ends", dynamic_viscosity_pa_s=0.01)
    a = p.initialize(1e6, 320.0, [0.5, 0.5], 2.0)
    b = h.initialize(1e6, 320.0, [0.5, 0.5], 2.0)
    a, _ = p._step(a, 1e-4, (None, None))
    b, _ = h._step(b, 1e-4, (None, None))
    assert a.component_moles == pytest.approx(np.asarray(b.component_moles), abs=1e-10)
    assert a.momentum_kg_m_s == pytest.approx(b.momentum_kg_m_s, abs=1e-9)
    assert a.total_energy_j == pytest.approx(b.total_energy_j, abs=1e-6)


def test_joint_hydrostatic_jacobian_and_thermal_wall():
    p = inclined()
    s = p.initialize(1e6, 320.0, [0.5, 0.5], 0.1)
    y, _, _, evaluate = p._problem(s, 1e-4, (None, None))
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-0.01, 0.01, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-6
    thermal = PRThermalMomentumPipe(p, VesselWall(100.0, 2.0, 1.0, 300.0, 320.0))
    ts = thermal.initialize(s, 330.0)
    report = thermal.advance(ts, 1e-5)
    assert abs(report["combined_energy_residual_j"]) < 1e-6
    restored = PRThermalMomentumPipe.from_dict(json.loads(json.dumps(thermal.to_dict())))
    from surface.pr_momentum_thermal import PRThermalMomentumState

    restored.inspect(PRThermalMomentumState.from_dict(report["state"]))


def test_restart_checks_face_variables_and_gauge():
    from surface.liquid import ConservationError

    p = inclined()
    s = p.initialize_equilibrium(1e6, 320.0, [0.5, 0.5])
    restored = PRMomentumPipe.from_dict(json.loads(json.dumps(p.to_dict())))
    restored.inspect(PRMomentumState.from_dict(json.loads(json.dumps(s.to_dict()))))
    shifted = inclined(reference_elevation_m=10000.0)
    t = shifted.initialize_equilibrium(1e6, 320.0, [0.5, 0.5])
    assert s.total_energy_j == t.total_energy_j
    values = list(s.solution_variables)
    values[-2] += 0.001
    with pytest.raises(ConservationError):
        p.inspect(replace(s, solution_variables=tuple(values)))


def test_gravitational_force_matches_independent_density_and_temporal_refinement():
    from test_surface_pr_momentum_pipe import Reference

    p = inclined()
    s = p.initialize(1e6, 320.0, [0.5, 0.5])
    _, r = p._step(s, 1e-7, (None, None))
    rho = Reference(p.fluid, np.array([0, 1])).state(1e6, 320.0, [0.5, 0.5], "stable")["rho"]
    expected = (
        -1e-7 * p.area * p.length_m * rho * p.gravity_m_s2 * p.elevation_change_m / p.length_m
    )
    assert r["body_force_impulse_kg_m_s"] == pytest.approx(expected, rel=3e-6, abs=1e-12)
    answers = []
    for pieces in (1, 2, 4):
        current = s
        for _ in range(pieces):
            current, _ = p._step(current, 0.001 / pieces, (None, None))
        answers.append(np.array(current.momentum_kg_m_s))
    assert np.linalg.norm(answers[1] - answers[2]) < 0.8 * np.linalg.norm(answers[0] - answers[2])


def test_inclined_spatial_refinement_of_perturbed_equilibrium():
    answers = []
    for cells in (4, 8, 16):
        p = inclined(cells=cells)
        s = p.initialize_equilibrium(1e6, 320.0, [0.5, 0.5])
        d = p.describe(s)
        pp = np.array(d["pressure_pa"]) + 100 * np.sin(2 * np.pi * (np.arange(cells) + 0.5) / cells)
        s = p.initialize(pp.tolist(), 320.0, d["mole_fractions"])
        for _ in range(4):
            s, _ = p._step(s, 5e-5, (None, None))
        answers.append(np.array(p.describe(s)["velocity_m_s"]).reshape(4, -1).mean(-1))
    assert np.linalg.norm(answers[1] - answers[2]) < 0.8 * np.linalg.norm(answers[0] - answers[2])


@pytest.mark.parametrize("periodic", [False, True])
def test_independent_inclined_audit_rejects_face_energy_and_force_corruption(monkeypatch, periodic):
    from copy import deepcopy

    from check_pr_momentum_pipe import audit

    p = inclined(
        boundary_mode="periodic" if periodic else "ends",
        elevation_change_m=0.0 if periodic else 1.0,
    )
    initial = {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]}
    if periodic:
        initial["velocity_m_s"] = 2.0
        s = p.initialize(**initial)
    else:
        s = p.initialize_equilibrium(**initial)
        initial["hydrostatic"] = True
    config = {"model": p.to_dict(), "initial": initial, "schedule": [{"duration_s": 1e-5}]}
    report = json.loads(
        json.dumps(
            {
                "config": config,
                "model": p.to_dict(),
                "initial": s.to_dict(),
                "intervals": [p.advance(s, 1e-5)],
            }
        )
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference invoked runtime inclined reconstruction")

    monkeypatch.setattr(PRInclinedMomentumPipe, "_decode", forbidden)
    monkeypatch.setattr(PRInclinedMomentumPipe, "inspect", forbidden)
    monkeypatch.setattr(PRInclinedMomentumPipe, "_problem", forbidden)
    assert audit(config, report)["accepted"]
    broken = deepcopy(report)
    broken["intervals"][0]["steps"][0]["body_force_impulse_kg_m_s"] += 1.0
    with pytest.raises(ValueError):
        audit(config, broken)
    broken = deepcopy(report)
    broken["intervals"][0]["steps"][0]["state"]["solution_variables"][-2] += 0.001
    with pytest.raises(ValueError):
        audit(config, broken)
    broken = deepcopy(report)
    broken["intervals"][0]["steps"][0]["state"]["total_energy_j"][0] += 0.1
    with pytest.raises(ValueError):
        audit(config, broken)


def test_geometry_budget_and_failed_interval_leave_checkpoint_unchanged():
    from surface.liquid import HydraulicConvergenceError

    with pytest.raises(ValueError, match="Elevation change"):
        inclined(length_m=0.5, elevation_change_m=1.0)
    with pytest.raises(ValueError, match="periodic"):
        inclined(boundary_mode="periodic")
    oversized = inclined(cells=128)
    with pytest.raises(ValueError, match="budget"):
        oversized.initialize(1e6, 320.0, [0.5, 0.5])
    p = inclined(maximum_attempts=1)
    state = p.initialize_equilibrium(1e6, 320.0, [0.5, 0.5])
    before = state.to_dict()
    with pytest.raises(HydraulicConvergenceError):
        p.advance(state, 0.001)
    assert state.to_dict() == before
