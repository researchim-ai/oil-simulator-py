"""Conservative PR momentum/kinetics: drag, acoustic modes and boundaries."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import root

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parents[1] / "scripts"),
]
from check_stratified_pr import Reference

from surface.equilibrium_properties import EquilibriumFluid
from surface.liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
from surface.pr_momentum_pipe import PRDirichletBoundary, PRMomentumPipe, PRMomentumState

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def fluid(device="cpu"):
    config = json.loads((ROOT / "configs/surface_stratified_pr_gas.json").read_text())["system"][
        "fluid"
    ]
    return EquilibriumFluid.from_dict(config, device=device)


def model(device="cpu", **options):
    config = {
        "length_m": 1.0,
        "diameter_m": 0.05,
        "cells": 4,
        "wave_speed_bound_m_s": 500.0,
        "maximum_velocity_m_s": 50.0,
        "boundary_mode": "periodic",
        "initial_step_s": 1e-5,
    }
    config.update(options)
    return PRMomentumPipe(fluid(device), **config)


@pytest.mark.parametrize("device", DEVICES)
def test_uniform_periodic_motion_and_serialization(device):
    m = model(device)
    s = m.initialize(1e6, 320.0, [0.5, 0.5], 2.0)
    before = s.to_dict()
    out, report = m._step(s, 1e-4, (None, None))
    assert np.allclose(out.component_moles, s.component_moles, rtol=1e-10, atol=1e-12)
    assert out.momentum_kg_m_s == pytest.approx(s.momentum_kg_m_s, abs=1e-10)
    assert out.total_energy_j == pytest.approx(s.total_energy_j, abs=1e-7)
    assert max(map(abs, report["conservation_residuals"])) < 1e-8
    assert s.to_dict() == before
    restored = PRMomentumPipe.from_dict(json.loads(json.dumps(m.to_dict())), device=device)
    restored.inspect(PRMomentumState.from_dict(json.loads(json.dumps(out.to_dict()))))


@pytest.mark.parametrize("device", DEVICES)
def test_uniform_laminar_drag_converts_kinetic_to_internal_energy(device):
    m = model(device, dynamic_viscosity_pa_s=0.01)
    s = m.initialize(1e6, 320.0, [0.5, 0.5], 2.0)
    dt = 0.01
    old = m._decode(m.fluid.tensor(s.solution_variables), m._support(s.component_moles))
    final, report = m._step(s, dt, (None, None))
    data = m._decode(m.fluid.tensor(final.solution_variables), m._support(final.component_moles))
    rho = float(old["q"]["density_kg_m3"][0])
    expected = 2 / (1 + dt * 32 * 0.01 / (rho * 0.05**2))
    assert data["v"].cpu().tolist() == pytest.approx([expected] * m.cells, rel=1e-8)
    assert sum(final.total_energy_j) == pytest.approx(sum(s.total_energy_j), abs=1e-7)
    assert float(data["internal"].sum() - old["internal"].sum()) == pytest.approx(
        float(old["kinetic"].sum() - data["kinetic"].sum()), abs=1e-5
    )
    assert float(data["t"].min()) > 320
    assert report["entropy_production_j_k"] > 0
    ref = Reference(m.fluid, np.arange(2))
    n = sum(s.component_moles[0])
    vm = m.volume / n
    target_u = (
        s.total_energy_j[0] / n
        - 0.5
        * sum(x * c.molar_mass_kg_mol for x, c in zip([0.5, 0.5], m.fluid.components))
        * expected**2
    )
    solved = root(
        lambda pt: [
            (ref.state(pt[0], pt[1], [0.5, 0.5], "stable")["vm"] - vm) / vm,
            (ref.state(pt[0], pt[1], [0.5, 0.5], "stable")["u"] - target_u) / 1000,
        ],
        [1e6, 320.0],
    )
    assert solved.success
    assert float(data["t"][0]) == pytest.approx(solved.x[1], abs=1e-6)


def test_sound_speed_matches_independent_entropy_derivative():
    m = model()
    f = m.fluid
    ref = Reference(f, np.arange(2))
    p, t = 1e6, 320.0
    dp, dt = 10.0, 0.0032
    a, b = ref.state(p + dp, t, [0.5, 0.5], "stable"), ref.state(p - dp, t, [0.5, 0.5], "stable")
    c, d = ref.state(p, t + dt, [0.5, 0.5], "stable"), ref.state(p, t - dt, [0.5, 0.5], "stable")
    rp = (a["rho"] - b["rho"]) / (2 * dp)
    rt = (c["rho"] - d["rho"]) / (2 * dt)
    sp = (a["s"] - b["s"]) / (2 * dp)
    st = (c["s"] - d["s"]) / (2 * dt)
    assert m.sound_speed(p, t, [0.5, 0.5]) == pytest.approx(
        1 / math.sqrt(rp - rt * sp / st), rel=1e-7
    )


def test_reflecting_walls_keep_mass_energy_and_account_for_momentum_reaction():
    m = model(boundary_mode="ends")
    s = m.initialize(1e6, 320.0, [0.5, 0.5], 2.0)
    final, report = m._step(s, 1e-4, m._prepare(s, None, None))
    assert report["left_flux"][:2] == [0.0, 0.0]
    assert report["right_flux"][:2] == [0.0, 0.0]
    assert report["left_flux"][-1] == 0
    assert report["right_flux"][-1] == 0
    assert sum(final.momentum_kg_m_s) < sum(s.momentum_kg_m_s)
    assert sum(final.total_energy_j) == pytest.approx(sum(s.total_energy_j), abs=1e-7)


def test_acoustic_right_running_mode_matches_linear_discrete_euler_reference():
    m = model(cells=8, length_m=8.0)
    p, t = 1e6, 320.0
    ref = Reference(m.fluid, np.arange(2))
    base = ref.state(p, t, [0.5, 0.5], "stable")
    c = m.sound_speed(p, t, [0.5, 0.5])
    dp, dt = 10.0, 0.0032
    sp = (
        ref.state(p + dp, t, [0.5, 0.5], "stable")["s"]
        - ref.state(p - dp, t, [0.5, 0.5], "stable")["s"]
    ) / (2 * dp)
    st = (
        ref.state(p, t + dt, [0.5, 0.5], "stable")["s"]
        - ref.state(p, t - dt, [0.5, 0.5], "stable")["s"]
    ) / (2 * dt)
    phase = 2 * np.pi * (np.arange(m.cells) + 0.5) / m.cells
    amplitude = 1.0
    perturbation = amplitude * np.cos(phase)
    initial = m.initialize(
        (p + perturbation).tolist(),
        (t - sp / st * perturbation).tolist(),
        [0.5, 0.5],
        (perturbation / (base["rho"] * c)).tolist(),
    )
    h = 1e-4
    final, _ = m._step(initial, h, (None, None))
    data = m._decode(m.fluid.tensor(final.solution_variables), m._support(final.component_moles))
    coefficient = (data["p"].cpu().numpy() - p) @ np.exp(-1j * phase) / (m.cells / 2)
    kd = 2 * np.pi / m.cells
    lam = -m.wave_speed_bound_m_s * (1 - np.cos(kd)) / m.dx - 1j * c * np.sin(kd) / m.dx
    expected = amplitude / (1 - h * lam)
    assert coefficient == pytest.approx(expected, rel=2e-4, abs=1e-5)


def test_pressure_driven_flow_reverses_by_boundary_swap():
    m = model(boundary_mode="ends")
    s = m.initialize(995000.0, 320.0, [0.5, 0.5])
    high = PRDirichletBoundary(1e6, 320.0, (0.5, 0.5), m.fluid.model_id)
    low = replace(high, pressure_pa=990000.0)
    forward, a = m._step(s, 1e-4, m._prepare(s, high, low))
    backward, b = m._step(s, 1e-4, m._prepare(s, low, high))
    assert sum(forward.momentum_kg_m_s) > 0
    assert sum(backward.momentum_kg_m_s) < 0
    assert forward.momentum_kg_m_s == pytest.approx(
        [-v for v in backward.momentum_kg_m_s[::-1]], abs=1e-8
    )
    assert max(map(abs, a["conservation_residuals"])) < 1e-7
    assert max(map(abs, b["conservation_residuals"])) < 1e-7


def test_atomic_budget_tamper_and_insufficient_wave_bound():
    m = model(maximum_attempts=1)
    s = m.initialize(1e6, 320.0, [0.5, 0.5], 2.0)
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        m.advance(s, 1e-4)
    bad = replace(s, total_energy_j=tuple(e + 1 for e in s.total_energy_j))
    with pytest.raises(ConservationError):
        m.inspect(bad)
    with pytest.raises(PropertyModelError, match="speed bound"):
        model(wave_speed_bound_m_s=200.0).initialize(1e6, 320.0, [0.5, 0.5])


def test_public_advance_restart_and_friction_balance():
    m = model(dynamic_viscosity_pa_s=0.01)
    s = m.initialize(1e6, 320.0, [0.5, 0.5], 2.0)
    original = s.to_dict()
    report = m.advance(s, 1e-4)
    final = PRMomentumState.from_dict(json.loads(json.dumps(report["state"])))
    assert final.time_s == 1e-4
    assert report["final_summary"]["velocity_m_s"][0] < 2.0
    assert sum(report["final_summary"]["kinetic_energy_j"]) < sum(
        report["initial_summary"]["kinetic_energy_j"]
    )
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    assert s.to_dict() == original
    resumed = PRMomentumPipe.from_dict(m.to_dict())
    resumed.inspect(final)
    second = resumed.advance(final, 1e-4)
    assert second["state"]["time_s"] == 2e-4


def test_acoustic_spatial_refinement_reduces_continuum_error():
    reference = None
    errors = []
    p, t = 1e6, 320.0
    for count in (8, 16):
        m = model(cells=count, length_m=8.0)
        c = m.sound_speed(p, t, [0.5, 0.5])
        ref = Reference(m.fluid, np.arange(2))
        dp, dt = 10.0, 0.0032
        sp = (
            ref.state(p + dp, t, [0.5, 0.5], "stable")["s"]
            - ref.state(p - dp, t, [0.5, 0.5], "stable")["s"]
        ) / (2 * dp)
        st = (
            ref.state(p, t + dt, [0.5, 0.5], "stable")["s"]
            - ref.state(p, t - dt, [0.5, 0.5], "stable")["s"]
        ) / (2 * dt)
        rho = ref.state(p, t, [0.5, 0.5], "stable")["rho"]
        phase = 2 * np.pi * (np.arange(count) + 0.5) / count
        delta = np.cos(phase)
        initial = m.initialize(
            (p + delta).tolist(),
            (t - sp / st * delta).tolist(),
            [0.5, 0.5],
            (delta / (rho * c)).tolist(),
        )
        final, _ = m._step(initial, 1e-4, (None, None))
        data = m.describe(final)
        gain = (np.array(data["pressure_pa"]) - p) @ np.exp(-1j * phase) / (count / 2)
        reference = np.exp(-1j * c * (2 * np.pi / 8) * 1e-4)
        errors.append(abs(gain - reference))
    assert errors[1] < errors[0] * 0.7


def test_general_jacobian_matches_finite_differences():
    m = model(cells=3, dynamic_viscosity_pa_s=0.01)
    s = m.initialize([999900.0, 1e6, 1000100.0], [320.0, 320.1, 319.9], [0.5, 0.5], [0.5, 0.7, 0.9])
    y, _lo, _hi, evaluate = m._problem(s, 1e-5, (None, None))
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-0.002, 0.002, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert torch.isfinite(jac).all()
    assert (jac @ direction - fd).abs().max() < 1e-6


def test_independent_audit_does_not_call_runtime_solver_and_rejects_kinetic_tamper(monkeypatch):
    from check_pr_momentum_pipe import audit

    m = model(dynamic_viscosity_pa_s=0.01)
    s = m.initialize(1e6, 320.0, [0.5, 0.5], 2.0)
    result = m.advance(s, 1e-4)
    config = {
        "model": m.to_dict(),
        "initial": {
            "pressure_pa": 1e6,
            "temperature_k": 320.0,
            "mole_fractions": [0.5, 0.5],
            "velocity_m_s": 2.0,
        },
        "schedule": [{"duration_s": 1e-4}],
    }
    report = {"config": config, "model": m.to_dict(), "initial": s.to_dict(), "intervals": [result]}

    def forbidden(*args, **kwargs):
        raise AssertionError("Auditor called runtime dynamics")

    monkeypatch.setattr(PRMomentumPipe, "_decode", forbidden)
    monkeypatch.setattr(PRMomentumPipe, "_step", forbidden)
    monkeypatch.setattr(PRMomentumPipe, "inspect", forbidden)
    assert audit(config, report)["accepted"]
    wrong = json.loads(json.dumps(report))
    wrong["intervals"][0]["final_summary"]["kinetic_energy_j"][0] *= 2
    with pytest.raises(ValueError):
        audit(config, wrong)
    wrong = json.loads(json.dumps(report))
    wrong["intervals"][0]["steps"][0]["friction_impulse_kg_m_s"] = 0.0
    with pytest.raises(ValueError):
        audit(config, wrong)
    wrong = json.loads(json.dumps(report))
    wrong["intervals"][0]["steps"][0]["state"]["total_energy_j"][0] += 0.1
    with pytest.raises(ValueError):
        audit(config, wrong)


def test_dirichlet_numerical_boundary_full_interval_passes_audit():
    from check_pr_momentum_pipe import audit

    m = model(boundary_mode="ends")
    s = m.initialize(995000.0, 320.0, [0.5, 0.5])
    high = PRDirichletBoundary(1e6, 320.0, (0.5, 0.5), m.fluid.model_id)
    low = replace(high, pressure_pa=990000.0)
    result = m.advance(s, 1e-5, left=high, right=low)
    left = {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]}
    right = {**left, "pressure_pa": 990000.0}
    config = {
        "model": m.to_dict(),
        "initial": {},
        "schedule": [{"duration_s": 1e-5, "left": left, "right": right}],
    }
    report = {"config": config, "model": m.to_dict(), "initial": s.to_dict(), "intervals": [result]}
    assert audit(config, report)["accepted"]
