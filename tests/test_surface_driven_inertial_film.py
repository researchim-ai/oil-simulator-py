"""Driven existing mechanics: force/work consistency, reverse work and restart."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from test_surface_inertial_capillary_film import setup

from surface.inertial_capillary_film import InertialCapillaryFilm, InertialCapillaryState
from surface.planar_film import resistance_response


@pytest.mark.parametrize("scheme", ["backward", "roe"])
@pytest.mark.parametrize("drag", [False, True])
def test_uniform_driven_momentum_matches_independent_exact_time_discretization(scheme, drag):
    m = setup(drag=drag, count=8, time_scheme=scheme)
    s = m.surface
    h, u0, dt, gravity, shear = 3e-6, 0.01, 1e-5, 9.81, -0.01
    old = m.initialize([h] * 8, [u0] * 8)
    new, report = m.step(old, dt, tangential_gravity_m_s2=gravity, interface_shear_pa=shear)
    # Derive the profile elimination independently from the integrated profile.
    a = h**3 / (3 * s.mu) + s.slip * h * h / s.mu
    b = h * h / (2 * s.mu) + s.slip * h / s.mu
    eta = h * b / a if drag else 1.0
    resistance = s.mu / (h / 3 + s.slip) if drag else 0.0
    gamma = resistance / (s.rho * h)
    acceleration = gravity + eta * shear / (s.rho * h)
    if scheme == "backward":
        expected = (u0 + dt * acceleration) / (1 + dt * gamma)
    else:
        expected = (u0 * (1 - dt * gamma / 2) + dt * acceleration) / (1 + dt * gamma / 2)
    np.testing.assert_allclose(new.face_velocity_m_s, expected, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(new.thickness_m, h, rtol=1e-13, atol=1e-23)
    assert report["external_work_j"] == pytest.approx(
        report["gravity_work_j"] + report["interface_work_j"], abs=1e-25
    )
    assert abs(report["energy_residual_j"]) < 1e-22
    assert max(map(abs, report["cell_energy_residual_j"])) < 1e-22
    assert max(map(abs, report["cell_momentum_residual_kg_m_s"])) < 1e-22


def test_opposing_interface_removes_energy_and_signed_work_survives_restart():
    m = setup(drag=True, count=8)
    old = m.initialize([3e-6] * 8, [0.01] * 8)
    end, report = m.step(old, 1e-5, interface_shear_pa=-0.001)
    assert end.external_work_j < 0 and report["interface_work_j"] < 0
    assert report["physical_heat_j"] > 0
    a, ra = m.step(end, 1e-5, interface_shear_pa=-0.001)
    saved = InertialCapillaryState.from_dict(json.loads(json.dumps(end.to_dict())))
    b, rb = m.step(saved, 1e-5, interface_shear_pa=-0.001)
    assert a.to_dict() == b.to_dict() and ra == rb
    legacy = old.to_dict()
    assert "external_work_j" not in legacy
    assert InertialCapillaryState.from_dict(legacy).external_work_j == 0


def test_planar_profile_response_has_stable_slip_and_zero_slip_limits():
    h = torch.tensor([1e-18, 1e-8, 1e-3], dtype=torch.float64)
    eta, beta, c = resistance_response(h, 0.001, 0.0)
    torch.testing.assert_close(eta, torch.full_like(h, 1.5))
    torch.testing.assert_close(beta, h / (4 * 0.001))
    torch.testing.assert_close(c, 3 * 0.001 / h)
    eta, beta, c = resistance_response(h, 0.001, 1e3)
    assert bool(
        (
            torch.isfinite(eta) & torch.isfinite(beta) & torch.isfinite(c) & (beta > 0) & (c > 0)
        ).all()
    )


@pytest.mark.parametrize("scheme,minimum_ratio", [("backward", 1.8), ("roe", 3.8)])
def test_forced_relaxation_converges_to_the_analytic_continuous_solution(scheme, minimum_ratio):
    m = setup(drag=True, count=8, time_scheme=scheme)
    s = m.surface
    h, u0, horizon, g, tau = 3e-6, 0.01, 5e-6, 9.81, -0.001
    gamma = s.mu / (s.rho * h * (h / 3 + s.slip))
    eta = (h / 2 + s.slip) / (h / 3 + s.slip)
    equilibrium = (g + eta * tau / (s.rho * h)) / gamma
    target = equilibrium + (u0 - equilibrium) * math.exp(-gamma * horizon)
    errors = []
    for count in (4, 8, 16):
        state = m.initialize([h] * 8, [u0] * 8)
        for _ in range(count):
            state, _ = m.step(
                state, horizon / count, tangential_gravity_m_s2=g, interface_shear_pa=tau
            )
        errors.append(abs(state.face_velocity_m_s[0] - target))
    assert all(minimum_ratio < a / b < minimum_ratio + 0.5 for a, b in zip(errors, errors[1:]))


def test_nonuniform_driven_runner_passes_independent_audit_and_rejects_fake_work():
    from check_inertial_capillary_film import audit
    from run_inertial_capillary_film import run

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_inertial_capillary_driven.json"
        ).read_text()
    )
    report = run(config)
    audit(config, report)
    damaged = copy.deepcopy(report)
    damaged["result"]["steps"][0]["cell_external_work_j"][0] += 1e-6
    with pytest.raises(ValueError, match="cell external work"):
        audit(config, damaged)
    damaged = copy.deepcopy(report)
    damaged["result"]["steps"][0]["state"]["external_work_j"] += 1e-6
    with pytest.raises(ValueError, match="external work ledger"):
        audit(config, damaged)


def test_driven_pr_material_accounts_work_and_keeps_component_ledger():
    from check_inertial_capillary_film import audit
    from run_inertial_capillary_film import run

    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/surface_pr_isothermal_capillary_driven.json"
        ).read_text()
    )
    report = run(config)
    assert audit(config, report)["accepted"]
    assert report["final"]["mechanics"]["external_work_j"] != 0
    a = np.array([p["component_moles"] for p in report["initial"]["phases"]]).sum(0)
    b = np.array([p["component_moles"] for p in report["final"]["phases"]]).sum(0)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-24)
    assert all(
        p["temperature_k"] == config["model"]["temperature_k"] for p in report["final"]["phases"]
    )
    assert all(
        step["external_work_j"] == step["mechanics"]["external_work_j"]
        for step in report["result"]["steps"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("family", ["mechanical", "pr"])
def test_nonuniform_driven_mechanics_cpu_cuda_parity(family):
    if family == "pr":
        from test_surface_pr_isothermal_capillary import initial
        from test_surface_pr_isothermal_capillary import setup as pr_setup

        cpu, gpu = pr_setup(drag=True), pr_setup(drag=True, device="cuda")
        old = initial(cpu)
    else:
        cpu = setup(drag=True, count=8, time_scheme="roe")
        gpu = InertialCapillaryFilm.from_dict(cpu.to_dict(), device="cuda")
        old = cpu.initialize([3e-6 * (1 + 0.03 * math.cos(2 * math.pi * i / 8)) for i in range(8)])
    a, ra = cpu.step(old, 1e-5, tangential_gravity_m_s2=9.81, interface_shear_pa=-0.01)
    b, rb = gpu.step(old, 1e-5, tangential_gravity_m_s2=9.81, interface_shear_pa=-0.01)
    am, bm = (a.mechanics, b.mechanics) if family == "pr" else (a, b)
    np.testing.assert_allclose(am.thickness_m, bm.thickness_m, rtol=1e-12, atol=1e-22)
    np.testing.assert_allclose(am.face_velocity_m_s, bm.face_velocity_m_s, rtol=1e-10, atol=1e-14)
    assert rb["external_work_j"] == pytest.approx(ra["external_work_j"], rel=1e-10, abs=1e-22)
