"""Closed PR chamber feedback: thermodynamic integral, wave force and balances."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_surface_pr_isothermal_capillary import setup

from surface.pr_gas_loaded_capillary import (
    PRGasLoadedCapillaryFilm,
    PRGasLoadedCapillaryState,
    _GasPotential,
)
from surface.pr_homogeneous_uv import caloric_tv


def case(device="cpu"):
    return PRGasLoadedCapillaryFilm(setup(device=device), [30e-6] * 16, 300.0)


def initial(m, perturbation=0.001):
    return m.initialize(
        [3e-6 * (1 + perturbation * math.cos(2 * math.pi * i / 16)) for i in range(16)], (1.0, 0.0)
    )


def test_pressure_integral_matches_independent_pr_quadrature_and_limit():
    m = case()
    st = initial(m)
    f = m.film.fluid
    load = _GasPotential(m, st.gas)
    h = f.tensor(st.film.mechanics.thickness_m)
    dh = f.tensor([1e-6 * math.cos(2 * math.pi * i / 16) for i in range(16)])
    p, e = load.pressure_and_energy(h, dh)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    for i, gas in enumerate(st.gas):
        amount = sum(gas.component_moles)
        dv = -m.film.mechanics.surface.cell_area * float(dh[i].cpu())
        points = (gas.volume_m3 + dv * (nodes + 1) / 2) / amount
        pressure = []
        for v in points:
            q = caloric_tv(f, f.tensor(300.0), f.tensor(v), f.tensor([1.0, 0.0]))
            pressure.append(float(q["p"].cpu()))
        average = np.dot(weights, pressure) / 2
        assert float(p[i].cpu()) == pytest.approx(average, rel=2e-14)
        assert float(e[i].cpu()) == pytest.approx(-dv * average, rel=2e-14, abs=1e-30)
    p0, _ = load.pressure_and_energy(h, h * 0)
    np.testing.assert_allclose(p0.cpu(), [g.pressure_pa for g in st.gas], rtol=1e-12)
    jac = torch.func.jacfwd(lambda x: load.pressure_and_energy(h, x)[0])(h * 0)
    assert bool(torch.isfinite(jac).all().cpu())


def test_joint_pressure_feedback_energy_volume_components_and_restart():
    m = case()
    st = initial(m)
    end, r = m.step(st, 1e-7)
    assert end.time_s == 1e-7
    for a, b, h in zip(st.gas, end.gas, end.film.mechanics.thickness_m):
        assert a.component_moles == b.component_moles
        assert b.volume_m3 == pytest.approx(
            m.film.mechanics.surface.cell_area * (30e-6 - h), rel=1e-13
        )
        assert b.temperature_k == 300.0
    assert abs(r["energy_residual_j"]) <= r["energy_roundoff_allowance_j"]
    raw = json.loads(json.dumps(m.to_dict()))
    restored = PRGasLoadedCapillaryFilm.from_dict(raw)
    checkpoint = PRGasLoadedCapillaryState.from_dict(json.loads(json.dumps(end.to_dict())))
    a, ra = m.step(end, 1e-7)
    b, rb = restored.step(checkpoint, 1e-7)
    assert a.to_dict() == b.to_dict()
    assert ra == rb
    assert a.time_s == 2e-7


def test_uniform_chambers_are_stationary_and_invalid_geometry_fails():
    m = case()
    st = initial(m, 0.0)
    end, _r = m.step(st, 1e-6)
    np.testing.assert_allclose(
        end.film.mechanics.thickness_m, st.film.mechanics.thickness_m, atol=1e-22
    )
    np.testing.assert_allclose(end.film.mechanics.face_velocity_m_s, 0.0, atol=1e-14)
    with pytest.raises(ValueError, match="clearance"):
        m.initialize([31e-6] * 16, (1.0, 0.0))
    wrong = replace(st, gas=(replace(st.gas[0], volume_m3=st.gas[0].volume_m3 * 1.1), *st.gas[1:]))
    with pytest.raises(ValueError, match="geometry"):
        m.inspect(wrong)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu():
    values = []
    for device in ("cpu", "cuda"):
        m = case(device)
        end, r = m.step(initial(m), 1e-7)
        values.append((end, r))
    a, b = values
    np.testing.assert_allclose(
        a[0].film.mechanics.thickness_m, b[0].film.mechanics.thickness_m, atol=1e-19
    )
    np.testing.assert_allclose(
        a[0].film.mechanics.face_velocity_m_s, b[0].film.mechanics.face_velocity_m_s, atol=1e-11
    )
    np.testing.assert_allclose(
        [p.pressure_pa for p in a[0].gas], [p.pressure_pa for p in b[0].gas], atol=1e-6
    )


def test_independent_linear_gas_spring_wave_and_thermostat_entropy():
    m = case()
    count = 16
    H = 3e-6
    velocity = 0.01
    theta = 2 * math.pi / count
    st = m.initialize(
        [H] * count,
        (1.0, 0.0),
        face_velocity_m_s=[velocity * math.sin(theta * (i + 0.5)) for i in range(count)],
    )
    dt = 1e-7
    end, r = m.step(st, dt)
    s = m.film.mechanics.surface
    gas = st.gas[0]
    amount = sum(gas.component_moles)
    q = caloric_tv(
        m.film.fluid,
        m.film.fluid.tensor(300.0),
        m.film.fluid.tensor(gas.volume_m3 / amount),
        m.film.fluid.tensor([1.0, 0.0]),
    )
    modulus = -gas.volume_m3 / amount * float(q["dpdv"].cpu())
    k = 2 * math.sin(theta / 2) / s.spacing[0]
    omega2 = H / s.rho * (s.sigma * k**4 + modulus / (30e-6 - H) * k * k)
    ratio = omega2 * dt * dt / 4
    hmode = (
        2
        / count
        * sum((h - H) * math.cos(theta * i) for i, h in enumerate(end.film.mechanics.thickness_m))
    )
    umode = (
        2
        / count
        * sum(
            u * math.sin(theta * (i + 0.5))
            for i, u in enumerate(end.film.mechanics.face_velocity_m_s)
        )
    )
    assert hmode == pytest.approx(-H * k * dt * velocity / (1 + ratio), rel=2e-6)
    assert umode == pytest.approx(velocity * (1 - ratio) / (1 + ratio), rel=2e-6)
    entropy = []
    for state in (st, end):
        vals = []
        for p in state.gas:
            q = m.film.fluid.homogeneous(p.pressure_pa, p.temperature_k, (1.0, 0.0), "vapor")
            vals.append(sum(p.component_moles) * float(q["molar_entropy_j_mol_k"].cpu()))
        entropy.append(math.fsum(vals))
    # Independent caloric EOS entropy, rather than a reused work ledger.
    entropy_heat = -300.0 * (entropy[1] - entropy[0])
    assert r["gas_thermostat_heat_j"] == pytest.approx(entropy_heat, abs=1e-19)
    # A physical support, not fictitious gas axial momentum, takes reaction.
    dl = m.film.inspect(end.film)["momentum_kg_m_s"] - m.film.inspect(st.film)["momentum_kg_m_s"]
    assert dl + r["support_impulse_kg_m_s"] == pytest.approx(
        r["mechanics"]["wall_impulse_kg_m_s"], abs=1e-22
    )


def test_report_interval_exact_and_unrepresentable_clock_rejected():
    m = case()
    st = initial(m, 0.0)
    end, r = m.advance(st, 3.5e-7, initial_step_s=1e-7)
    assert end.time_s == 3.5e-7
    assert len(r["steps"]) == 4
    wrong = replace(st, film=replace(st.film, mechanics=replace(st.film.mechanics, time_s=1e16)))
    with pytest.raises(ValueError, match="clock"):
        m.advance(wrong, 1e-7, initial_step_s=1e-7)


def test_independent_trajectory_auditor_detects_corrupt_work_and_pressure():
    import copy
    from pathlib import Path

    from check_pr_gas_loaded_capillary import audit
    from run_pr_gas_loaded_capillary import run

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_gas_loaded_capillary.json"
        ).read_text()
    )
    config["duration_s"] = 2e-7
    report = run(config)
    assert audit(config, report)["checked_steps"] == 2
    for key, delta in (
        ("cell_normal_pressure_pa", 100.0),
        ("cell_normal_load_potential_change_j", 1e-12),
    ):
        bad = copy.deepcopy(report)
        bad["result"]["steps"][0]["mechanics"][key][0] += delta
        with pytest.raises(ValueError):
            audit(config, bad)
