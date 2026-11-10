"""A joint mechanical step must balance both kinetic and surface energy."""

import json
import math

import numpy as np
import pytest
import torch
from test_surface_thin_film_dynamics import model

from surface.inertial_capillary_film import InertialCapillaryFilm, InertialCapillaryState


def setup(*, drag=False, count=16, device="cpu", time_scheme="backward"):
    s = model(shape=(count,), theta=0.0).to_dict()
    return InertialCapillaryFilm(s, viscous_drag=drag, device=device, time_scheme=time_scheme)


def test_capillary_impulse_generates_motion_and_preserves_volume_momentum_energy():
    m = setup()
    h = [3e-6 * (1 + 0.05 * math.cos(2 * math.pi * i / 16)) for i in range(16)]
    old = m.initialize(h)
    new, r = m.step(old, 1e-4)
    a, b = m.inspect(old), m.inspect(new)
    assert b["kinetic_energy_j"] > 0.0
    assert b["potential_energy_j"] < a["potential_energy_j"]
    assert abs(b["liquid_volume_m3"] - a["liquid_volume_m3"]) < 1e-22
    assert abs(r["momentum_residual_kg_m_s"]) < 1e-20
    assert abs(r["energy_residual_j"]) < 1e-20
    assert r["physical_heat_j"] == 0.0
    assert r["numerical_heat_j"] > 0.0
    assert max(abs(v) for v in r["capillary_force_n"]) > 0.0
    assert abs(sum(r["capillary_force_n"])) < 1e-17


def test_uniform_wall_decay_has_exact_backward_euler_momentum_and_separate_heat():
    m = setup(drag=True)
    s = m.surface
    H = 3e-6
    u = 0.01
    dt = 0.001
    old = m.initialize([H] * 16, [u] * 16)
    end, r = m.step(old, dt)
    gamma = s.mu / (s.rho * H * (H / 3 + s.slip))
    expected = u / (1 + dt * gamma)
    np.testing.assert_allclose(end.face_velocity_m_s, expected, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(end.thickness_m, old.thickness_m, rtol=2e-14, atol=1e-23)
    assert r["physical_heat_j"] > 0.0
    assert r["numerical_heat_j"] > 0.0
    assert abs(r["energy_residual_j"]) < 1e-20
    assert r["wall_impulse_kg_m_s"] < 0.0


def test_small_amplitude_capillary_mode_matches_independent_two_by_two_step():
    m = setup()
    s = m.surface
    H = 3e-6
    eps = 1e-6
    dt = 1e-4
    n = 16
    theta = 2 * math.pi / n
    old = m.initialize([H * (1 + eps * math.cos(theta * i)) for i in range(n)])
    end, _ = m.step(old, dt)
    # Cell height is cosine; velocity lives at half-cell faces and is sine.
    k = 2 * math.sin(theta / 2) / s.spacing[0]
    A = H * k
    B = s.sigma / s.rho * k**3
    expected = np.linalg.solve([[1.0, dt * A], [-dt * B / 2, 1.0]], [H * eps, dt * B / 2 * H * eps])
    x = np.arange(n)
    height = np.sum((np.array(end.thickness_m) - H) * np.cos(theta * x)) * 2 / n
    speed = np.sum(np.array(end.face_velocity_m_s) * np.sin(theta * (x + 0.5))) * 2 / n
    assert height == pytest.approx(expected[0], rel=2e-6)
    assert speed == pytest.approx(expected[1], rel=2e-6)


def test_nyquist_height_is_not_an_uncoupled_checkerboard_mode():
    m = setup()
    h = [3e-6 * (1 + (0.01 if i % 2 else -0.01)) for i in range(16)]
    old = m.initialize(h)
    end, r = m.step(old, 1e-5)
    assert max(abs(v) for v in end.face_velocity_m_s) > 0.0
    assert max(end.thickness_m) - min(end.thickness_m) < max(h) - min(h)
    assert abs(r["energy_residual_j"]) < 1e-20


def test_restart_preserves_mechanical_and_thermal_state():
    m = setup(drag=True)
    state = m.initialize([3e-6 * (1 + 0.03 * math.cos(2 * math.pi * i / 16)) for i in range(16)])
    end, _ = m.advance(state, 0.002, initial_step_s=0.001)
    rebuilt = InertialCapillaryFilm.from_dict(json.loads(json.dumps(m.to_dict())))
    checkpoint = InertialCapillaryState.from_dict(json.loads(json.dumps(end.to_dict())))
    a, ra = m.step(end, 0.001)
    b, rb = rebuilt.step(checkpoint, 0.001)
    assert a.to_dict() == b.to_dict()
    assert ra == rb
    assert end.time_s == pytest.approx(0.002, abs=1e-16)


def test_wetting_gas_and_size_limits_are_explicit():
    with pytest.raises(ValueError, match="neutral"):
        InertialCapillaryFilm(model(theta=5.0).to_dict(), viscous_drag=True)
    with pytest.raises(ValueError, match="bound"):
        InertialCapillaryFilm(
            model(shape=(128,)).to_dict(), viscous_drag=False, maximum_unknowns=32
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_joint_mechanics_cuda_parity():
    m = setup()
    g = InertialCapillaryFilm.from_dict(m.to_dict(), device="cuda")
    old = m.initialize([3e-6 * (1 + 0.05 * math.cos(2 * math.pi * i / 16)) for i in range(16)])
    a, _ = m.step(old, 1e-4)
    b, rb = g.step(old, 1e-4)
    torch.testing.assert_close(
        m.surface.tensor(a.thickness_m), m.surface.tensor(b.thickness_m), atol=1e-18, rtol=1e-10
    )
    torch.testing.assert_close(
        m.surface.tensor(a.face_velocity_m_s),
        m.surface.tensor(b.face_velocity_m_s),
        atol=1e-12,
        rtol=1e-9,
    )
    assert abs(rb["energy_residual_j"]) < 1e-20


def test_roe_time_velocity_has_no_artificial_damping_in_the_nonlinear_inviscid_mode():
    m = setup(time_scheme="auto")
    old = m.initialize(
        [3e-6 * (1 + 0.05 * math.cos(2 * math.pi * i / 16)) for i in range(16)],
        [0.001 * math.sin(2 * math.pi * (i + 0.5) / 16) for i in range(16)],
    )
    _, r = m.step(old, 1e-4)
    assert r["time_scheme"] == "roe"
    assert r["numerical_heat_j"] == 0.0
    assert r["physical_heat_j"] == 0.0
    assert abs(r["energy_residual_j"]) < 1e-20
    assert abs(r["momentum_residual_kg_m_s"]) < 1e-20


def test_auto_time_scheme_uses_monotone_decay_for_unresolved_wall_relaxation():
    m = setup(drag=True, time_scheme="auto")
    old = m.initialize([3e-6] * 16, [0.01] * 16)
    end, r = m.step(old, 0.001)
    assert r["time_scheme"] == "backward"
    assert min(end.face_velocity_m_s) > 0.0
    assert r["numerical_heat_j"] > 0.0


def test_time_centered_wave_matches_crank_nicolson_without_fitting_a_frequency():
    m = setup(time_scheme="roe")
    s = m.surface
    H = 3e-6
    eps = 1e-6
    dt = 1e-4
    n = 16
    theta = 2 * math.pi / n
    k = 2 * math.sin(theta / 2) / s.spacing[0]
    old = m.initialize([H * (1 + eps * math.cos(theta * i)) for i in range(n)])
    end, r = m.step(old, dt)
    A = H * k
    B = s.sigma / s.rho * k**3
    expected = np.linalg.solve(
        [[1.0, dt * A / 2], [-dt * B / 2, 1.0]], [H * eps, dt * B / 2 * H * eps]
    )
    x = np.arange(n)
    height = np.sum((np.array(end.thickness_m) - H) * np.cos(theta * x)) * 2 / n
    speed = np.sum(np.array(end.face_velocity_m_s) * np.sin(theta * (x + 0.5))) * 2 / n
    assert height == pytest.approx(expected[0], rel=2e-6)
    assert speed == pytest.approx(expected[1], rel=2e-6)
    assert r["numerical_heat_j"] == 0.0


def test_local_energy_and_momentum_ports_are_verified_independently():
    from copy import deepcopy

    from check_inertial_capillary_film import audit
    from run_inertial_capillary_film import run

    m = setup(time_scheme="auto")
    c = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial_thickness_m": [
                    3e-6 * (1 + 0.05 * math.cos(2 * math.pi * i / 16)) for i in range(16)
                ],
                "duration_s": 0.0004,
                "initial_step_s": 0.0001,
            }
        )
    )
    r = json.loads(json.dumps(run(c)))
    assert audit(c, r)["checked_steps"] == 4
    bad = deepcopy(r)
    bad["result"]["steps"][0]["capillary_force_n"][0] += 1e-6
    with pytest.raises(ValueError, match="force"):
        audit(c, bad)
    assert max(abs(v) for v in r["result"]["steps"][0]["cell_energy_residual_j"]) < 1e-20
    assert max(abs(v) for v in r["result"]["steps"][0]["cell_momentum_residual_kg_m_s"]) < 1e-20


def test_normal_gravity_potential_keeps_the_same_closed_mechanical_ledger():
    raw = setup(time_scheme="roe").to_dict()
    raw["surface"]["normal_gravity_m_s2"] = 9.80665
    m = InertialCapillaryFilm.from_dict(raw)
    old = m.initialize([3e-6 * (1 + 0.05 * math.cos(2 * math.pi * i / 16)) for i in range(16)])
    _, r = m.step(old, 1e-4)
    assert abs(r["energy_residual_j"]) < 1e-20
    assert abs(r["momentum_residual_kg_m_s"]) < 1e-20


def _wave_phase_error(count, dt, *, continuum):
    m = setup(count=count, time_scheme="roe")
    s = m.surface
    H = 3e-6
    theta = 2 * math.pi / count
    k = 2 * math.sin(theta / 2) / s.spacing[0]
    old = m.initialize([H * (1 + 1e-4 * math.cos(theta * i)) for i in range(count)])
    duration = 0.002 if continuum else 0.008
    end, _ = m.advance(old, duration, initial_step_s=dt)
    height = sum((h - H) * math.cos(theta * i) for i, h in enumerate(end.thickness_m)) * 2 / count
    speed = (
        sum(u * math.sin(theta * (i + 0.5)) for i, u in enumerate(end.face_velocity_m_s))
        * 2
        / count
    )
    omega = (s.sigma * H / s.rho) ** 0.5 * k * k
    observed = math.atan2(speed * H * k / omega, height)
    exact = (
        (s.sigma * H / s.rho) ** 0.5 * (2 * math.pi / s.lengths[0]) ** 2 if continuum else omega
    ) * duration
    return abs(math.atan2(math.sin(observed - exact), math.cos(observed - exact)))


def test_roe_wave_phase_converges_at_second_order_in_time():
    coarse = _wave_phase_error(16, 2e-4, continuum=False)
    fine = _wave_phase_error(16, 1e-4, continuum=False)
    assert 3.7 < coarse / fine < 4.3


def test_capillary_dispersion_converges_to_continuum_at_second_order_in_space():
    coarse = _wave_phase_error(16, 2e-5, continuum=True)
    fine = _wave_phase_error(32, 2e-5, continuum=True)
    assert 3.7 < coarse / fine < 4.3


def test_device_parity_uses_height_units_instead_of_a_generic_absolute_tolerance():
    from copy import deepcopy

    from reference_capillary_device_parity import audit
    from run_inertial_capillary_film import run

    m = setup(time_scheme="auto")
    config = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial_thickness_m": [3e-6] * 16,
                "duration_s": 1e-4,
                "initial_step_s": 1e-4,
            }
        )
    )
    a = json.loads(json.dumps(run(config)))
    b = deepcopy(a)
    b["device"] = "cuda"
    assert audit(a, b)["accepted"]
    b["result"]["steps"][0]["state"]["thickness_m"][0] += 1e-12
    with pytest.raises(ValueError, match="parity"):
        audit(a, b)


def test_finite_coordinates_cannot_hide_a_nan_in_derived_energy():
    from surface.liquid import PropertyModelError

    raw = setup(time_scheme="roe").to_dict()
    raw["maximum_velocity_m_s"] = 1e200
    m = InertialCapillaryFilm.from_dict(raw)
    # Finite primitive coordinates can still overflow the kinetic energy.
    # Do not rely on the former 0*inf bug in the inactive wall potential.
    with pytest.raises(PropertyModelError, match="Nonfinite"):
        m.initialize([3e-6] * 16, [1e190] * 16)
