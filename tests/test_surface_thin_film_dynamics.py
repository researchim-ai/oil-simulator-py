"""Resolved wall-film modes, work/energy, positivity and restart."""

import json
import math

import pytest
import torch

from surface.thin_film_dynamics import ThinFilmDynamics, ThinFilmState


def model(*, shape=(16,), theta=0.0, device="cpu", **kw):
    return ThinFilmDynamics(
        shape, (0.001,) * len(shape), 0.02, 0.001, 600.0, 1e-6, theta, 1e-8, device=device, **kw
    )


def mode(m, mean=3e-6, amplitude=1e-6):
    x = torch.arange(m.shape[0], dtype=torch.float64, device=m.device)
    return mean * (1 + amplitude * torch.cos(2 * math.pi * x / m.shape[0]))


def test_capillary_mode_recovers_discrete_linear_decay_and_energy_identity():
    m = model()
    h = mode(m)
    initial = m.initialize(h)
    dt = 0.05
    end, r = m.step(initial, dt)
    eig = (2 * math.sin(math.pi / m.shape[0]) / m.spacing[0]) ** 2
    H = 3e-6
    mobility = H**3 / (3 * m.mu) + m.slip * H**2 / m.mu
    rate = mobility * m.sigma * eig**2
    exact = (1 - dt * rate / 2) / (1 + dt * rate / 2)
    cosine = torch.cos(2 * math.pi * torch.arange(m.shape[0], dtype=h.dtype) / m.shape[0])
    ratio = float(((m.tensor(end.thickness_m) - H) * cosine).sum() / ((h - H) * cosine).sum())
    assert ratio == pytest.approx(exact, rel=2e-6)
    assert r["dissipation_j"] > 0
    assert r["free_energy_change_j"] < 0
    assert abs(r["energy_residual_j"]) < 1e-20
    assert m.inspect(end)["liquid_volume_m3"] == pytest.approx(
        m.inspect(initial)["liquid_volume_m3"], rel=1e-12
    )


def test_spinodal_wetting_mode_grows_with_the_analytical_dispersion_relation():
    m = model(theta=5.0)
    h = mode(m)
    initial = m.initialize(h)
    dt = 0.005
    end, r = m.step(initial, dt)
    k2 = (2 * math.sin(math.pi / m.shape[0]) / m.spacing[0]) ** 2
    H = 3e-6
    rr = m.adsorption / H
    phi2 = m.wetting_scale / m.adsorption**2 * (-3 * rr**4 + 6 * rr**7)
    mobility = H**3 / (3 * m.mu) + m.slip * H**2 / m.mu
    rate = -mobility * k2 * (m.sigma * k2 + phi2)
    assert rate > 0
    expected = (1 + dt * rate / 2) / (1 - dt * rate / 2)
    assert (max(end.thickness_m) - min(end.thickness_m)) / (
        float(h.max() - h.min())
    ) == pytest.approx(expected, rel=2e-6)
    assert r["free_energy_change_j"] < 0


def test_two_dimensional_film_work_restart_and_volume_without_hidden_floor():
    m = model(shape=(4, 8), theta=2.0)
    x = torch.arange(4, dtype=torch.float64)[:, None]
    y = torch.arange(8, dtype=torch.float64)[None, :]
    h = 3e-6 * (1 + 0.05 * torch.cos(2 * math.pi * x / 4) * torch.cos(2 * math.pi * y / 8))
    state = m.initialize(h)
    end, report = m.advance(
        state,
        0.002,
        initial_step_s=0.001,
        tangential_gravity_m_s2=(0.1, -0.1),
        interface_shear_pa=(0.01, -0.02),
    )
    assert end.time_s == pytest.approx(0.002, abs=1e-16)
    assert min(end.thickness_m) > 0
    assert (
        abs(
            m.inspect(end)["free_energy_j"]
            - m.inspect(state)["free_energy_j"]
            + end.dissipated_energy_j
            - end.external_work_j
        )
        < 1e-19
    )
    restored = ThinFilmDynamics.from_dict(json.loads(json.dumps(m.to_dict())))
    checkpoint = ThinFilmState.from_dict(json.loads(json.dumps(end.to_dict())))
    a, ra = m.step(end, 0.001)
    b, rb = restored.step(checkpoint, 0.001)
    assert a.to_dict() == b.to_dict()
    assert ra == rb
    assert len(report["steps"]) == 2


def test_adsorption_film_is_explicit_real_stock_and_not_clipped_at_material_length():
    m = model(theta=5.0)
    h = torch.full(m.shape, 0.9 * m.adsorption, dtype=torch.float64)
    old = m.initialize(h)
    end, r = m.step(old, 0.001)
    assert end.thickness_m == old.thickness_m
    assert r["dissipation_j"] == 0
    assert m.inspect(end)["liquid_volume_m3"] == pytest.approx(0.9 * m.adsorption * 0.001)
    with pytest.raises(ValueError, match="identity"):
        model(theta=4.0).inspect(end)


def test_invalid_geometry_and_nonphysical_inputs_rejected_atomically():
    m = model()
    with pytest.raises(ValueError):
        model(theta=45.0)
    with pytest.raises(ValueError):
        model(shape=(32, 32))
    with pytest.raises(ValueError):
        ThinFilmDynamics.from_dict({**m.to_dict(), "slip_length_m": -1.0})
    from surface.liquid import PropertyModelError

    old = m.initialize(mode(m))
    snapshot = old.to_dict()
    with pytest.raises(PropertyModelError):
        m.initialize(torch.zeros(m.shape))
    assert old.to_dict() == snapshot


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_resolved_wave_cuda_parity():
    m = model(theta=5.0)
    s = m.initialize(mode(m, amplitude=0.01))
    gpu = ThinFilmDynamics.from_dict(m.to_dict(), device="cuda")
    a, ra = m.step(s, 0.001)
    b, rb = gpu.step(s, 0.001)
    torch.testing.assert_close(
        m.tensor(a.thickness_m), m.tensor(b.thickness_m), atol=1e-18, rtol=1e-10
    )
    assert rb["dissipation_j"] == pytest.approx(ra["dissipation_j"], rel=1e-9)


def test_nonlinear_dewetting_is_independently_audited_and_tampering_rejected():
    from copy import deepcopy

    from check_thin_film_dynamics import audit
    from run_thin_film_dynamics import run

    m = model(theta=5.0)
    config = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial_thickness_m": mode(m, amplitude=0.2).tolist(),
                "duration_s": 1.0,
                "initial_step_s": 0.02,
            }
        )
    )
    report = json.loads(json.dumps(run(config)))
    result = audit(config, report)
    assert result["accepted"]
    assert result["checked_steps"] == 50
    assert min(report["final"]["thickness_m"]) < 1.1 * m.adsorption
    assert max(report["final"]["thickness_m"]) > 2 * max(report["initial"]["thickness_m"])
    bad = deepcopy(report)
    bad["result"]["steps"][0]["face_volume_flux_m2_s"][0][0] += 1e-9
    with pytest.raises(ValueError, match="face flux"):
        audit(config, bad)
    bad = deepcopy(report)
    bad["result"]["steps"][0]["state"]["dissipated_energy_j"] += 1e-9
    with pytest.raises(ValueError, match="heat ledger"):
        audit(config, bad)


def test_explicit_drive_cutback_covers_the_exact_interval_and_preserves_input():
    from surface.liquid import HydraulicConvergenceError

    m = model()
    s = m.initialize(mode(m))
    snapshot = s.to_dict()
    with pytest.raises(HydraulicConvergenceError, match="Courant"):
        m.step(s, 0.1, interface_shear_pa=(1.0,))
    assert s.to_dict() == snapshot
    end, r = m.advance(s, 0.01, initial_step_s=0.01, interface_shear_pa=(1.0,))
    assert end.time_s == pytest.approx(0.01, abs=1e-16)
    assert r["rejected_steps"] > 0
    assert abs(m.inspect(end)["liquid_volume_m3"] - m.inspect(s)["liquid_volume_m3"]) < 1e-21


@pytest.mark.parametrize("slip", [0.0, 1e-8, 1e-3])
def test_uniform_sheared_film_heats_even_when_surface_free_energy_does_not_change(slip):
    c = model().to_dict()
    c["slip_length_m"] = slip
    m = ThinFilmDynamics.from_dict(c)
    h = 3e-6
    tau = 0.01
    dt = 1e-5
    old = m.initialize([h] * 16)
    new, r = m.step(old, dt, interface_shear_pa=(tau,))
    expected = dt * math.prod(m.lengths) * m.width * tau**2 * (h + slip) / m.mu
    assert new.thickness_m == old.thickness_m
    assert r["free_energy_change_j"] == 0.0
    assert r["dissipation_j"] == pytest.approx(expected, rel=1e-12)
    assert r["interface_work_j"] == pytest.approx(expected, rel=1e-12)
    assert r["gravity_work_j"] == 0.0
    assert new.dissipated_energy_j > 0


def test_driven_stokes_power_matches_independent_profile_integration():
    from check_thin_film_dynamics import audit
    from run_thin_film_dynamics import run

    m = model(theta=0.0)
    c = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial_thickness_m": [3e-6] * 16,
                "duration_s": 1e-5,
                "initial_step_s": 1e-5,
                "tangential_gravity_m_s2": [0.1],
                "interface_shear_pa": [-0.01],
            }
        )
    )
    r = json.loads(json.dumps(run(c)))
    assert audit(c, r)["accepted"]
    step = r["result"]["steps"][0]
    assert step["dissipation_j"] > 0
    assert step["free_energy_change_j"] == 0.0
    assert step["interface_work_j"] + step["gravity_work_j"] == pytest.approx(
        step["dissipation_j"], rel=1e-12
    )


def test_nonfinite_derived_energy_cannot_be_accepted_from_finite_height():
    from surface.liquid import PropertyModelError

    m = model(theta=0.0)
    with pytest.raises(PropertyModelError, match="Nonfinite"):
        m.initialize([1e-100] * 16)


@pytest.mark.parametrize("kind", ["stokes", "inertial"])
def test_sub_ulp_interval_cannot_report_success_without_advancing_time(kind):
    from dataclasses import replace

    from surface.inertial_capillary_film import InertialCapillaryFilm

    m = model(theta=0.0)
    if kind == "inertial":
        m = InertialCapillaryFilm(m.to_dict(), viscous_drag=False)
    state = replace(m.initialize([3e-6] * 16), time_s=1e20)
    with pytest.raises(ValueError, match="clock"):
        m.step(state, 1e-4)
    with pytest.raises(ValueError, match="clock"):
        m.advance(state, 1e-4, initial_step_s=1e-4)
