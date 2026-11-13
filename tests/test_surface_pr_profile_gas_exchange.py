"""Profile work, stationary-wall impulse, stiff limits and driven restart."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from reference_pr_capillary_gas_exchange import audit
from test_surface_pr_capillary_gas_exchange import gas_stocks
from test_surface_pr_isothermal_capillary import setup

from surface.pr_capillary_gas_exchange import PRCapillaryGasExchange
from surface.pr_interphase_relaxation import PRRelaxationPhase
from surface.pr_isothermal_capillary_film import PRIsothermalCapillaryState


@pytest.mark.parametrize("velocity", [-2.0, 0.01, 2.0])
@pytest.mark.parametrize("k", [0.0, 1e-12, 1e-8, 1e-6])
def test_profile_exchange_matches_independent_two_by_two_system_and_full_audit(velocity, k):
    source = PRCapillaryGasExchange(setup(drag=True), interface_model="stokes_profile")
    m, s = source.film, source.film.mechanics.surface
    old = m.initialize([3e-6] * 16, [0.01] * 16)
    gas = gas_stocks(m, velocity)
    dt = 0.001
    end, new_gas, report = source.apply(old, gas, dt, [k] * 16)
    assert audit(source, old, gas, end, new_gas, dt, [k] * 16, report)["accepted"]
    h = 3e-6
    area = s.cell_area
    a = h**3 / (3 * s.mu) + s.slip * h * h / s.mu
    b = h * h / (2 * s.mu) + s.slip * h / s.mu
    eta = h * b / a
    beta = (h + s.slip) / s.mu - b * b / a
    mg = sum(n * c.molar_mass_kg_mol for n, c in zip(gas[0].component_moles, m.fluid.components))
    ml = s.rho * area * h
    effective = k / (1 + k * beta / area)
    matrix = np.array(
        [[mg / dt + effective, -eta * effective], [-eta * effective, ml / dt + eta**2 * effective]]
    )
    ug, ul = np.linalg.solve(matrix, [mg / dt * velocity, ml / dt * 0.01])
    np.testing.assert_allclose(end.mechanics.face_velocity_m_s, ul, rtol=2e-11, atol=1e-14)
    np.testing.assert_allclose(
        [p.momentum_kg_m_s / mg for p in new_gas], ug, rtol=2e-11, atol=1e-14
    )
    assert all(q >= 0 for q in report["liquid_profile_heat_j"])
    if k and velocity == 0.01:
        assert all(j < 0 for j in report["gas_to_liquid_impulse_kg_m_s"])


@pytest.mark.parametrize("mode", ["plug", "stokes_profile"])
def test_large_finite_conductance_reaches_its_limit_without_overflow_or_nonfinite_report(mode):
    source = PRCapillaryGasExchange(setup(drag=True), interface_model=mode)
    m = source.film
    old = m.initialize([3e-6] * 16, [0.01] * 16)
    gas = gas_stocks(m, 2.0)
    a, ga, _ra = source.apply(old, gas, 0.001, [1e8] * 16)
    b, gb, rb = source.apply(old, gas, 0.001, [1e308] * 16)
    np.testing.assert_allclose(
        a.mechanics.face_velocity_m_s, b.mechanics.face_velocity_m_s, rtol=1e-11, atol=1e-14
    )
    np.testing.assert_allclose(
        [p.momentum_kg_m_s for p in ga], [p.momentum_kg_m_s for p in gb], rtol=1e-9, atol=1e-24
    )
    assert audit(source, old, gas, b, gb, 0.001, [1e308] * 16, rb)["accepted"]
    json.dumps(rb, allow_nan=False)


def test_source_preserves_signed_external_work_after_driving_stop_and_checkpoint():
    m = setup(drag=True)
    old = m.initialize([3e-6] * 16, [0.01] * 16)
    driven, _ = m.step(old, 1e-5, interface_shear_pa=-0.001)
    assert driven.mechanics.external_work_j < 0
    gas = gas_stocks(m, -2.0)
    source = PRCapillaryGasExchange(m, interface_model="stokes_profile")
    checkpoint = copy.deepcopy(driven.to_dict())
    end, new_gas, r = source.apply(driven, gas, 0.001, [1e-8] * 16)
    assert audit(source, driven, gas, end, new_gas, 0.001, [1e-8] * 16, r)["accepted"]
    assert driven.to_dict() == checkpoint
    restored = PRCapillaryGasExchange.from_dict(json.loads(json.dumps(source.to_dict())))
    fs = PRIsothermalCapillaryState.from_dict(json.loads(json.dumps(end.to_dict())))
    gs = tuple(PRRelaxationPhase.from_dict(json.loads(json.dumps(asdict(p)))) for p in new_gas)
    a, ag, ar = source.apply(end, new_gas, 0.001, [1e-8] * 16)
    b, bg, br = restored.apply(fs, gs, 0.001, [1e-8] * 16)
    assert a.to_dict() == b.to_dict() and ag == bg and ar == br
    stopped, sg, sr = restored.apply(fs, gs, 0.001, [0.0] * 16)
    assert stopped.mechanics == fs.mechanics
    assert sr["gas_to_liquid_work_j"] == [0.0] * 16
    assert audit(restored, fs, gs, stopped, sg, 0.001, [0.0] * 16, sr)["accepted"]


def test_profile_auditor_rejects_wrong_surface_work_missing_wall_reaction_and_fake_heat():
    source = PRCapillaryGasExchange(setup(drag=True), interface_model="stokes_profile")
    m = source.film
    old = m.initialize([3e-6] * 16, [0.01] * 16)
    gas = gas_stocks(m, 2.0)
    end, new_gas, r = source.apply(old, gas, 0.001, [1e-8] * 16)
    for key in (
        "stationary_wall_impulse_kg_m_s",
        "liquid_profile_heat_j",
        "interface_velocity_m_s",
    ):
        bad = copy.deepcopy(r)
        bad[key][0] += 1e-6
        with pytest.raises(ValueError, match="audit differs"):
            audit(source, old, gas, end, new_gas, 0.001, [1e-8] * 16, bad)


def test_profile_choice_and_immutable_source_identity_are_explicit():
    with pytest.raises(ValueError, match="viscous"):
        PRCapillaryGasExchange(setup(), interface_model="stokes_profile")
    with pytest.raises(ValueError, match="interface model"):
        PRCapillaryGasExchange(setup(), interface_model="automatic_fit")
    source = PRCapillaryGasExchange(setup(drag=True), interface_model="stokes_profile")
    old = source.film.initialize([3e-6] * 16)
    gas = gas_stocks(source.film, 2.0)
    source.interface_model = "plug"
    with pytest.raises(ValueError, match="identity changed"):
        source.apply(old, gas, 0.001, [1e-8] * 16)


@pytest.mark.parametrize("state", [{"external_work_j": -2e-17}, {"component_moles": [6e-11, 0.0]}])
def test_small_physical_stocks_cannot_pass_device_parity_after_being_zeroed(state):
    from reference_capillary_device_parity import compare_checkpoints

    zero = {k: [0.0] * len(v) if isinstance(v, list) else 0.0 for k, v in state.items()}
    with pytest.raises(ValueError, match="device parity failed"):
        compare_checkpoints([("state", state, zero)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_source_keeps_cpu_checkpoint_component_stocks_and_volumes_exact():
    from check_pr_capillary_gas_exchange import audit as request_audit
    from run_pr_capillary_gas_exchange import run

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_profile_gas_exchange.json"
        ).read_text()
    )
    report = run(config, device="cuda")
    assert request_audit(config, report)["accepted"]
    for old, new in zip(
        config["initial_film"]["phases"], report["final_film"]["phases"], strict=True
    ):
        assert tuple(old["component_moles"]) == tuple(new["component_moles"])
        assert old["volume_m3"] == new["volume_m3"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_profile_exchange_cpu_cuda_parity_after_external_driving():
    records = []
    for device in ("cpu", "cuda"):
        m = setup(drag=True, device=device)
        old = m.initialize([3e-6] * 16, [0.01] * 16)
        driven, _ = m.step(old, 1e-5, tangential_gravity_m_s2=9.81, interface_shear_pa=-0.001)
        source = PRCapillaryGasExchange(m, interface_model="stokes_profile")
        gas = gas_stocks(m, -2.0)
        end, new_gas, r = source.apply(driven, gas, 0.001, [1e-8] * 16)
        assert audit(source, driven, gas, end, new_gas, 0.001, [1e-8] * 16, r)["accepted"]
        records.append((end, new_gas, r))
    a, b = records
    np.testing.assert_allclose(
        a[0].mechanics.face_velocity_m_s, b[0].mechanics.face_velocity_m_s, rtol=1e-10, atol=1e-14
    )
    for key in ("gas_to_liquid_work_j", "stationary_wall_impulse_kg_m_s", "liquid_profile_heat_j"):
        np.testing.assert_allclose(a[2][key], b[2][key], rtol=1e-10, atol=1e-24)
