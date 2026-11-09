"""Analytic two-mass reference, independent local energy audit and CUDA parity."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_surface_pr_isothermal_capillary import setup

from surface.liquid import PropertyModelError
from surface.pr_capillary_gas_exchange import PRCapillaryGasExchange
from surface.pr_interphase_relaxation import PRRelaxationPhase


def gas_stocks(m, velocity):
    f = m.fluid
    q = f.homogeneous(1e5, 300.0, (1.0, 0.0), "vapor")
    volume = 1e-11
    amount = volume / float(q["molar_volume_m3_mol"].cpu())
    mass = amount * float(q["molar_mass_kg_mol"].cpu())
    energy = amount * float(q["molar_internal_energy_j_mol"].cpu()) + 0.5 * mass * velocity**2
    return tuple(
        PRRelaxationPhase((amount, 0.0), volume, 1e5, 300.0, mass * velocity, energy)
        for _ in range(16)
    )


@pytest.mark.parametrize("velocity", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("stiffness", [0.0, 1e-10, 1.0, 1e8])
def test_two_mass_reference_local_heat_and_global_stocks(velocity, stiffness):
    m = setup()
    old = m.initialize([3e-6] * 16, [0.01] * 16)
    gas = gas_stocks(m, velocity)
    source = PRCapillaryGasExchange(m)
    mg = gas[0].component_moles[0] * m.fluid.components[0].molar_mass_kg_mol
    ml = m.rho * m.mechanics.surface.cell_area * 3e-6
    reduced = mg * ml / (mg + ml)
    dt = 0.001
    k = stiffness * reduced / dt
    end, gn, r = source.apply(old, gas, dt, [k] * 16)
    J = reduced * stiffness / (1 + stiffness) * (velocity - 0.01)
    np.testing.assert_allclose(r["gas_to_liquid_impulse_kg_m_s"], J, atol=1e-30, rtol=2e-15)
    ul = 0.01 + J / ml
    ug = velocity - J / mg
    np.testing.assert_allclose(end.mechanics.face_velocity_m_s, ul, atol=1e-15)
    np.testing.assert_allclose([p.momentum_kg_m_s / mg for p in gn], ug, atol=1e-14)
    np.testing.assert_allclose(
        r["gas_physical_heat_j"], dt * k * (ug - ul) ** 2, atol=1e-28, rtol=1e-7
    )
    for a, b in zip(gas, gn):
        assert a.component_moles == b.component_moles
        assert a.volume_m3 == b.volume_m3
        internal_gain = (
            b.total_energy_j - a.total_energy_j - 0.5 * mg * (ug - velocity) * (ug + velocity)
        )
        assert internal_gain == pytest.approx(
            J * (ug - ul) + 0.5 * J**2 / mg, abs=8 * math.ulp(a.total_energy_j) + 1e-25
        )
        assert b.temperature_k >= a.temperature_k - 1e-10
    assert old.time_s == end.time_s
    assert not r["clock_advanced"]
    assert abs(r["momentum_residual_kg_m_s"]) < 1e-24
    assert abs(r["energy_residual_j"]) <= r["energy_roundoff_allowance_j"]
    assert m.inspect(old)["component_moles"] == m.inspect(end)["component_moles"]


def test_nonuniform_capillary_sequence_and_immutable_failed_source():
    from test_surface_pr_isothermal_capillary import initial

    m = setup()
    old = initial(m)
    intermediate, _ = m.step(old, 1e-4)
    gas = gas_stocks(m, 2.0)
    before = intermediate.to_dict()
    source = PRCapillaryGasExchange(m)
    end, _gn, r = source.apply(intermediate, gas, 1e-4, [1e-8] * 16)
    assert intermediate.to_dict() == before
    assert end.mechanics.thickness_m == intermediate.mechanics.thickness_m
    assert end.time_s == 1e-4
    assert abs(r["energy_residual_j"]) <= r["energy_roundoff_allowance_j"]
    with pytest.raises(ValueError, match="conductance"):
        source.apply(intermediate, gas, 1e-4, [-1.0] * 16)
    corrupt = (replace(gas[0], temperature_k=400.0), *gas[1:])
    with pytest.raises(PropertyModelError, match="caloric checkpoint"):
        source.apply(intermediate, corrupt, 1e-4, [1e-8] * 16)
    assert intermediate.to_dict() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_source_matches_cpu():
    results = []
    for device in ("cpu", "cuda"):
        m = setup(device=device)
        st = m.initialize([3e-6] * 16, [0.01] * 16)
        end, gas, r = PRCapillaryGasExchange(m).apply(st, gas_stocks(m, -2.0), 0.001, [1e-8] * 16)
        results.append((end, gas, r))
    a, b = results
    np.testing.assert_allclose(
        a[0].mechanics.face_velocity_m_s, b[0].mechanics.face_velocity_m_s, atol=1e-14
    )
    np.testing.assert_allclose(
        [p.temperature_k for p in a[1]], [p.temperature_k for p in b[1]], atol=1e-10
    )
    np.testing.assert_allclose(
        a[2]["gas_to_liquid_work_j"], b[2]["gas_to_liquid_work_j"], atol=1e-25
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_rejects_invalid_conductance_without_mutation(bad):
    m = setup()
    st = m.initialize([3e-6] * 16)
    gas = gas_stocks(m, 1.0)
    snapshot = st.to_dict()
    with pytest.raises(ValueError, match="conductance"):
        PRCapillaryGasExchange(m).apply(st, gas, 0.001, [bad] * 16)
    assert st.to_dict() == snapshot


def test_uniform_galilean_shift_changes_work_but_not_heat_or_impulse():
    m = setup()
    records = []
    for boost in (0.0, 0.5):
        st = m.initialize([3e-6] * 16, [0.01 + boost] * 16)
        _, _, r = PRCapillaryGasExchange(m).apply(
            st, gas_stocks(m, 2.0 + boost), 0.001, [1e-8] * 16
        )
        records.append(r)
    a, b = records
    for name in (
        "gas_to_liquid_impulse_kg_m_s",
        "gas_physical_heat_j",
        "gas_numerical_heat_j",
        "liquid_numerical_heat_j",
    ):
        np.testing.assert_allclose(a[name], b[name], atol=1e-27, rtol=1e-13)
    np.testing.assert_allclose(
        np.array(b["gas_to_liquid_work_j"]) - a["gas_to_liquid_work_j"],
        0.5 * np.array(a["gas_to_liquid_impulse_kg_m_s"]),
        atol=1e-27,
        rtol=1e-13,
    )
