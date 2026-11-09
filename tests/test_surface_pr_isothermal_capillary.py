"""Actual PR material, mean momentum and nonduplicated subcell kinetic energy."""

import json
import math

import numpy as np
import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_thin_film_dynamics import model

from surface.inertial_capillary_film import InertialCapillaryFilm
from surface.pr_isothermal_capillary_film import (
    PRIsothermalCapillaryFilm,
    PRIsothermalCapillaryState,
)


def setup(*, drag=False, device="cpu"):
    from surface.equilibrium_properties import EquilibriumFluid

    f = EquilibriumFluid.from_dict(fluid().to_dict(), device=device)
    pref, T, x = 1e6, 200.0, (0.0, 1.0)
    q = f.homogeneous(pref, T, x, "liquid")
    rho = float((q["molar_mass_kg_mol"] / q["molar_volume_m3_mol"]).cpu())
    raw = model(theta=0.0).to_dict()
    raw["density_kg_m3"] = rho
    raw["transverse_width_m"] = 0.001
    mechanics = InertialCapillaryFilm(raw, viscous_drag=drag, device=device)
    return PRIsothermalCapillaryFilm(f, mechanics, pref, T, x)


def initial(m):
    h = [3e-6 * (1 + 0.05 * math.cos(2 * math.pi * i / 16)) for i in range(16)]
    return m.initialize(h, [0.001 * math.sin(2 * math.pi * (i + 0.5) / 16) for i in range(16)])


def test_pr_mechanics_closes_components_impulse_and_energy_without_kinetic_double_count():
    m = setup()
    old = initial(m)
    end, r = m.step(old, 1e-4)
    a, b = m.inspect(old), m.inspect(end)
    np.testing.assert_allclose(a["component_moles"], b["component_moles"], rtol=1e-12, atol=1e-24)
    assert abs(r["joint_energy_residual_j"]) < 1e-15
    assert abs(r["momentum_residual_kg_m_s"]) < 1e-20
    assert b["velocity_variance_energy_j"] >= 0
    bulkK = sum(
        0.5
        * p.momentum_kg_m_s**2
        / sum(n * c.molar_mass_kg_mol for n, c in zip(p.component_moles, m.fluid.components))
        for p in end.phases
    )
    mechanical = m.mechanics.inspect(end.mechanics)
    assert bulkK + b["velocity_variance_energy_j"] == pytest.approx(
        mechanical["kinetic_energy_j"], rel=1e-12
    )
    assert all(p.temperature_k == 200.0 and p.pressure_pa == 1e6 for p in end.phases)
    assert b["estimated_density_response"] < 1e-4


def test_local_component_flux_and_volume_follow_the_joint_mechanical_step():
    m = setup()
    old = initial(m)
    dt = 1e-4
    new, r = m.step(old, dt)
    n0 = np.array([p.component_moles for p in old.phases])
    n1 = np.array([p.component_moles for p in new.phases])
    flux = np.array(r["component_flux_mol_s"])
    np.testing.assert_allclose(n1 - n0, -dt * (flux - np.roll(flux, 1, 0)), rtol=1e-8, atol=1e-21)


def test_dissipation_enters_a_single_thermostat_ledger_and_checkpoint_restarts():
    m = setup(drag=True)
    old = initial(m)
    end, _ = m.advance(old, 0.002, initial_step_s=0.001)
    stats = m.inspect(end)
    assert stats["thermostat_energy_j"] > 0
    assert stats["physical_entropy_production_j_k"] > 0
    rebuilt = PRIsothermalCapillaryFilm.from_dict(json.loads(json.dumps(m.to_dict())))
    checkpoint = PRIsothermalCapillaryState.from_dict(json.loads(json.dumps(end.to_dict())))
    a, ra = m.step(end, 0.001)
    b, rb = rebuilt.step(checkpoint, 0.001)
    assert a.to_dict() == b.to_dict()
    assert ra == rb


def test_wrong_density_and_small_response_domain_fail_before_any_correction():
    from surface.liquid import PropertyModelError

    m = setup()
    raw = m.to_dict()
    raw["mechanics"]["surface"]["density_kg_m3"] *= 1.1
    with pytest.raises(ValueError, match="density"):
        PRIsothermalCapillaryFilm.from_dict(raw)
    raw = m.to_dict()
    raw["maximum_density_response"] = 1e-18
    strict = PRIsothermalCapillaryFilm.from_dict(raw)
    with pytest.raises(PropertyModelError, match="response"):
        initial(strict)


def test_overdamped_material_bridge_rejects_inertial_state_instead_of_losing_energy():
    from test_surface_pr_film_transfer import film_setup

    from surface.pr_phase_volume_transport import transfer_film_increment

    p, s, _, _ = film_setup()
    m = setup()
    state = initial(m)
    with pytest.raises(TypeError, match="kinetic"):
        transfer_film_increment(
            p, s, m.mechanics, state.mechanics, 1e-4, liquid_slot="liquid", gas_slot="gas"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_coupled_pr_capillary_cuda_parity():
    a = setup()
    b = setup(device="cuda")
    x, _ = a.step(initial(a), 1e-4)
    y, ry = b.step(initial(b), 1e-4)
    for p, q in zip(x.phases, y.phases):
        np.testing.assert_allclose(p.component_moles, q.component_moles, atol=1e-22, rtol=1e-10)
        assert q.momentum_kg_m_s == pytest.approx(p.momentum_kg_m_s, abs=1e-20, rel=1e-9)
    assert abs(ry["joint_energy_residual_j"]) < 1e-15


def test_independent_scalar_pr_and_mechanical_flux_audit_rejects_bad_energy():
    from copy import deepcopy

    from check_inertial_capillary_film import audit
    from run_inertial_capillary_film import run

    m = setup()
    s = initial(m)
    c = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial_state": s.to_dict(),
                "duration_s": 0.0002,
                "initial_step_s": 0.0001,
            }
        )
    )
    r = json.loads(json.dumps(run(c)))
    assert audit(c, r)["checked_steps"] == 2
    bad = deepcopy(r)
    bad["result"]["steps"][0]["face_total_energy_flux_w"][0] += 1e-6
    with pytest.raises(ValueError, match="flux"):
        audit(c, bad)
    bad = deepcopy(r)
    bad["result"]["steps"][0]["state"]["phases"][0]["total_energy_j"] += 1e-12
    with pytest.raises(ValueError, match="calorics"):
        audit(c, bad)


def test_duplicate_kinetic_energy_is_detected_in_a_restored_material_state():
    from dataclasses import replace

    from surface.liquid import ConservationError

    m = setup()
    old = initial(m)
    kinetic = m.mechanics.inspect(old.mechanics)["kinetic_energy_j"]
    corrupt = replace(
        old,
        phases=tuple(
            replace(p, total_energy_j=p.total_energy_j + kinetic / len(old.phases))
            for p in old.phases
        ),
    )
    with pytest.raises(ConservationError, match="energy"):
        m.inspect(corrupt)
