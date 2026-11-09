"""Independent laminar/Colebrook mechanics and adiabatic energy conversion."""

import math

import pytest
from scipy.optimize import brentq
from test_surface_pr_active_multifluid_pipe import phase, setup

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe


def friction_model(velocity, *, viscosity=1e-5, roughness=0.0):
    base, v = setup(mode="periodic", count=2)
    raw = base.to_dict()
    raw["model"] = "pr_active_multifluid_pipe_v3"
    raw["wall_friction"] = {
        "model": "darcy_volume_wetted_v1",
        "viscosity_pa_s": {n: viscosity for n in base.names},
        "roughness_m": roughness,
    }
    model = PRActiveMultifluidPipe.from_dict(raw)
    cell = {"gas": phase(model.fluid, 1e6, 320.0, "vapor", v, velocity, x=(0.5, 0.5))}
    return model, model.initialize([cell, cell])


@pytest.mark.parametrize("velocity", [0.001, -0.001, 0.0])
def test_laminar_pipe_decay_matches_poiseuille_be_with_internal_heating(velocity):
    model, old = friction_model(velocity, viscosity=0.001)
    dt = 0.5 * model.maximum_step_s
    new, r = model.step(old, dt)
    a = dict(old.cells[0])["gas"]
    b = dict(new.cells[0])["gas"]
    mass = sum(n * c.molar_mass_kg_mol for n, c in zip(a.component_moles, model.fluid.components))
    rho = mass / a.volume_m3
    expected = velocity / (1 + dt * 32 * 0.001 / (rho * model.diameter**2))
    assert b.momentum_kg_m_s / mass == pytest.approx(expected, abs=1e-15)
    assert b.total_energy_j == a.total_energy_j
    assert b.temperature_k >= a.temperature_k - 1e-10
    assert r["friction"]["entropy_production_j_k"] >= -1e-7
    assert max(map(abs, r["conservation_residuals"])) < 1e-7
    assert len(new.cells[0]) == 1


@pytest.mark.parametrize("velocity", [30.0, -30.0])
def test_turbulent_factor_matches_independent_colebrook_root(velocity):
    model, old = friction_model(velocity, roughness=1e-5)
    dt = 0.5 * model.maximum_step_s
    new, r = model.step(old, dt)
    re = r["friction"]["reynolds_numbers"][0]
    assert re > 4000
    z = brentq(
        lambda z: z + 2 * math.log10(1e-5 / model.diameter / 3.7 + 2.51 * z / re),
        1,
        100,
        xtol=1e-13,
    )
    assert r["friction"]["darcy_factors"][0] == pytest.approx(1 / z**2, rel=1e-12)
    before = dict(old.cells[0])["gas"]
    after = dict(new.cells[0])["gas"]
    assert abs(after.momentum_kg_m_s) < abs(before.momentum_kg_m_s)
    assert after.total_energy_j == before.total_energy_j
    assert r["friction"]["dissipation_j"][0] > 0


def test_active_two_phase_friction_does_not_create_absent_slots_or_drop_species():
    model, _old = friction_model(2.0)
    p = model.fluid.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    v = model.volumes[0]
    cells = [
        {"gas": phase(model.fluid, p, 200.0, "vapor", v, 2.0)},
        {"liquid": phase(model.fluid, p, 200.0, "liquid", v, -1.0)},
    ]
    state = model.initialize(cells)
    new, r = model.step(state, 0.01 * model.maximum_step_s)
    assert all(len(c) == 2 for c in new.cells)
    assert max(map(abs, r["conservation_residuals"])) < 1e-7


def test_independent_wall_source_audit_rejects_missing_dissipation():
    from copy import deepcopy

    from check_pr_active_surface import friction_audit, pipe_audit

    from surface.pr_active_multifluid_pipe import PRActiveMultifluidState

    model, old = friction_model(30.0, roughness=1e-5)
    new, r = model.step(old, 0.2 * model.maximum_step_s)
    source = PRActiveMultifluidState.from_dict(r["chemical_state"])
    friction_audit(model, source, new, r["friction"])
    pipe_audit(model, old, new, r)
    bad = deepcopy(r["friction"])
    bad["dissipation_j"][0] += 1.0
    with pytest.raises(ValueError, match="audit differs"):
        friction_audit(model, source, new, bad)


def test_friction_standalone_runner_and_independent_auditor_roundtrip():
    import json

    from check_pr_active_multifluid_pipe import audit
    from run_pr_thermochemical_relaxation import run

    model, state = friction_model(30.0)
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [{"duration_s": 1.5 * model.maximum_step_s}],
            }
        )
    )
    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    assert len(report["intervals"][0]["steps"]) >= 2
