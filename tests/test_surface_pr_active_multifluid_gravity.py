"""Surveyed active phases conserve E+PE and retain verified PR hydrostatics."""

import json
import math
from dataclasses import asdict

import pytest
from scipy.optimize import brentq
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_pr_active_multifluid_pipe import phase

from surface.liquid import ConservationError
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState


def hydrostatic(*, reverse=False, reference=True, device="cpu"):
    f = fluid()
    if device != "cpu":
        f = type(f).from_dict(f.to_dict(), device=device)
    lengths = [0.1, 0.15, 0.2]
    faces = [0.0, 0.05, 0.1, 0.2]
    if reverse:
        faces = [-z for z in faces]
    centers = [(a + b) / 2 for a, b in zip(faces, faces[1:])]
    gravity = 9.80665
    t = 320.0
    x = (1.0, 0.0)
    mw = f.components[0].molar_mass_kg_mol

    def gibbs(p):
        q = f.homogeneous(p, t, x, "vapor")
        return float((q["molar_enthalpy_j_mol"] - t * q["molar_entropy_j_mol_k"]).cpu()) / mw

    g0 = gibbs(1e6)
    pressures = [
        brentq(lambda p, z=z: gibbs(p) + gravity * (z - centers[0]) - g0, 0.9e6, 1.1e6, xtol=1e-6)
        for z in centers
    ]
    diameter = 0.01
    volumes = [math.pi * diameter**2 / 4 * l for l in lengths]
    specs = [{"gas": phase(f, p, t, "vapor", v, x=x)} for p, v in zip(pressures, volumes)]
    model = PRActiveMultifluidPipe(
        f,
        [{"name": "gas", "root": "vapor"}],
        lengths,
        diameter,
        face_elevations_m=faces,
        gravity_m_s2=gravity,
        hydrostatic_reference=[{n: asdict(p) for n, p in c.items()} for c in specs]
        if reference
        else None,
    )
    return model, model.initialize(specs)


@pytest.mark.parametrize("reverse", [False, True])
def test_verified_isothermal_hydrostatics_preserved_without_spurious_flow(reverse):
    model, old = hydrostatic(reverse=reverse)
    new, r = model.step(old, 0.5 * model.maximum_step_s)
    assert new.cells == old.cells
    assert max(map(abs, r["conservation_residuals"])) < 1e-8
    restored = PRActiveMultifluidPipe.from_dict(json.loads(json.dumps(model.to_dict())))
    restored.inspect(PRActiveMultifluidState.from_dict(json.loads(json.dumps(new.to_dict()))))


def test_gravity_force_and_potential_energy_balance_without_reference():
    model, old = hydrostatic(reference=False)
    new, r = model.step(old, 0.05 * model.maximum_step_s)
    expected = -r["dt_s"] * sum(
        sum(
            p.component_moles[j] * c.molar_mass_kg_mol for j, c in enumerate(model.fluid.components)
        )
        * model.gravity
        * (b - a)
        / l
        for cell, a, b, l in zip(old.cells, model.elevations, model.elevations[1:], model.lengths)
        for _, p in cell
    )
    assert r["gravity_impulse_kg_m_s"] == pytest.approx(expected, rel=1e-13)
    assert model._totals(new)[-2] == pytest.approx(model._totals(old)[-2], abs=1e-7)
    assert max(map(abs, r["conservation_residuals"])) < 1e-7


def test_invalid_survey_and_nonphysical_reference_rejected():
    model, _old = hydrostatic()
    raw = model.to_dict()
    raw["face_elevations_m"] = [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="survey"):
        PRActiveMultifluidPipe.from_dict(raw)
    raw = model.to_dict()
    raw["hydrostatic_reference"][1]["gas"]["momentum_kg_m_s"] = 1e-5
    with pytest.raises((ValueError, ConservationError)):
        PRActiveMultifluidPipe.from_dict(raw)


def test_surveyed_reference_standalone_runner_and_independent_spatial_audit():
    from check_pr_active_multifluid_pipe import audit
    from run_pr_thermochemical_relaxation import run

    model, state = hydrostatic()
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
