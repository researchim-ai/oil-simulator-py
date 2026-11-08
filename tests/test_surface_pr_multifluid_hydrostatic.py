"""A declared equilibrium is checked thermodynamically before residual subtraction."""

import copy

import numpy as np
import pytest
from check_stratified_pr import Reference
from test_surface_pr_multifluid_gravity import inclined
from test_surface_pr_multifluid_pipe import initial

from surface.pr_multifluid_network import PRMultifluidNetwork


def hydrostatic(*, count=3, three=False, device="cpu"):
    model = inclined(count=count, three=three, device=device)
    state = initial(model, velocity=0.0)
    specs = []
    for cell, z in zip(state.cells, model.elevations):
        phases = []
        for k, p in enumerate(cell.phases):
            x = np.array(p.component_moles) / sum(p.component_moles)
            ref = Reference(model.fluid, np.flatnonzero(x > 0))
            root = model.phase_roots[k]
            mw = sum(x[j] * c.molar_mass_kg_mol for j, c in enumerate(model.fluid.components))
            base = ref.state(1e6, 350.0, x, root)
            gibbs = (base["h"] - 350.0 * base["s"]) / mw
            target = gibbs - model.gravity * z
            low, high = 9e5, 1.1e6
            for _ in range(55):
                pressure = (low + high) * 0.5
                q = ref.state(pressure, 350.0, x, root)
                if (q["h"] - 350.0 * q["s"]) / mw > target:
                    high = pressure
                else:
                    low = pressure
            phases.append(
                {
                    "pressure_pa": 0.5 * (low + high),
                    "temperature_k": 350.0,
                    "mole_fractions": x.tolist(),
                    "volume_m3": p.volume_m3,
                    "velocity_m_s": 0.0,
                }
            )
        specs.append(phases)
    raw = model.to_dict()
    raw.update(model="pr_multifluid_network_v3", hydrostatic_reference=specs)
    return PRMultifluidNetwork.from_dict(raw, device=device), specs


@pytest.mark.parametrize("three", [False, True])
def test_stratified_chemical_hydrostatics_remains_fixed_and_restarts(three):
    model, specs = hydrostatic(three=three)
    old = model.initialize(specs)
    result = model.advance(old, 1e-4, maximum_step_s=5e-5)
    new = result["state"]
    for before, after in zip(old.to_dict()["cells"], new["cells"]):
        for a, b in zip(before["phases"], after["phases"]):
            assert b["component_moles"] == pytest.approx(a["component_moles"], abs=1e-12)
            assert b["pressure_pa"] == pytest.approx(a["pressure_pa"], abs=1e-7)
            assert b["total_energy_j"] == pytest.approx(a["total_energy_j"], abs=1e-8)
            assert b["momentum_kg_m_s"] == pytest.approx(0.0, abs=1e-12)
    restored = PRMultifluidNetwork.from_dict(model.to_dict())
    assert restored.model_id == model.model_id
    assert max(map(abs, result["conservation_residuals"])) < 1e-7


def test_nonhydrostatic_or_source_incompatible_reference_is_rejected():
    model, _specs = hydrostatic()
    raw = copy.deepcopy(model.to_dict())
    raw["hydrostatic_reference"][1][0]["pressure_pa"] += 100.0
    with pytest.raises(ValueError, match="PR hydrostatics"):
        PRMultifluidNetwork.from_dict(raw)
    raw = copy.deepcopy(model.to_dict())
    raw["sources"][0]["pressure_conductance_m3_pa_s"] = ((0.0, 1e-10), (1e-10, 0.0))
    # Cell zero has equal phase pressure, but at higher elevations phase weights
    # differ. Put conductance into all local sources to expose that incompatibility.
    raw["sources"] = [raw["sources"][0]] * len(raw["sources"])
    with pytest.raises(ValueError, match="pressure exchange"):
        PRMultifluidNetwork.from_dict(raw)


def test_reference_does_not_freeze_pressure_perturbation():
    model, specs = hydrostatic()
    specs = copy.deepcopy(specs)
    specs[0][0]["pressure_pa"] += 100.0
    old = model.initialize(specs)
    end, r = model.step(old, 1e-6)
    assert abs(end.cells[1].phases[0].momentum_kg_m_s) > 1e-10
    assert end.cells[0].phases[0].pressure_pa != old.cells[0].phases[0].pressure_pa
    assert max(map(abs, r["conservation_residuals"])) < 1e-7


def test_independent_audit_of_rest_and_perturbation(monkeypatch):
    import json

    from check_pr_multifluid_pipe import audit
    from run_pr_multifluid_pipe import run

    model, specs = hydrostatic()
    specs = copy.deepcopy(specs)
    specs[0][0]["pressure_pa"] += 10.0
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"cells": specs},
                "schedule": [{"duration_s": 2e-7, "maximum_step_s": 2e-7}],
            }
        )
    )
    result = json.loads(json.dumps(run(config)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime physics used by independent audit")

    monkeypatch.setattr(PRMultifluidNetwork, "_network_operator", forbidden)
    monkeypatch.setattr(PRMultifluidNetwork, "_prepare_hydrostatic_reference", forbidden)
    assert audit(config, result)["accepted"]
