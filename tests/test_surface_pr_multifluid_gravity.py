"""Gravity is closed through numerical mass flux, not an unrelated velocity."""

import copy
import math

import numpy as np
import pytest
from test_surface_pr_multifluid_pipe import initial, setup

from surface.pr_multifluid_network import PRMultifluidNetwork
from surface.pr_multifluid_pipe import PRMultifluidPipeState


def inclined(
    *, count=1, slope=1.0, offset=0.0, gravity=9.80665, three=False, device="cpu", dense=128
):
    source = setup(three=three).sources[0]
    if device != "cpu":
        source = type(source).from_dict(source.to_dict(), device=device)
    names = ["A"] + [f"p{i}" for i in range(count)] + ["B"]
    area, length = 0.01, 0.2 / count
    direction = (math.sqrt(1 - slope * slope), 0.0, slope)
    elevations = (
        [offset]
        + [offset + (i + 0.5) * length * slope for i in range(count)]
        + [offset + 0.2 * slope]
    )
    faces = [
        {"source": a, "target": b, "area_m2": area, "direction": direction}
        for a, b in zip(names, names[1:])
    ]
    return PRMultifluidNetwork(
        [source] * len(names),
        [0.001] + [area * length] * count + [0.001],
        names,
        faces,
        ["A", "B"],
        elevations_m=elevations,
        gravity_m_s2=gravity,
        dense_limit=dense,
    )


def specific_mass(model, phase):
    return math.fsum(
        n * c.molar_mass_kg_mol for n, c in zip(phase.component_moles, model.fluid.components)
    )


@pytest.mark.parametrize("three", [False, True])
@pytest.mark.parametrize("slope", [-1.0, 0.6, 1.0])
def test_gravity_drives_downhill_and_preserves_total_energy(three, slope):
    model = inclined(three=three, slope=slope)
    state = initial(model, velocity=0.0)
    dt = 1e-6
    end, report = model.step(state, dt)
    assert all(p.momentum_kg_m_s * slope < 0 for p in end.cells[1].phases)
    assert all(p.momentum_kg_m_s == 0 for i in model.supported_cells for p in end.cells[i].phases)
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    assert model._stored_energy(end) == pytest.approx(model._stored_energy(state), abs=1e-7)
    restore = PRMultifluidNetwork.from_dict(model.to_dict())
    assert restore.model_id == model.model_id
    restore.inspect(PRMultifluidPipeState.from_dict(end.to_dict()))


def test_gravity_acceleration_small_time_limit_and_zero_gravity_rest():
    model = inclined(count=3)
    old = initial(model, velocity=0.0)
    dt = 1e-8
    new, _ = model.transport_step(old, dt)
    for p in new.cells[2].phases:
        acceleration = p.momentum_kg_m_s / specific_mass(model, p) / dt
        assert acceleration == pytest.approx(-model.gravity, rel=0.002)
    model = inclined(gravity=0.0)
    old = initial(model, velocity=0.0)
    new, _ = model.transport_step(old, 1e-6)
    assert all(abs(p.momentum_kg_m_s) < 1e-12 for c in new.cells for p in c.phases)


def test_height_origin_and_rotation_leave_thermodynamics_unchanged():
    results = []
    for model in [inclined(slope=0.6), inclined(slope=0.6, offset=10000.0)]:
        old = initial(model, velocity=0.0)
        end, report = model.step(old, 1e-6)
        results.append(
            np.array(
                [
                    [p.pressure_pa, p.temperature_k, p.momentum_kg_m_s]
                    for c in end.cells
                    for p in c.phases
                ]
            )
        )
        assert max(map(abs, report["conservation_residuals"])) < 1e-7
    assert np.max(abs(results[0] - results[1])) < 1e-6


def test_invalid_geometries_fail_before_solve_and_legacy_restart_stays_legacy():
    model = inclined()
    raw = copy.deepcopy(model.to_dict())
    raw["elevations_m"] = (0.0, 0.09, 0.2)
    with pytest.raises(ValueError, match="elevations disagree"):
        PRMultifluidNetwork.from_dict(raw)
    raw = copy.deepcopy(model.to_dict())
    raw["model"] = "pr_multifluid_network_v1"
    with pytest.raises(ValueError, match="v2"):
        PRMultifluidNetwork.from_dict(raw)
    raw = copy.deepcopy(model.to_dict())
    del raw["elevations_m"]
    with pytest.raises(ValueError, match="requires elevations"):
        PRMultifluidNetwork.from_dict(raw)


def test_independent_gravity_mass_flux_work_vector_load_and_sparse_solution():
    from reference_pr_multifluid_network import reconstruct

    outputs = []
    for dense in [128, 1]:
        model = inclined(count=3, slope=0.6, dense=dense)
        rest = initial(model, velocity=0.0)
        specs = []
        for i, c in enumerate(rest.cells):
            specs.append(
                [
                    {
                        "pressure_pa": p.pressure_pa + (500 if i == 0 else 0),
                        "temperature_k": p.temperature_k,
                        "volume_m3": p.volume_m3,
                        "mole_fractions": [n / sum(p.component_moles) for n in p.component_moles],
                        "velocity_m_s": 0.0 if i in model.supported_cells else 0.03,
                    }
                    for p in c.phases
                ]
            )
        state = model.initialize(specs)
        dt = 1e-6
        end, report = model.transport_step(state, dt)
        old = reconstruct(model.to_dict(), state.to_dict(), model.fluid)
        new = reconstruct(model.to_dict(), end.to_dict(), model.fluid)
        assert np.max(abs(dt * new["force"] - report["wall_impulse_kg_m_s"])) < 1e-10
        for i, (a, b) in enumerate(zip(old["packets"], new["packets"])):
            residual = (b["conserved"] - a["conserved"]) * model.volumes[i] + dt * new[
                "phase_rate"
            ][i]
            assert np.max(abs(residual[:, :-2])) < 1e-9
            assert np.max(abs(residual[:, -1])) < 1e-7
            if i not in model.supported_cells:
                assert np.max(abs(residual[:, -2])) < 1e-9
            alpha = b["alpha"][1:] - a["alpha"][1:] + dt * new["alpha_rate"][i]
            assert np.max(abs(alpha)) < 1e-10
        outputs.append(np.array([p.pressure_pa for c in end.cells for p in c.phases]))
    assert np.max(abs(outputs[0] - outputs[1])) < 1e-5


def test_constant_density_hydrostatic_pressure_and_weight_cancel_at_finite_ports():
    from check_stratified_pr import Reference
    from reference_pr_multifluid_network import reconstruct

    model = inclined(count=3)
    rest = initial(model, velocity=0.0)
    specs = []
    for cell, z in zip(rest.cells, model.elevations):
        phases = []
        for k, p in enumerate(cell.phases):
            x = np.array(p.component_moles) / sum(p.component_moles)
            ref = Reference(model.fluid, np.flatnonzero(x > 0))
            root = model.phase_roots[k]
            centre = ref.state(1e6, 350.0, x, root)
            pressure = 1e6 - centre["rho"] * model.gravity * z
            low, high = 345.0, 355.0
            for _ in range(55):
                temperature = 0.5 * (low + high)
                if ref.state(pressure, temperature, x, root)["vm"] > centre["vm"]:
                    high = temperature
                else:
                    low = temperature
            phases.append(
                {
                    "pressure_pa": pressure,
                    "temperature_k": 0.5 * (low + high),
                    "mole_fractions": x.tolist(),
                    "volume_m3": p.volume_m3,
                    "velocity_m_s": 0.0,
                }
            )
        specs.append(phases)
    state = model.initialize(specs)
    spatial = reconstruct(model.to_dict(), state.to_dict(), model.fluid)
    for i in range(1, len(model.sources) - 1):
        assert np.max(abs(spatial["phase_rate"][i, :, -2])) < 1e-8
    # This checks pressure/weight balance, not disappearance of Rusanov thermal
    # diffusion in a stratified equilibrium; the latter remains an open closure.


def test_network_report_audit_checks_balances_and_rejects_tampering(monkeypatch):
    import json
    from pathlib import Path

    from check_pr_multifluid_pipe import audit
    from run_pr_multifluid_pipe import run

    config = json.loads(
        (Path(__file__).parents[1] / "configs/surface_pr_multifluid_gravity.json").read_text()
    )
    result = json.loads(json.dumps(run(config)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime physics is not an independent reference")

    monkeypatch.setattr(PRMultifluidNetwork, "_network_operator", forbidden)
    monkeypatch.setattr(PRMultifluidNetwork, "inspect", forbidden)
    monkeypatch.setattr(PRMultifluidNetwork, "ledger", forbidden)
    assert audit(config, result)["accepted"]
    changed = copy.deepcopy(result)
    changed["intervals"][0]["steps"][0]["transport"]["wall_impulse_kg_m_s"][2] += 0.01
    with pytest.raises(ValueError):
        audit(config, changed)


def test_gravity_composes_with_pressure_heat_drag_and_wall_sources():
    import json

    from check_pr_multifluid_pipe import audit
    from run_pr_multifluid_pipe import run

    raw = inclined(slope=0.6).to_dict()
    source = setup(exchange=True).sources[0].to_dict()
    source["wall_drag_kg_s"] = (0.01, 0.02)
    raw["sources"] = [source] * 3
    model = PRMultifluidNetwork.from_dict(raw)
    rest = initial(model, velocity=0.0)
    specs = []
    for i, c in enumerate(rest.cells):
        specs.append(
            [
                {
                    "pressure_pa": p.pressure_pa + (1000 if k == 0 else -1000),
                    "temperature_k": p.temperature_k + k,
                    "volume_m3": p.volume_m3,
                    "mole_fractions": [n / sum(p.component_moles) for n in p.component_moles],
                    "velocity_m_s": 0.0
                    if i in model.supported_cells
                    else (0.03 if k == 0 else -0.01),
                }
                for k, p in enumerate(c.phases)
            ]
        )
    state = model.initialize(specs)
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [{"duration_s": 2e-7, "maximum_step_s": 2e-7}],
            }
        )
    )
    result = json.loads(json.dumps(run(config)))
    assert audit(config, result)["accepted"]
    assert max(map(abs, result["conservation_residuals"])) < 1e-7


def test_gravity_schema_cannot_silently_drop_null_elevations():
    raw = inclined().to_dict()
    raw["elevations_m"] = None
    with pytest.raises(ValueError, match="requires elevations"):
        PRMultifluidNetwork.from_dict(raw)
