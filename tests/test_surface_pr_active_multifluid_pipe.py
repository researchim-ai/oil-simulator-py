"""Active multifluid transport: no residual ghost inventories or fixed slots."""

import math

import pytest
from test_surface_equilibrium_thermodynamics import fluid

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_interphase_relaxation import PRRelaxationPhase


def phase(f, p, t, root, volume, u=0.0, x=(0.0, 1.0)):
    q = f.homogeneous(p, t, x, root)
    n = volume / float(q["molar_volume_m3_mol"].cpu())
    mass = n * float(q["molar_mass_kg_mol"].cpu())
    return PRRelaxationPhase(
        tuple(n * v for v in x),
        volume,
        p,
        t,
        mass * u,
        n * float(q["molar_internal_energy_j_mol"].cpu()) + 0.5 * mass * u * u,
    )


def setup(*, mode="walls", chemical=None, count=3):
    f = fluid()
    diameter = 0.1
    volume = 1e-4
    area = math.pi * diameter**2 / 4
    model = PRActiveMultifluidPipe(
        f,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [volume / area] * count,
        diameter,
        boundary_mode=mode,
        thermochemical_relaxation_time_s=chemical,
    )
    return model, volume


def test_stationary_material_interface_preserves_holdup_and_exactly_absent_phases():
    model, v = setup(count=2)
    f = model.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    old = model.initialize(
        [{"liquid": phase(f, p, 200.0, "liquid", v)}, {"gas": phase(f, p, 200.0, "vapor", v)}]
    )
    end, r = model.step(old, 0.5 * model.maximum_step_s)
    assert r["active_phase_counts"] == [1, 1]
    assert end.cells == old.cells
    assert max(map(abs, r["conservation_residuals"])) == 0.0
    assert r["contact_faces"] == [True, True, True]


def test_material_contact_advects_a_phase_into_a_previously_empty_cell():
    model, v = setup(mode="periodic")
    f = model.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    old = model.initialize(
        [
            {"liquid": phase(f, p, 200.0, "liquid", v, 1.0)},
            {"gas": phase(f, p, 200.0, "vapor", v, 1.0)},
            {"gas": phase(f, p, 200.0, "vapor", v, 1.0)},
        ]
    )
    end, r = model.step(old, 0.5 * model.maximum_step_s)
    assert [n for n, _ in end.cells[1]] == ["liquid", "gas"]
    assert sum(dict(end.cells[1])["liquid"].component_moles) > 0.0
    assert r["entropy_production_j_k"] >= -1e-7
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_single_present_gas_has_no_phantom_pressure_or_energy_in_absent_liquid():
    model, v = setup(mode="periodic")
    f = model.fluid
    old = model.initialize(
        [{"gas": phase(f, p, 320.0, "vapor", v, x=(0.5, 0.5))} for p in (1e6, 0.99e6, 1e6)]
    )
    end, r = model.step(old, 0.2 * model.maximum_step_s)
    assert all([name for name, _ in cell] == ["gas"] for cell in end.cells)
    assert any(abs(p.momentum_kg_m_s) > 0.0 for cell in end.cells for _, p in cell)
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_phase_birth_from_chemical_source_is_followed_by_actual_spatial_transport():
    model, v = setup(mode="periodic", chemical=0.001)
    f = model.fluid
    p = f.components[1].saturation_at_temperature(250.0)["pressure_pa"] * 0.9
    old = model.initialize(
        [{"liquid": phase(f, pp, 250.0, "liquid", v)} for pp in (p, p * 1.25, p * 1.25)]
    )
    born, r = model.step(old, 0.5 * model.maximum_step_s)
    assert r["active_phase_counts"] == [2, 1, 1]
    assert (
        abs(dict(born.cells[0])["liquid"].pressure_pa - dict(born.cells[0])["gas"].pressure_pa)
        > 1.0
    )
    end, r = model.step(born, 0.5 * model.maximum_step_s)
    assert all(len(cell) == 2 for cell in end.cells)
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_instantaneous_condensation_removes_the_gas_from_every_cell():
    f = fluid()
    l = phase(f, 2e6, 180.0, "liquid", 1e-4)
    g = phase(f, 1e5, 220.0, "vapor", 1e-10)
    volume = l.volume_m3 + g.volume_m3
    diameter = 0.1
    area = math.pi * diameter**2 / 4
    model = PRActiveMultifluidPipe(
        f,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [volume / area] * 2,
        diameter,
        boundary_mode="periodic",
        thermochemical_relaxation_time_s=0.0,
    )
    old = model.initialize([{"liquid": l, "gas": g}] * 2)
    end, r = model.step(old, 0.25 * model.maximum_step_s)
    assert r["active_phase_counts"] == [1, 1]
    assert all(tuple(dict(cell)) == ("liquid",) for cell in end.cells)
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_three_liquid_phase_markers_keep_static_and_advected_interfaces():
    from test_surface_equilibrium_thermodynamics import symmetric

    f = symmetric()
    equilibrium = f.flash_pt(1e6, 300.0, [1 / 3] * 3)
    xs = {
        int(q["mole_fractions"].argmax().cpu()): q["mole_fractions"].cpu().tolist()
        for q in equilibrium["phases"]
    }
    volume = 1e-4
    diameter = 0.1
    area = math.pi * diameter**2 / 4
    slots = [
        {"name": name, "root": "liquid", "component_marker": j}
        for j, name in enumerate(("a", "b", "c"))
    ]
    model = PRActiveMultifluidPipe(
        f, slots, [volume / area] * 3, diameter, boundary_mode="periodic"
    )
    old = model.initialize(
        [{slots[j]["name"]: phase(f, 1e6, 300.0, "liquid", volume, x=xs[j])} for j in range(3)]
    )
    still, r = model.step(old, 0.25 * model.maximum_step_s)
    assert still.cells == old.cells
    assert r["active_phase_counts"] == [1, 1, 1]
    moving = model.initialize(
        [{slots[j]["name"]: phase(f, 1e6, 300.0, "liquid", volume, 0.3, xs[j])} for j in range(3)]
    )
    _end, r = model.step(moving, 0.25 * model.maximum_step_s)
    assert r["active_phase_counts"] == [2, 2, 2]
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_independent_active_audit_without_runtime_flux_or_recovery(monkeypatch):
    import copy
    import json

    from check_pr_active_multifluid_pipe import audit
    from run_pr_active_multifluid_pipe import run

    from surface.equilibrium_properties import EquilibriumFluid

    model, v = setup(mode="periodic")
    f = model.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    old = model.initialize(
        [
            {"liquid": phase(f, p, 200.0, "liquid", v, 1.0)},
            {"gas": phase(f, p, 200.0, "vapor", v, 1.0)},
            {"gas": phase(f, p, 200.0, "vapor", v, 1.0)},
        ]
    )
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": old.to_dict()},
                "schedule": [{"duration_s": model.maximum_step_s * 0.5}],
            }
        )
    )
    report = json.loads(json.dumps(run(config)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent active audit called runtime physics")

    monkeypatch.setattr(PRActiveMultifluidPipe, "inspect", forbidden)
    monkeypatch.setattr(EquilibriumFluid, "flash_uv", forbidden)
    assert audit(config, report)["accepted"]
    changed = copy.deepcopy(report)
    changed["intervals"][0]["steps"][0]["face_fluxes"][0][0][-1] += 1.0
    with pytest.raises(ValueError):
        audit(config, changed)


def test_no_eos_calls_are_made_for_the_absent_slot(monkeypatch):
    model, v = setup(mode="periodic")
    f = model.fluid
    old = model.initialize(
        [{"gas": phase(f, p, 320.0, "vapor", v, x=(0.5, 0.5))} for p in (1e6, 0.99e6, 1e6)]
    )
    original = f.homogeneous

    def only_gas(p, t, x, root="stable"):
        assert root == "vapor"
        return original(p, t, x, root)

    monkeypatch.setattr(f, "homogeneous", only_gas)
    end, _ = model.step(old, 0.1 * model.maximum_step_s)
    assert all(tuple(dict(cell)) == ("gas",) for cell in end.cells)


def test_independent_audit_of_nonuniform_birth_and_next_transport_step():
    import json

    from check_pr_active_multifluid_pipe import audit
    from run_pr_active_multifluid_pipe import run

    model, v = setup(mode="periodic", chemical=0.001)
    f = model.fluid
    p = 0.9 * f.components[1].saturation_at_temperature(250.0)["pressure_pa"]
    old = model.initialize(
        [{"liquid": phase(f, pp, 250.0, "liquid", v)} for pp in (p, p * 1.25, p * 1.25)]
    )
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": old.to_dict()},
                "schedule": [
                    {"duration_s": model.maximum_step_s * 0.5},
                    {"duration_s": model.maximum_step_s * 0.5},
                ],
            }
        )
    )
    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    assert [len(c) for c in report["intervals"][0]["state"]["cells"]] == [2, 1, 1]
    assert [len(c) for c in report["state"]["cells"]] == [2, 2, 2]


def test_gas_oil_water_counterflow_uses_three_actual_phase_velocities():
    import json
    from pathlib import Path

    from surface.equilibrium_properties import EquilibriumFluid

    raw = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_oil_water_clean.json"
        ).read_text()
    )
    f = EquilibriumFluid.from_dict(raw["model"]["fluid"])
    eq = f.flash_pt(1e6, 350.0, raw["initial"]["nodes"]["0"]["mole_fractions"])
    compositions = {
        int(q["mole_fractions"].argmax().cpu()): q["mole_fractions"].cpu().tolist()
        for q in eq["phases"]
    }
    assert set(compositions) == {0, 1, 2}
    volume = 1e-4
    diameter = 0.1
    area = math.pi * diameter**2 / 4
    slots = [
        {"name": "gas", "root": "vapor", "component_marker": 0},
        {"name": "oil", "root": "liquid", "component_marker": 1},
        {"name": "water", "root": "liquid", "component_marker": 2},
    ]
    model = PRActiveMultifluidPipe(
        f, slots, [volume / area] * 3, diameter, boundary_mode="periodic"
    )
    fractions = [(0.2, 0.04, 0.76), (0.3, 0.05, 0.65), (0.25, 0.03, 0.72)]
    velocities = (0.2, 0.05, -0.01)
    cells = [
        {
            slot["name"]: phase(f, 1e6, 350.0, slot["root"], volume * alpha, u, compositions[j])
            for j, (slot, alpha, u) in enumerate(zip(slots, weights, velocities))
        }
        for weights in fractions
    ]
    old = model.initialize(cells)
    end, r = model.step(old, 0.01 * model.maximum_step_s)
    assert r["active_phase_counts"] == [3, 3, 3]
    for cell in end.cells:
        data = dict(cell)

        def speed(name, data=data):
            p = data[name]
            mass = sum(n * c.molar_mass_kg_mol for n, c in zip(p.component_moles, f.components))
            return p.momentum_kg_m_s / mass

        assert speed("gas") > speed("oil") > speed("water")
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_many_contact_steps_follow_upwind_holdup_without_acoustic_smearing():
    import numpy as np

    model, v = setup(mode="periodic")
    f = model.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    state = model.initialize(
        [
            {"liquid": phase(f, p, 200.0, "liquid", v, 1.0)},
            {"gas": phase(f, p, 200.0, "vapor", v, 1.0)},
            {"gas": phase(f, p, 200.0, "vapor", v, 1.0)},
        ]
    )
    dt = 0.5 * model.maximum_step_s
    fraction = dt / model.lengths[0]
    expected = np.array([1.0, 0.0, 0.0])
    for _ in range(20):
        state, r = model.step(state, dt)
        expected = (1 - fraction) * expected + fraction * np.roll(expected, 1)
        assert all(r["contact_faces"])
    actual = [
        dict(cell)["liquid"].volume_m3 / v if "liquid" in dict(cell) else 0.0
        for cell in state.cells
    ]
    assert actual == pytest.approx(expected, abs=1e-11)


def test_conditioned_contact_recognition_does_not_hide_resolved_pressure_waves():
    model, v = setup(mode="periodic")
    f = model.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    old = model.initialize(
        [{"liquid": phase(f, pp, 200.0, "liquid", v)} for pp in (p, p + 0.01, p + 0.01)]
    )
    new, r = model.step(old, 0.1 * model.maximum_step_s)
    assert not all(r["contact_faces"])
    assert any(abs(stock.momentum_kg_m_s) > 0.0 for cell in new.cells for _, stock in cell)


def test_active_restart_interval_and_cfl_refusal_keep_original_state():
    import json

    from surface.pr_active_multifluid_pipe import PRActiveMultifluidState

    model, v = setup(mode="periodic")
    f = model.fluid
    old = model.initialize(
        [{"gas": phase(f, 1e6, 320.0, "vapor", v, x=(0.5, 0.5))} for _ in model.volumes]
    )
    saved = old.to_dict()
    with pytest.raises(ValueError, match="CFL"):
        model.step(old, model.maximum_step_s * 2)
    assert old.to_dict() == saved
    copied = PRActiveMultifluidPipe.from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PRActiveMultifluidState.from_dict(json.loads(json.dumps(old.to_dict())))
    duration = 2.5 * model.maximum_step_s
    report = copied.advance(checkpoint, duration)
    assert report["state"]["time_s"] == duration
    assert len(report["steps"]) == 3
    assert max(map(abs, report["conservation_residuals"])) < 1e-6


def test_single_phase_transport_matches_the_separate_hem_euler_solver():
    from surface.pr_equilibrium_euler_pipe import PREquilibriumEulerPipe

    active, v = setup(mode="periodic")
    f = active.fluid
    old = active.initialize(
        [{"gas": phase(f, p, 320.0, "vapor", v, x=(0.5, 0.5))} for p in (1e6, 0.99e6, 1e6)]
    )
    hem = PREquilibriumEulerPipe(f, active.lengths, active.diameter, boundary_mode="periodic")
    specifications = []
    for cell in old.cells:
        p = dict(cell)["gas"]
        specifications.append(
            {
                "component_moles": p.component_moles,
                "total_energy_j": p.total_energy_j,
                "momentum_kg_m_s": p.momentum_kg_m_s,
                "pressure_guess_pa": p.pressure_pa,
                "temperature_guess_k": p.temperature_k,
            }
        )
    reference = hem.initialize(specifications)
    dt = 0.2 * active.maximum_step_s
    end, _ = active.step(old, dt)
    expected, _ = hem.step(reference, dt)
    for cell, bulk in zip(end.cells, expected.bulk_stocks):
        p = dict(cell)["gas"]
        assert [*p.component_moles, p.momentum_kg_m_s, p.total_energy_j] == pytest.approx(
            bulk, abs=1e-8
        )


def test_transport_entropy_loss_cannot_be_hidden_by_chemical_relaxation(monkeypatch):
    import surface.pr_active_multifluid_pipe as implementation
    from surface.liquid import ConservationError

    model, v = setup(mode="periodic", chemical=0.001)
    f = model.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    old = model.initialize(
        [
            {
                "liquid": phase(f, p, 200.0, "liquid", v * 0.5),
                "gas": phase(f, p, 200.0, "vapor", v * 0.5),
            }
            for _ in model.volumes
        ]
    )
    dt = 0.1 * model.maximum_step_s
    original = implementation.fluctuations

    def bad_transport(*args, **kwargs):
        dm, dp, am, ap, faces = original(*args, **kwargs)
        dm = dm.clone()
        dm[:, 0, -2] -= 0.001 / v * model.lengths[0] / dt
        dm[:, 1, -2] += 0.001 / v * model.lengths[0] / dt
        return dm, dp, am, ap, faces

    def forbidden(*args, **kwargs):
        raise AssertionError("Chemical source hid a transport error")

    monkeypatch.setattr(implementation, "fluctuations", bad_transport)
    monkeypatch.setattr(model.chemical, "step", forbidden)
    saved = old.to_dict()
    with pytest.raises(ConservationError, match="before chemical"):
        model.step(old, dt)
    assert old.to_dict() == saved


@pytest.mark.parametrize("velocity", [1.0, -1.0])
def test_open_endpoints_uniform_throughflow_preserves_state_and_shared_ledger(velocity):
    model, v = setup(count=2)
    f = model.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    cell = {"gas": phase(f, p, 200.0, "vapor", v, velocity)}
    old = model.initialize([cell, cell])
    new, report = model.step(old, 0.2 * model.maximum_step_s, boundary_cells=[cell, cell])
    assert new.cells == old.cells
    assert max(map(abs, report["conservation_residuals"])) < 1e-8
    assert abs(report["incoming_entropy_j_k"]) < 1e-12
    assert abs(report["wall_impulse_kg_m_s"]) < 1e-12
    assert max(map(abs, report["external_transfer"])) < 1e-12


@pytest.mark.parametrize("velocity", [1.0, -1.0])
def test_open_boundary_introduces_previously_absent_phase_without_ghost_stock(velocity):
    model, v = setup(count=2)
    f = model.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    gas = {"gas": phase(f, p, 200.0, "vapor", v, velocity)}
    liquid = {"liquid": phase(f, p, 200.0, "liquid", v, velocity)}
    old = model.initialize([gas, gas])
    boundaries = [liquid, gas] if velocity > 0 else [gas, liquid]
    new, report = model.step(old, 0.2 * model.maximum_step_s, boundary_cells=boundaries)
    endpoint = 0 if velocity > 0 else -1
    assert "liquid" in dict(new.cells[endpoint])
    assert len(new.cells[1 if endpoint == 0 else 0]) == 1
    before = model._totals(old)
    after = model._totals(new)
    expected = report["external_transfer"]
    for index in range(len(before) - 1):
        assert after[index] - before[index] == pytest.approx(expected[index], abs=1e-7)
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    assert report["transport_entropy_production_j_k"] >= -1e-7
    assert old.cells == model.initialize([gas, gas]).cells


def test_open_boundary_rejects_periodic_and_invalid_phase_inventory_atomically():
    model, v = setup(mode="periodic", count=2)
    cell = {"gas": phase(model.fluid, 1e6, 320.0, "vapor", v, x=(0.5, 0.5))}
    old = model.initialize([cell, cell])
    with pytest.raises(ValueError, match="endpoint"):
        model.step(old, model.maximum_step_s, boundary_cells=[cell, None])
    closed, _ = setup(count=2)
    old = closed.initialize([cell, cell])
    with pytest.raises(ValueError, match="nonempty"):
        closed.step(old, closed.maximum_step_s, boundary_cells=[{}, None])


def test_open_boundary_flux_matches_independent_scalar_pr_path_quadrature():
    import numpy as np
    from reference_pr_active_multifluid import face, packets

    model, v = setup(count=2)
    interior = {"gas": phase(model.fluid, 1e6, 320.0, "vapor", v, 0.2, x=(0.5, 0.5))}
    boundary = {"gas": phase(model.fluid, 1.001e6, 321.0, "vapor", v, -0.1, x=(0.49, 0.51))}
    old = model.initialize([interior, interior])
    ghost = model.initialize([boundary, boundary])
    _, report = model.step(old, 0.01 * model.maximum_step_s, boundary_cells=[boundary, None])
    a, b = packets(model, ghost.to_dict()), packets(model, old.to_dict())
    keys = (
        "conserved",
        "alpha",
        "pressure",
        "temperature",
        "velocity",
        "pressure_precision",
        "flux",
    )
    ref = face({k: a[k][0] for k in keys}, {k: b[k][0] for k in keys}, model.wave_speed)[-1]
    np.testing.assert_allclose(report["face_fluxes"][0], ref["flux"], rtol=1e-11, atol=1e-6)
    np.testing.assert_allclose(report["phase_path"][0], ref["phase_path"], rtol=1e-11, atol=1e-6)
    np.testing.assert_allclose(report["alpha_path"][0], ref["alpha_path"], rtol=1e-11, atol=1e-10)
