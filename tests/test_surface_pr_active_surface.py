"""Finite-area gravitational column faces couple active phase inventories."""

import json

import pytest
from test_surface_compositional_separator import binary
from test_surface_pr_active_multifluid_pipe import phase

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_active_surface import (
    PRActiveColumnPort,
    PRActiveSurfaceNetwork,
    PRActiveSurfaceState,
)
from surface.separator_geometry import SeparatorGeometry
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState


def setup(*, two_phase=False, reverse=False, device="cpu"):
    f = binary(device)
    c = StratifiedPRColumn(
        f, SeparatorGeometry(0.1, 0.1, "vertical"), quadrature_order=2, maximum_quadrature_order=8
    )
    temperature = 196.2 if two_phase else 320.0
    cs = StratifiedPRState.from_dict(c.initialize(1e6, temperature, [0.5, 0.5])["state"])
    h = cs.interface_heights_m[0] if cs.interface_heights_m else 0.05
    diameter = 0.0005
    p = PRActiveMultifluidPipe(
        f,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [0.1, 0.1],
        diameter,
        face_elevations_m=[h, h + 0.001, h + 0.002],
        gravity_m_s2=c.gravity_m_s2,
    )
    pressure = 1.001e6 if reverse else 0.999e6
    q = f.flash_pt(pressure, temperature, [0.5, 0.5])
    phases = {}
    for index, ph in enumerate(q["phases"]):
        root = "liquid" if float(ph["density_kg_m3"].cpu()) > 100 else "vapor"
        name = "liquid" if root == "liquid" else "gas"
        fraction = float(
            (ph["molar_fraction"] * ph["molar_volume_m3_mol"] / q["molar_volume_m3_mol"]).cpu()
        )
        phases[name] = phase(
            f,
            pressure,
            temperature,
            root,
            p.volumes[0] * fraction,
            x=ph["mole_fractions"].cpu().tolist(),
        )
    ps = p.initialize([phases, phases])
    model = PRActiveSurfaceNetwork(
        {"line": {"pipe": p, "boundary_nodes": ("pickup", None)}},
        {"separator": c},
        [PRActiveColumnPort("pickup", "separator", h)],
    )
    return model, model.initialize({"separator": cs}, {"line": ps})


@pytest.mark.parametrize("reverse", [False, True])
def test_finite_column_exchange_conserves_components_and_gravitational_energy(reverse):
    model, old = setup(reverse=reverse)
    new, r = model.step(old, 0.01 * model.maximum_step_s)
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    assert r["pipes"]["line"]["aperture_quadrature_order"] >= 4
    assert r["pipes"]["line"]["boundary_sample_fluxes"][0]
    assert sum(dict(new.columns)["separator"].component_moles) != sum(
        dict(old.columns)["separator"].component_moles
    )
    saved = PRActiveSurfaceNetwork.from_dict(json.loads(json.dumps(model.to_dict())))
    state = PRActiveSurfaceState.from_dict(json.loads(json.dumps(new.to_dict())))
    saved.inspect(state)
    after, rr = saved.step(state, 0.01 * model.maximum_step_s)
    assert max(map(abs, rr["conservation_residuals"])) < 1e-5
    assert after.time_s > new.time_s


def test_aperture_straddles_actual_interface_without_plenum_or_phase_floor():
    model, old = setup(two_phase=True)
    pipe = model.pipes["line"]["pipe"]
    port = model.ports["pickup"]
    rows = model._samples(
        pipe, dict(old.pipes)["line"].cells[0], port, dict(old.columns)["separator"], 4
    )
    assert sum(r["weight"] for r in rows) == pytest.approx(1.0, abs=1e-12)
    assert {"liquid", "gas"} == {n for r in rows for n in r["cell"]}
    assert len(old.columns) == 1
    _new, r = model.step(old, 0.001 * model.maximum_step_s)
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    assert r["entropy_production_j_k"] >= -1e-6


def test_closed_aperture_keeps_column_inventory_and_full_interval_restart():
    model, old = setup()
    duration = 2.1 * model.maximum_step_s
    r = model.advance(old, duration, port_openings={"pickup": False})
    new = PRActiveSurfaceState.from_dict(r["state"])
    assert dict(new.columns)["separator"] == dict(old.columns)["separator"]
    assert new.time_s == duration
    assert len(r["steps"]) >= 3
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    snapshot = old.to_dict()
    with pytest.raises(ValueError, match="isolation"):
        model.step(old, 0.1 * model.maximum_step_s, port_openings={"unknown": False})
    assert old.to_dict() == snapshot


def test_parallel_apertures_share_one_column_with_opposite_flows():
    model, old = setup()
    pipe = model.pipes["line"]["pipe"]
    h = model.ports["pickup"].height_m
    second = PRActiveMultifluidPipe(
        pipe.fluid,
        pipe.slots,
        pipe.lengths,
        pipe.diameter,
        face_elevations_m=[h, h - 0.001, h - 0.002],
        gravity_m_s2=pipe.gravity,
    )
    cells = []
    for volume in second.volumes:
        cells.append({"gas": phase(pipe.fluid, 1.001e6, 320.0, "vapor", volume, x=(0.5, 0.5))})
    ps = second.initialize(cells)
    network = PRActiveSurfaceNetwork(
        {
            "line": model.pipes["line"],
            "return": {"pipe": second, "boundary_nodes": ("return_pickup", None)},
        },
        model.columns,
        [*model.ports.values(), PRActiveColumnPort("return_pickup", "separator", h)],
    )
    start = network.initialize(dict(old.columns), {**dict(old.pipes), "return": ps})
    _end, r = network.step(start, 0.01 * network.maximum_step_s)
    assert len(r["columns"]) == 1
    assert sum(r["pipes"]["line"]["face_fluxes"][0][j][0] for j in range(2)) > 0
    assert sum(r["pipes"]["return"]["face_fluxes"][0][j][0] for j in range(2)) < 0
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_active_column_equipment_graph_and_pump_rotor_share_total_energy():
    from test_surface_pr_pump_connection import setup as pump_setup

    from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
    from surface.separator_geometry import SeparatorGeometry

    graph, gs, _ = pump_setup(driven=True, dense_limit=1)
    # Existing drive, actual fluid and finite equipment stocks are retained;
    # all nodes lie at datum zero, so changing g does not alter these stocks.
    raw = graph.to_dict()
    raw["gravity_m_s2"] = 9.80665
    graph = PREquilibriumTransportNetwork.from_dict(raw)
    gs = graph.initialize(
        {
            n: {
                "pressure_pa": s.pressure_pa,
                "temperature_k": s.temperature_k,
                "mole_fractions": [v / sum(s.component_moles) for v in s.component_moles],
            }
            for n, s in gs.nodes
        },
        rotor_speeds=dict(gs.rotors),
    )
    graph.inspect(gs)
    f = graph.fluid
    feed = dict(gs.nodes)["feed"]
    c = StratifiedPRColumn(
        f,
        SeparatorGeometry(0.1, 0.1, "vertical"),
        elevation_m=-0.05,
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    cs = StratifiedPRState.from_dict(
        c.initialize(
            feed.pressure_pa,
            feed.temperature_k,
            [v / sum(feed.component_moles) for v in feed.component_moles],
        )["state"]
    )
    diameter = 0.0005
    p = PRActiveMultifluidPipe(
        f,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [0.1, 0.1],
        diameter,
        face_elevations_m=[0.0, 0.001, 0.0],
        gravity_m_s2=9.80665,
    )
    cell = {
        "liquid": phase(
            f,
            feed.pressure_pa,
            feed.temperature_k,
            "liquid",
            p.volumes[0],
            x=tuple(v / sum(feed.component_moles) for v in feed.component_moles),
        )
    }
    ps = p.initialize([cell, cell])
    model = PRActiveSurfaceNetwork(
        {"line": {"pipe": p, "boundary_nodes": ("pickup", "feed")}},
        {"separator": c},
        [PRActiveColumnPort("pickup", "separator", 0.05)],
        equipment=graph,
    )
    old = model.initialize({"separator": cs}, {"line": ps}, gs)
    new, r = model.step(old, 0.001 * model.maximum_step_s)
    assert r["equipment"]["pump_connections"]["booster"]["shaft_power_w"] > 0
    assert dict(new.equipment.rotors)["booster"] != dict(old.equipment.rotors)["booster"]
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_independent_aperture_path_and_column_profile_audit_detects_tampering():
    from copy import deepcopy

    from check_pr_active_surface import audit

    model, old = setup(two_phase=True)
    new, r = model.step(old, 0.001 * model.maximum_step_s)
    assert audit(model, old, new, r)["accepted"]
    damaged = deepcopy(r)
    damaged["pipes"]["line"]["boundary_sample_fluxes"][0][0]["flux"][0][0] += 1.0
    with pytest.raises(ValueError, match="audit differs"):
        audit(model, old, new, damaged)


@pytest.mark.parametrize("wall_temperature", [180.0, 220.0])
def test_thermal_column_wall_moves_interface_with_full_energy_accounting(wall_temperature):
    from surface.vessel_wall import VesselWall

    model, old = setup(two_phase=True)
    wall = VesselWall(100.0, 1e4, 0.0, 200.0, max_temperature_change_k=20.0)
    coupled = PRActiveSurfaceNetwork(
        model.pipes, model.columns, list(model.ports.values()), column_walls={"separator": wall}
    )
    start = coupled.initialize(
        dict(old.columns), dict(old.pipes), wall_temperatures_k={"separator": wall_temperature}
    )
    new, r = coupled.step(start, 0.1 * coupled.maximum_step_s, port_openings={"pickup": False})
    a = dict(start.columns)["separator"]
    b = dict(new.columns)["separator"]
    assert b.component_moles == a.component_moles
    assert (
        b.total_energy_j > a.total_energy_j
        if wall_temperature > 200
        else b.total_energy_j < a.total_energy_j
    )
    assert b.interface_heights_m != a.interface_heights_m
    assert r["columns"]["separator"]["thermal"]["fluid_heat_j"] != 0.0
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    restored = PRActiveSurfaceNetwork.from_dict(json.loads(json.dumps(coupled.to_dict())))
    checkpoint = PRActiveSurfaceState.from_dict(json.loads(json.dumps(new.to_dict())))
    restored.inspect(checkpoint)
    assert dict(checkpoint.wall_temperatures) == dict(new.wall_temperatures)


def test_thermal_column_and_open_aperture_share_material_and_ambient_energy():
    from surface.vessel_wall import VesselWall

    model, old = setup()
    wall = VesselWall(100.0, 1e4, 50.0, 400.0, max_temperature_change_k=20.0)
    coupled = PRActiveSurfaceNetwork(
        model.pipes, model.columns, list(model.ports.values()), column_walls={"separator": wall}
    )
    start = coupled.initialize(
        dict(old.columns), dict(old.pipes), wall_temperatures_k={"separator": 350.0}
    )
    _new, r = coupled.step(start, 0.1 * coupled.maximum_step_s)
    assert r["external_energy_j"] > 0
    assert r["columns"]["separator"]["incoming_rates"][0] != 0
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_independent_thermal_wall_audit_and_tampered_heat_detection():
    from copy import deepcopy

    from check_pr_active_surface import audit

    from surface.vessel_wall import VesselWall

    model, old = setup(two_phase=True)
    wall = VesselWall(100.0, 1e4, 50.0, 250.0, max_temperature_change_k=20.0)
    coupled = PRActiveSurfaceNetwork(
        model.pipes, model.columns, list(model.ports.values()), column_walls={"separator": wall}
    )
    start = coupled.initialize(
        dict(old.columns), dict(old.pipes), wall_temperatures_k={"separator": 220.0}
    )
    new, r = coupled.step(start, 0.1 * coupled.maximum_step_s)
    assert audit(coupled, start, new, r)["accepted"]
    bad = deepcopy(r)
    bad["columns"]["separator"]["thermal"]["ambient_heat_j"] += 1.0
    with pytest.raises(ValueError, match="audit differs"):
        audit(coupled, start, new, bad)


def test_partial_aperture_uses_open_area_plus_closed_wall_traction():
    import numpy as np

    model, old = setup(two_phase=True)
    dt = 0.001 * model.maximum_step_s
    _, opened = model.step(old, dt)
    _, closed = model.step(old, dt, port_positions={"pickup": 0.0})
    _new, half = model.step(old, dt, port_positions={"pickup": 0.5})
    fan = lambda r: np.array(r["pipes"]["line"]["face_fluxes"][0])
    np.testing.assert_allclose(
        fan(half), 0.5 * fan(opened) + 0.5 * fan(closed), atol=1e-5, rtol=1e-10
    )
    np.testing.assert_allclose(
        half["columns"]["separator"]["incoming_rates"],
        0.5 * np.array(opened["columns"]["separator"]["incoming_rates"]),
        atol=1e-9,
        rtol=1e-9,
    )
    rays = half["pipes"]["line"]["boundary_sample_fluxes"][0]
    assert sum(r["weight"] for r in rays if r["kind"] == "wall") == pytest.approx(0.5)
    assert sum(r["weight"] for r in rays) == pytest.approx(1.0)
    assert max(map(abs, half["conservation_residuals"])) < 1e-5


def test_aperture_stop_and_restart_with_fractional_position_is_atomic():
    model, state = setup()
    dt = 0.001 * model.maximum_step_s
    for position in (1.0, 0.0, 0.25, 1.0):
        state, r = model.step(state, dt, port_positions={"pickup": position})
        assert max(map(abs, r["conservation_residuals"])) < 1e-5
    snapshot = state.to_dict()
    with pytest.raises(ValueError, match="fraction"):
        model.step(state, dt, port_positions={"pickup": 1.1})
    assert state.to_dict() == snapshot


def test_fractional_aperture_independent_audit_checks_mask_and_wall_ghosts():
    from copy import deepcopy

    from check_pr_active_surface import audit

    model, old = setup(two_phase=True)
    new, r = model.step(old, 0.001 * model.maximum_step_s, port_positions={"pickup": 0.5})
    assert audit(model, old, new, r)["accepted"]
    bad = deepcopy(r)
    bad["port_positions"]["pickup"] = 0.75
    with pytest.raises(ValueError, match="audit differs"):
        audit(model, old, new, bad)


def test_operating_schedule_persists_commands_and_report_intervals(tmp_path, monkeypatch):
    from run_pr_active_surface import run

    model, state = setup()
    dt = 0.001 * model.maximum_step_s
    config = json.loads(
        json.dumps(
            {
                "system": model.to_dict(),
                "initial": state.to_dict(),
                "schedule": [
                    {"duration_s": dt, "commands": {"port_positions": {"pickup": 0.5}}},
                    {"duration_s": dt, "commands": {"port_openings": {"pickup": False}}},
                    {"duration_s": dt, "commands": {"port_openings": {"pickup": True}}},
                ],
            }
        )
    )
    report = run(config)
    assert len(report["intervals"]) == 3
    assert report["intervals"][-1]["commands"]["port_positions"]["pickup"] == 0.5
    assert report["intervals"][1]["steps"][0]["port_positions"]["pickup"] == 0.0
    assert report["result"]["state"]["time_s"] == pytest.approx(3 * dt, abs=1e-20)
    assert max(map(abs, report["result"]["conservation_residuals"])) < 1e-5
    from check_pr_active_surface import main

    saved = tmp_path / "scheduled.json"
    saved.write_text(json.dumps(report))
    monkeypatch.setattr("sys.argv", ["check_pr_active_surface.py", str(saved)])
    main()


@pytest.mark.parametrize(
    "initial_two_phase,wall_temperature,expected_phases", [(True, 480.0, 1), (False, 170.0, 2)]
)
def test_stiff_wall_heat_discovers_phase_birth_and_extinction_without_invalid_explicit_uv_seed(
    initial_two_phase, wall_temperature, expected_phases
):
    from check_pr_active_surface import audit

    from surface.vessel_wall import VesselWall

    model, old = setup(two_phase=initial_two_phase)
    wall = VesselWall(1e4, 1e9, 0.0, 300.0, max_temperature_change_k=20.0)
    coupled = PRActiveSurfaceNetwork(
        model.pipes, model.columns, list(model.ports.values()), column_walls={"separator": wall}
    )
    state = coupled.initialize(
        dict(old.columns), dict(old.pipes), wall_temperatures_k={"separator": wall_temperature}
    )
    new, r = coupled.step(state, 0.1 * coupled.maximum_step_s, port_openings={"pickup": False})
    assert len(dict(new.columns)["separator"].phase_roots) == expected_phases
    assert r["columns"]["separator"]["thermal"]["seed_strategy"] == "bounded_temperature_pt"
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    assert audit(coupled, state, new, r)["accepted"]
