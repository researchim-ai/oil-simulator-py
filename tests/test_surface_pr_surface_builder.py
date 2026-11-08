"""Direct finite apertures, surveyed drift pipes and native graph persistence."""

import json
import math

import pytest
from test_surface_compositional_separator import binary

from surface.drift_flux_closure import RELAP7VerticalDriftClosure, ZuberFindlayClosure
from surface.pr_equilibrium_transport import PREquilibriumNode
from surface.pr_gravity_transport import PRGravityTransportNetwork, PRGravityTransportState
from surface.pr_surface_builder import PRColumnTerminal, build_pr_surface_network
from surface.pr_terrain_pipe import PRTerrainPipe
from surface.pr_transport_builder import PREquilibriumPipe, initialize_grouped
from surface.separator_geometry import SeparatorGeometry
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState


def setup(*, reverse=False, slip=False, parallel=False, device="cpu"):
    f = binary(device)
    column = StratifiedPRColumn(
        f, SeparatorGeometry(0.1, 0.1, "vertical"), quadrature_order=2, maximum_quadrature_order=8
    )
    temperature = 196.2 if slip else 320.0
    cs = StratifiedPRState.from_dict(column.initialize(1e6, temperature, [0.5, 0.5])["state"])
    h = cs.interface_heights_m[0] if cs.interface_heights_m else 0.05
    closure = (
        ZuberFindlayClosure(1.0, 0.001, maximum_gas_volume_fraction=0.999999) if slip else None
    )
    pipes = [
        PRTerrainPipe(
            "line", "pickup", "receiver", 0.0005, (0.1, 0.1), (h, h, h), (1, 1), drift_flux=closure
        )
    ]
    if parallel:
        pipes.append(
            PREquilibriumPipe("parallel", "pickup", "receiver", 0.2, 0.0005, 2, drift_flux=closure)
        )
    model = build_pr_surface_network(
        f,
        [PREquilibriumNode("receiver", 0.001, elevation_m=h)],
        pipes,
        columns={"separator": column},
        column_terminals=[PRColumnTerminal("pickup", "separator", h)],
        dense_limit=768,
        initial_step_s=1e-6,
        maximum_step_s=1e-5,
    )
    spec = {
        "pressure_pa": 1.01e6 if reverse else 0.99e6,
        "temperature_k": temperature,
        "mole_fractions": [0.5, 0.5],
    }
    initial = initialize_grouped(
        model.transport, {"receiver": spec, **{p.name: spec for p in pipes}}
    )
    state = model.initialize(initial, {"separator": cs})
    commands = {**model.transport.default_openings, **{p.name: True for p in model.ports}}
    return model, state, commands


@pytest.mark.parametrize("reverse", [False, True])
def test_direct_column_nozzle_uses_actual_pipe_face_and_joint_balances(reverse):
    model, old, commands = setup(reverse=reverse)
    assert "pickup" not in model.transport.names
    assert len(model.transport.nodes) == 3
    assert model.ports[0].node == "line__0000"
    end, r = model._step(old, 1e-6, commands, {}, {}, {})
    flux = r["columns"]["ports"]["line__column_left"]["flux"]
    assert flux[0] < 0 if reverse else flux[0] > 0
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    copied = PRGravityTransportNetwork.from_dict(json.loads(json.dumps(model.to_dict())))
    copied.inspect(PRGravityTransportState.from_dict(json.loads(json.dumps(end.to_dict()))))


def test_surveyed_slip_closure_reaches_each_cell_and_actual_fluxes():
    model, old, commands = setup(slip=True)
    for node in model.transport.nodes:
        if node.kind == "pipe_cell":
            assert isinstance(node.drift_flux, ZuberFindlayClosure)
    olddata = model.transport._state_data(
        model.transport.names.index("line__0000"), dict(old.transport.nodes)["line__0000"]
    )
    assert float((olddata["phase_velocity"].max() - olddata["phase_velocity"].min()).cpu()) > 0
    _, report = model._step(old, 1e-7, commands, {}, {}, {})
    assert max(map(abs, report["conservation_residuals"])) < 1e-5


def test_parallel_native_branches_share_one_finite_gravitational_column():
    model, old, commands = setup(parallel=True)
    assert len(model.ports) == 2
    assert len(old.columns) == 1
    end, r = model._step(old, 1e-6, commands, {}, {}, {})
    assert sum(dict(end.columns)["separator"].component_moles) < sum(
        dict(old.columns)["separator"].component_moles
    )
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_survey_cannot_extrapolate_a_vertical_closure_to_horizontal_cells():
    f = binary()
    closure = RELAP7VerticalDriftClosure("dispersed_bubbly", 0.02, 0.001)
    pipe = PRTerrainPipe(
        "line", "a", "b", 0.01, (1.0, 1.0), (0.0, 0.0, 0.0), (1, 1), drift_flux=closure
    )
    with pytest.raises(ValueError, match="vertical"):
        build_pr_surface_network(
            f,
            [PREquilibriumNode("a", 0.001), PREquilibriumNode("b", 0.001)],
            [pipe],
            gravity_m_s2=9.80665,
        )


def test_terminal_registry_rejects_unused_or_nonexistent_column():
    f = binary()
    with pytest.raises(ValueError, match="Unknown terminal column"):
        build_pr_surface_network(
            f, [], [], column_terminals=[PRColumnTerminal("pickup", "missing", 0.01)]
        )


def test_native_phase_birth_and_death_keep_slip_energy_in_the_same_unknowns():
    from surface.peng_robinson import R
    from surface.pr_equilibrium_transport import (
        PREquilibriumTransportNetwork,
        PREquilibriumTransportState,
    )

    f = binary()
    c = f.components[1]
    p = c.saturation_at_temperature(200.0)["pressure_pa"]
    l = c.state(p, 200.0, "liquid", derivatives=False)
    v = c.state(p, 200.0, "vapor", derivatives=False)
    mw = c.molar_mass_kg_mol
    vm = mw * (0.4 * l["specific_volume_m3_kg"] + 0.6 * v["specific_volume_m3_kg"])
    target = (
        10
        * mw
        * (
            0.4 * (l["enthalpy_j_kg"] - p * l["specific_volume_m3_kg"])
            + 0.6 * (v["enthalpy_j_kg"] - p * v["specific_volume_m3_kg"])
        )
    )
    attraction, _ = c._a(220.0)
    p0 = R * 220 / (vm - c.b) - attraction / (vm * vm + 2 * c.b * vm - c.b * c.b)
    volume = 10 * vm
    area = math.pi * 0.05**2 / 4
    closure = ZuberFindlayClosure(1.0, 0.001, maximum_gas_volume_fraction=0.999999)
    node = PREquilibriumNode(
        "cell", volume, "pipe_cell", length_m=volume / area, diameter_m=0.05, drift_flux=closure
    )
    model = PREquilibriumTransportNetwork(f, [node], initial_step_s=1.0, maximum_step_s=1.0)
    old = model.initialize(
        {"cell": {"pressure_pa": p0, "temperature_k": 220.0, "mole_fractions": [0.0, 1.0]}}
    )
    energy = old.nodes[0][1].total_energy_j
    r = model.advance(
        old, 1.0, heat_w={"cell": target - energy}, utility_temperature_k={"cell": 170.0}
    )
    born = PREquilibriumTransportState.from_dict(r["state"])
    assert len(born.nodes[0][1].phase_roots) == 2
    data = model._state_data(0, born.nodes[0][1])
    assert float(data["kinetic"].cpu()) > 0.0
    assert float((data["phase_velocity"].max() - data["phase_velocity"].min()).cpu()) > 0.0
    assert born.nodes[0][1].total_energy_j == pytest.approx(target, abs=1e-5)
    r = model.advance(
        born, 1.0, heat_w={"cell": energy - target}, utility_temperature_k={"cell": 300.0}
    )
    dead = PREquilibriumTransportState.from_dict(r["state"])
    assert len(dead.nodes[0][1].phase_roots) == 1
    assert dead.nodes[0][1].component_moles == pytest.approx(
        old.nodes[0][1].component_moles, abs=1e-10
    )
    assert dead.nodes[0][1].total_energy_j == pytest.approx(energy, abs=1e-5)


def test_resolved_separator_and_lateral_nozzle_compile_into_the_same_graph():
    from surface.pr_resolved_separator import PRLateralNozzle, PRResolvedSeparator

    f = binary()
    separator = PRResolvedSeparator("resolved", 0.01, 0.04, 2, 1e-6)
    z = separator.cell_elevation_m(0)
    vessel = PREquilibriumNode("receiver", 1e-5, elevation_m=z)
    nozzle = PRLateralNozzle(
        "product", separator.cell_name(0), "receiver", 0.001, separator.cell_height_m / 2
    )
    model = build_pr_surface_network(
        f,
        [vessel],
        [],
        resolved_separators=[separator],
        resolved_nozzles=[nozzle],
        gravity_m_s2=9.80665,
        initial_step_s=1e-6,
        maximum_step_s=1e-4,
        minimum_step_s=1e-10,
        dense_limit=1,
    )
    spec = {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]}
    ts = initialize_grouped(
        model.transport, {"resolved": spec, "receiver": {**spec, "pressure_pa": 999000.0}}
    )
    state = model.initialize(ts, {})
    end, r = model._step(state, 1e-7, model.transport.default_openings, {}, {}, {})
    assert r["transport"]["links"]["product"]["flux"][0] > 0
    assert len(end.transport.nodes) == 3
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_direct_column_to_column_line_has_no_fictitious_terminal_stocks():
    base, old, _ = setup()
    c = base.columns["separator"]
    h = base.ports[0].height_m
    pipe = PREquilibriumPipe("line", "inlet", "outlet", 0.2, 0.0005, 2)
    model = build_pr_surface_network(
        c.fluid,
        [],
        [pipe],
        columns={"a": c, "b": c},
        column_terminals=[PRColumnTerminal("inlet", "a", h), PRColumnTerminal("outlet", "b", h)],
        dense_limit=768,
        initial_step_s=1e-6,
        maximum_step_s=1e-5,
    )
    ts = initialize_grouped(
        model.transport,
        {"line": {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]}},
    )
    cs = dict(old.columns)["separator"]
    state = model.initialize(ts, {"a": cs, "b": cs})
    commands = {**model.transport.default_openings, **{p.name: True for p in model.ports}}
    _, r = model._step(state, 1e-7, commands, {}, {}, {})
    assert set(model.transport.names) == {"line__0000", "line__0001"}
    assert len(state.columns) == 2
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_three_phase_marker_slip_and_resolved_aperture_share_surveyed_energy():
    from pathlib import Path

    from surface.drift_flux_closure import drift_closure_from_dict
    from surface.equilibrium_properties import EquilibriumFluid

    raw = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_oil_water_clean.json"
        ).read_text()
    )
    f = EquilibriumFluid.from_dict(raw["model"]["fluid"])
    closure = drift_closure_from_dict(raw["model"]["nodes"][0]["drift_flux"])
    spec = raw["initial"]["nodes"]["0"]
    column = StratifiedPRColumn(
        f, SeparatorGeometry(0.1, 0.1, "vertical"), quadrature_order=2, maximum_quadrature_order=8
    )
    cs = StratifiedPRState.from_dict(column.initialize(**spec)["state"])
    assert len(cs.phase_roots) == 3
    h = cs.interface_heights_m[0]
    pipe = PRTerrainPipe(
        "line",
        "pickup",
        "receiver",
        0.001,
        (0.1, 0.1),
        (h, h - 0.005, h),
        (1, 1),
        drift_flux=closure,
    )
    model = build_pr_surface_network(
        f,
        [PREquilibriumNode("receiver", 0.001, elevation_m=h)],
        [pipe],
        columns={"separator": column},
        column_terminals=[PRColumnTerminal("pickup", "separator", h)],
        dense_limit=1,
        initial_step_s=1e-8,
        maximum_step_s=1e-6,
    )
    ts = initialize_grouped(model.transport, {"receiver": spec, "line": spec})
    old = model.initialize(ts, {"separator": cs})
    assert all(len(s.phase_roots) == 3 for _, s in ts.nodes)
    commands = {**model.transport.default_openings, **{p.name: True for p in model.ports}}
    end, r = model._step(old, 1e-8, commands, {}, {}, {})
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    assert all(len(s.phase_roots) == 3 for _, s in end.transport.nodes)


@pytest.mark.parametrize("power", [1e5, -1e5])
def test_moving_phase_interface_changes_finite_aperture_flux_in_joint_solve(power):
    model, old, commands = setup(slip=True)
    end, report = model._step(
        old, 1e-5, commands, {"separator": power}, {"separator": 300.0 if power > 0 else 170.0}, {}
    )
    before = dict(old.columns)["separator"]
    after = dict(end.columns)["separator"]
    assert len(before.phase_roots) == len(after.phase_roots) == 2
    port = report["columns"]["ports"]["line__column_left"]
    assert sum(port["area_fractions"]) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 < q < 1.0 for q in port["area_fractions"])
    assert abs(port["area_fractions"][0] - 0.5) > 1e-6
    assert after.interface_heights_m != before.interface_heights_m
    assert report["total_heat_j"] == pytest.approx(power * 1e-5, abs=1e-8)
    assert max(map(abs, report["conservation_residuals"])) < 1e-5


def test_column_pipe_and_finite_pump_rotor_solve_together_without_pressure_overwrite():
    from test_surface_pr_pump_connection import setup as equipment

    from surface.pr_surface_builder import initialize_pr_surface_network

    base, _, initial = equipment(driven=True, dense_limit=1)
    column = StratifiedPRColumn(
        base.fluid,
        SeparatorGeometry(0.1, 0.1, "vertical"),
        elevation_m=-0.05,
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    pipe = PREquilibriumPipe("suction", "pickup", "feed", 0.2, 0.001, 2)
    model = build_pr_surface_network(
        base.fluid,
        base.nodes,
        [pipe],
        columns={"separator": column},
        column_terminals=[PRColumnTerminal("pickup", "separator", 0.05)],
        pump_connections=base.pump_connections,
        drives=base.drives,
        dense_limit=1,
        initial_step_s=1e-7,
        maximum_step_s=1e-5,
    )
    spec = initial["feed"]
    state = initialize_pr_surface_network(model, {**initial, "suction": spec}, {"separator": spec})
    commands = {**model.transport.default_openings, **{p.name: True for p in model.ports}}
    end, r = model._step(state, 1e-7, commands, {}, {}, {})
    assert r["transport"]["pump_connections"]["booster"]["shaft_power_w"] > 0.0
    assert dict(end.transport.rotors)["booster"] != dict(state.transport.rotors)["booster"]
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_terrain_dictionary_roundtrip_preserves_declared_closure_and_mesh():
    from dataclasses import asdict

    closure = ZuberFindlayClosure(1.0, 0.01)
    pipe = PRTerrainPipe(
        "line", "a", "b", 0.01, (1.0, 1.0), (0.0, -0.1, 0.0), (2, 2), drift_flux=closure
    )
    restored = PRTerrainPipe(**json.loads(json.dumps(asdict(pipe))))
    assert restored == pipe
    assert restored.cell_routes() == pipe.cell_routes()
    assert restored.split_at_levels([-0.05]).drift_flux == closure


def test_logical_terminal_cannot_alias_a_real_pipe_cell():
    base, _, _ = setup()
    column = base.columns["separator"]
    h = base.ports[0].height_m
    pipe = PREquilibriumPipe("line", "line__0000", "receiver", 0.2, 0.0005, 2)
    with pytest.raises(ValueError, match="collides with a generated"):
        build_pr_surface_network(
            column.fluid,
            [PREquilibriumNode("receiver", 0.001, elevation_m=h)],
            [pipe],
            columns={"separator": column},
            column_terminals=[PRColumnTerminal("line__0000", "separator", h)],
        )


def test_resolved_separator_can_be_the_only_physical_graph_block():
    from surface.pr_resolved_separator import PRResolvedSeparator

    f = binary()
    spec = PRResolvedSeparator("separator", 0.01, 0.02, 2, 1e-6)
    model = build_pr_surface_network(f, [], [], resolved_separators=[spec], gravity_m_s2=9.80665)
    assert set(model.transport.names) == {spec.cell_name(0), spec.cell_name(1)}


def test_cross_block_thermal_contact_is_added_after_all_physical_nodes_exist():
    from surface.pr_heat_link import PRHeatLink
    from surface.pr_resolved_separator import PRResolvedSeparator
    from surface.pr_surface_builder import initialize_pr_surface_network

    f = binary()
    separator = PRResolvedSeparator("separator", 0.01, 0.02, 2, 1e-6)
    vessel = PREquilibriumNode("hot", 0.001, elevation_m=separator.cell_elevation_m(0))
    contact = PRHeatLink("heater", "hot", separator.cell_name(0), 0.01)
    model = build_pr_surface_network(
        f,
        [vessel],
        [],
        resolved_separators=[separator],
        heat_links=[contact],
        gravity_m_s2=9.80665,
        dense_limit=1,
        initial_step_s=1e-6,
        maximum_step_s=1e-4,
    )
    common = {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]}
    state = initialize_pr_surface_network(
        model, {"hot": {**common, "temperature_k": 330.0}, "separator": common}, {}
    )
    _, r = model._step(state, 1e-6, model.transport.default_openings, {}, {}, {})
    assert r["transport"]["heat_links"]["heater"]["heat_rate_w"] > 0.0
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
