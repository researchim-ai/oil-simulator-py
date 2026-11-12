"""One joint open graph: supported vessels, distributed pipes and finite column ports."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from test_surface_compositional_separator import binary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_pr_open_gravity_transport import audit

from surface.component_energy import ComponentEnergyStream
from surface.pr_equilibrium_transport import PREquilibriumNode
from surface.pr_gravity_transport import PRGravityTransportState
from surface.pr_open_gravity_transport import PROpenGravityTransportNetwork
from surface.pr_open_transport import PRTransportFeed, PRTransportPressurePort
from surface.pr_surface_builder import PRColumnTerminal, build_pr_surface_network
from surface.pr_transport_builder import PREquilibriumPipe, initialize_grouped
from surface.separator_geometry import SeparatorGeometry
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
from surface.vessel_wall import VesselWall

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def build(device="cpu", *, temperature=320.0, reverse=False, sparse=False, diameter=0.0005):
    f = binary(device)
    c = StratifiedPRColumn(
        f,
        SeparatorGeometry(0.1, 0.1, "vertical"),
        elevation_m=-0.05,
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    col = StratifiedPRState.from_dict(c.initialize(1e6, temperature, [0.5, 0.5])["state"])
    h = 0.05
    if col.interface_heights_m:
        # Apertures straddle the actual initial interface, not a fixed phase selector.
        h = col.interface_heights_m[0]
        # Preserve shared terminal elevation exactly in this standalone geometry.
    z = c.elevation_m + h
    graph = build_pr_surface_network(
        f,
        [
            PREquilibriumNode("inlet", 0.1, elevation_m=z),
            PREquilibriumNode("export", 0.1, elevation_m=z),
        ],
        [
            PREquilibriumPipe("supply", "inlet", "pickup_in", 0.1, diameter, 2),
            PREquilibriumPipe("delivery", "pickup_out", "export", 0.1, diameter, 2),
        ],
        columns={"separator": c},
        column_terminals=[
            PRColumnTerminal("pickup_in", "separator", h),
            PRColumnTerminal("pickup_out", "separator", h),
        ],
        column_walls={"separator": VesselWall(1000.0, 2.0, 1.0, 310.0)},
        initial_step_s=1e-6,
        maximum_step_s=1e-5,
        dense_limit=32 if sparse else 768,
    )
    spec = {
        "pressure_pa": c.sample(col, h, side="below")["pressure_pa"],
        "temperature_k": temperature,
        "mole_fractions": [0.5, 0.5],
        "velocity_m_s": 0.0,
    }
    initial = initialize_grouped(
        graph.transport, {name: spec for name in ["inlet", "export", "supply", "delivery"]}
    )
    state = graph.initialize(initial, {"separator": col})
    stream = ComponentEnergyStream.from_pt(
        f,
        [0.001, 0.002],
        1.01e6,
        temperature + 1,
        elevation_m=z,
        gravity_m_s2=graph.transport.gravity_m_s2,
    )
    boundary = PRTransportPressurePort(
        "boundary",
        "export",
        1.01e6 if reverse else 0.99e6,
        temperature,
        (0.6, 0.4),
        1e-9,
        allow_reverse=reverse,
    )
    return PROpenGravityTransportNetwork(
        graph, [PRTransportFeed("feed", "inlet", stream)], [boundary]
    ), state


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reverse", [False, True])
def test_joint_open_gravitational_graph_and_independent_profile_audit(device, reverse, tmp_path):
    model, old = build(device, reverse=reverse)
    result = model.advance(old, 1e-6)
    check = audit(model.to_dict(), old.to_dict(), result)
    assert check["accepted"]
    assert all(
        step["columns"]["material"]["ports"]["boundary"]["volume_rate_m3_s"] < 0
        if reverse
        else step["columns"]["material"]["ports"]["boundary"]["volume_rate_m3_s"] > 0
        for step in result["steps"]
    )
    assert tuple(result["state"]["columns"]["separator"]["component_moles"]) != tuple(
        dict(old.columns)["separator"].component_moles
    )
    bad = copy.deepcopy(result)
    bad["steps"][0]["columns"]["material"]["ports"]["boundary"]["component_rates_mol_s"][0] *= 1.1
    with pytest.raises(ValueError):
        audit(model.to_dict(), old.to_dict(), bad)
    (tmp_path / f"gravity-{device}-{reverse}.json").write_text(
        json.dumps(
            {"model": model.to_dict(), "initial": old.to_dict(), "result": result, "audit": check},
            indent=2,
        )
        + "\n"
    )


def test_two_phase_finite_apertures_sparse_graph_and_restart():
    m, s = build(temperature=196.2, sparse=True)
    assert len(dict(s.columns)["separator"].phase_roots) == 2
    result = m.advance(s, 1e-6)
    assert audit(m.to_dict(), s.to_dict(), result)["accepted"]
    other = PROpenGravityTransportNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    state = PRGravityTransportState.from_dict(result["state"])
    a = m.advance(state, 1e-6, port_openings={"boundary": 0.0})
    b = other.advance(
        PRGravityTransportState.from_dict(json.loads(json.dumps(state.to_dict()))),
        1e-6,
        port_openings={"boundary": 0.0},
    )
    assert a == b
    assert all(
        v["columns"]["material"]["ports"]["boundary"]["component_rates_mol_s"] == [0.0, 0.0]
        for v in a["steps"]
    )
    assert audit(m.to_dict(), state.to_dict(), a, port_openings={"boundary": 0.0})["accepted"]


def test_isolation_and_bad_boundary_pressure_preserve_every_old_stock():
    m, s = build()
    before = s.to_dict()
    cmd = {p.name: False for p in m.graph.ports}
    r = m.advance(s, 1e-6, openings=cmd, port_openings={"boundary": 0.0})
    assert tuple(r["state"]["columns"]["separator"]["component_moles"]) == pytest.approx(
        dict(s.columns)["separator"].component_moles, abs=1e-14
    )
    assert audit(m.to_dict(), before, r, openings=cmd, port_openings={"boundary": 0.0})["accepted"]
    assert s.to_dict() == before
    with pytest.raises(ValueError, match="Unknown"):
        m.advance(s, 1e-6, port_openings={"typo": 0.0})


def test_empty_column_registry_reduces_to_existing_open_transport_without_extra_stocks():
    from test_surface_pr_open_transport import build as ordinary

    from surface.pr_gravity_transport import PRGravityTransportNetwork

    base, s = ordinary()
    graph = PRGravityTransportNetwork(base.transport, {}, [])
    m = PROpenGravityTransportNetwork(graph, base.feeds, base.pressure_ports)
    state = graph.initialize(s, {})
    result = m.advance(state, 1e-5)
    direct = base.advance(s, 1e-5)
    assert result["state"]["columns"] == {} and result["state"]["walls"] == {}
    assert result["state"]["transport"] == direct["state"]
    assert audit(m.to_dict(), state.to_dict(), result)["accepted"]


def test_material_boundary_and_gravity_jacobian_has_both_cross_blocks():
    from surface.pr_gravity_transport import _ColumnCoupling
    from surface.pr_open_transport import _BoundaryProperties, _OpenCoupling

    m, s = build()
    g, t = m.graph, m.transport
    commands = {**t.default_openings, **{p.name: True for p in g.ports}}
    material = _OpenCoupling(
        m, _BoundaryProperties(m, {"boundary": 1.0}), s.transport, 1e-6, commands
    )
    c = _ColumnCoupling(g, s, 1e-6, 2, commands, {}, {}, material=material)
    seeds = [t._seed(i, v) for i, (_, v) in enumerate(s.transport.nodes)]
    y, lo, hi, base, offsets, _, _ = t._problem(
        s.transport,
        seeds,
        1e-6,
        commands,
        [0.0] * len(t.names),
        external_ports=c.external_ports,
        active_override=c.active,
        guesses_override=c.ordinary_guesses,
    )
    y, _, _, ev, _ = c.prepare(y, lo, hi, base, offsets, 1e-6, None)
    jac = torch.func.jacfwd(lambda v: ev(v)[0])(y)
    direction = torch.linspace(-0.0001, 0.0001, y.numel(), dtype=y.dtype)
    eps = 1e-5
    fd = (ev(y + eps * direction)[0] - ev(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-5
    assert float(jac[: c.size, c.size :].abs().max()) > 1e-10
    assert float(jac[c.size :, : c.size].abs().max()) > 1e-10


def test_open_gravity_pump_rotor_and_external_streams_share_one_energy_ledger():
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
    graph = build_pr_surface_network(
        base.fluid,
        base.nodes,
        [PREquilibriumPipe("suction", "pickup", "feed", 0.2, 0.001, 2)],
        columns={"separator": column},
        column_terminals=[PRColumnTerminal("pickup", "separator", 0.05)],
        pump_connections=base.pump_connections,
        drives=base.drives,
        dense_limit=1,
        initial_step_s=1e-7,
        maximum_step_s=1e-5,
    )
    state = initialize_pr_surface_network(
        graph, {**initial, "suction": initial["feed"]}, {"separator": initial["feed"]}
    )
    si = dict(state.transport.nodes)["feed"]
    z = [v / sum(si.component_moles) for v in si.component_moles]
    stream = ComponentEnergyStream.from_pt(
        base.fluid,
        [1e-4 * v for v in z],
        si.pressure_pa * 1.01,
        si.temperature_k,
        gravity_m_s2=graph.transport.gravity_m_s2,
    )
    m = PROpenGravityTransportNetwork(
        graph,
        [PRTransportFeed("source", "feed", stream)],
        [PRTransportPressurePort("export", "return", 0.99e6, 350.0, tuple(z), 1e-10)],
    )
    result = m.advance(state, 1e-7)
    assert result["total_work_j"] > 0
    assert all(
        r["transport"]["pump_connections"]["booster"]["shaft_power_w"] > 0 for r in result["steps"]
    )
    assert audit(m.to_dict(), state.to_dict(), result)["accepted"]


@pytest.mark.parametrize("device", DEVICES)
def test_open_material_flow_and_heat_move_interface_out_of_finite_aperture(device, tmp_path):
    from surface.pr_open_transport import _BoundaryProperties, _OpenCoupling

    m, state = build(device, temperature=196.2, diameter=2e-5)
    g, t = m.graph, m.transport
    dt = 0.5
    commands = {**t.default_openings, **{p.name: True for p in g.ports}}
    material = _OpenCoupling(
        m, _BoundaryProperties(m, {"boundary": 1.0}), state.transport, dt, commands
    )
    end, raw = g._step(
        state, dt, commands, {"separator": 100.0}, {"separator": 300.0}, {}, _material=material
    )
    before = dict(state.columns)["separator"]
    after = dict(end.columns)["separator"]
    assert len(before.phase_roots) == len(after.phase_roots) == 2
    assert all(abs(before.interface_heights_m[0] - p.height_m) < 1e-12 for p in g.ports)
    assert all(after.interface_heights_m[0] < p.height_m - p.diameter_m / 2 for p in g.ports)
    for p in raw["columns"]["ports"].values():
        assert p["area_fractions"][0] < 1e-10 and p["area_fractions"][1] > 1 - 1e-10
    assert raw["columns"]["material"]["ports"]["feed"]["component_rates_mol_s"][0] > 0
    assert raw["columns"]["material"]["ports"]["boundary"]["volume_rate_m3_s"] > 0
    result = {
        "state": end.to_dict(),
        "steps": [raw],
        "openings": commands,
        "loss_coefficients": {},
        "total_heat_j": raw["total_heat_j"],
        "total_work_j": raw["total_work_j"],
        "conservation_residuals": raw["conservation_residuals"],
        "material_component_moles": raw["material_component_moles"],
        "material_energy_j": raw["material_energy_j"],
        "open_model_id": m.model_id,
        "port_openings": {"boundary": 1.0},
        "boundary_component_moles": raw["material_component_moles"],
        "boundary_energy_j": raw["material_energy_j"],
    }
    check = audit(
        m.to_dict(),
        state.to_dict(),
        result,
        heat_w={"separator": 100.0},
        utility_temperature_k={"separator": 300.0},
    )
    assert check["accepted"]
    (tmp_path / f"crossing-{device}.json").write_text(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": state.to_dict(),
                "result": result,
                "audit": check,
                "qualification": "Nonadaptive synthetic geometric/thermodynamic step with finite aperture phase-area crossing and actual material flows. No capillary tension and no temporal-convergence claim for the half-second transient.",
            },
            indent=2,
        )
        + "\n"
    )
