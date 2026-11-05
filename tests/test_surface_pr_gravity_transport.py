"""Resolved gravitational aperture coupled to inertial multi-phase transport."""

import json
import math

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork
from surface.pr_gravity_transport import (
    PRColumnPort,
    PRGravityTransportNetwork,
    PRGravityTransportState,
    _ColumnCoupling,
)
from surface.separator_geometry import SeparatorGeometry
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState


def setup(temperature=320.0, reverse=False, isolated=False, diameter=0.0005):
    f = binary()
    c = StratifiedPRColumn(
        f, SeparatorGeometry(0.1, 0.1, "vertical"), quadrature_order=2, maximum_quadrature_order=8
    )
    cs = StratifiedPRState.from_dict(c.initialize(1e6, temperature, [0.5, 0.5])["state"])
    h = cs.interface_heights_m[0] if cs.interface_heights_m else 0.05
    node = PREquilibriumNode(
        "pipe",
        math.pi * diameter**2 / 4 * 0.1,
        "pipe_cell",
        elevation_m=h,
        length_m=0.1,
        diameter_m=diameter,
    )
    t = PREquilibriumTransportNetwork(
        f,
        [node],
        gravity_m_s2=c.gravity_m_s2,
        dense_limit=768,
        initial_step_s=1e-6,
        maximum_step_s=1e-5,
    )
    ps = t.initialize(
        {
            "pipe": {
                "pressure_pa": 1010000.0 if reverse else 990000.0,
                "temperature_k": temperature,
                "mole_fractions": [0.5, 0.5],
            }
        }
    )
    m = PRGravityTransportNetwork(
        t, {"separator": c}, [PRColumnPort("port", "separator", "pipe", h, diameter)]
    )
    s = m.initialize(ps, {"separator": cs})
    return m, s, {"port": not isolated}


def test_gravitational_aperture_and_pipe_share_component_energy_balances():
    m, s, commands = setup()
    new, r = m._step(s, 1e-6, commands, {}, {}, {})
    assert r["columns"]["ports"]["port"]["flux"][0] > 0
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    assert sum(new.columns[0][1].component_moles) < sum(s.columns[0][1].component_moles)
    assert r["entropy_production_j_k"] >= -1e-6
    restored = PRGravityTransportNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(PRGravityTransportState.from_dict(json.loads(json.dumps(new.to_dict()))))


def test_gravitational_joint_jacobian_no_frozen_column_pressure():
    m, s, commands = setup()
    t = m.transport
    seeds = [t._seed(i, v) for i, (_, v) in enumerate(s.transport.nodes)]
    coupling = _ColumnCoupling(m, s, 1e-6, 2, commands, {}, {})
    y, lo, hi, base, offsets, _, _ = t._problem(
        s.transport, seeds, 1e-6, {}, [0.0], external_ports=coupling.external_ports
    )
    y, lo, hi, evaluate, _ = coupling.prepare(y, lo, hi, base, offsets, 1e-6, None)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-0.001, 0.001, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-5
    assert bool((jac[: offsets[-1], offsets[-1] :].abs() > 1e-12).any())


@pytest.mark.parametrize("reverse,isolated", [(True, False), (False, True)])
def test_reverse_and_isolation(reverse, isolated):
    m, s, commands = setup(reverse=reverse, isolated=isolated)
    new, r = m._step(s, 1e-6, commands, {}, {}, {})
    flux = r["columns"]["ports"]["port"]["flux"]
    if isolated:
        assert flux[:-2] == [0.0, 0.0]
        assert flux[-1] == 0.0
        assert new.columns[0][1].component_moles == pytest.approx(
            s.columns[0][1].component_moles, abs=1e-12
        )
    else:
        assert flux[0] < 0
        assert sum(new.columns[0][1].component_moles) > sum(s.columns[0][1].component_moles)


def test_two_phase_column_resolved_aperture_feeds_two_phase_pipe():
    m, s, commands = setup(temperature=196.2)
    assert len(s.columns[0][1].phase_roots) == 2
    new, report = m._step(s, 1e-7, commands, {}, {}, {})
    weights = report["columns"]["ports"]["port"]["area_fractions"]
    assert sum(weights) == pytest.approx(1.0, abs=1e-10)
    assert all(0 < v < 1 for v in weights)
    assert max(map(abs, report["conservation_residuals"])) < 1e-6
    assert new.transport.nodes[0][1].total_energy_j != s.transport.nodes[0][1].total_energy_j


def test_coupled_public_interval_restart_and_failed_interval_are_atomic():
    import copy
    from dataclasses import replace

    from surface.liquid import HydraulicConvergenceError

    m, s, commands = setup()
    original = copy.deepcopy(s.to_dict())
    report = m.advance(s, 1e-6, openings=commands)
    end = PRGravityTransportState.from_dict(report["state"])
    assert end.time_s == 1e-6
    restored = PRGravityTransportNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    rs = PRGravityTransportState.from_dict(json.loads(json.dumps(end.to_dict())))
    a = m.advance(end, 1e-6, openings={"port": False})
    b = restored.advance(rs, 1e-6, openings={"port": False})
    assert a == b
    assert s.to_dict() == original
    with pytest.raises(ValueError):
        m.advance(s, 1e-6, openings={"bad": True})
    m.transport.maximum_attempts = 1
    m.transport.model_id = __import__("surface.operation", fromlist=["_digest"])._digest(
        m.transport.to_dict()
    )
    ts = replace(s.transport, model_id=m.transport.model_id)
    m.model_id = __import__("surface.operation", fromlist=["_digest"])._digest(m.to_dict())
    limited = replace(s, transport=ts, model_id=m.model_id)
    with pytest.raises(HydraulicConvergenceError):
        m.advance(limited, 1e-4)
    assert s.to_dict() == original


def test_gravitational_column_phase_birth_and_disappearance_keep_heat_and_stocks():
    from surface.peng_robinson import R

    f = binary()
    pure = f.components[1]
    p = pure.saturation_at_temperature(200.0)["pressure_pa"]
    liquid = pure.state(p, 200.0, "liquid", derivatives=False)
    vapor = pure.state(p, 200.0, "vapor", derivatives=False)
    mw = pure.molar_mass_kg_mol
    vm = mw * (0.4 * liquid["specific_volume_m3_kg"] + 0.6 * vapor["specific_volume_m3_kg"])
    target = (
        10
        * mw
        * (
            0.4 * (liquid["enthalpy_j_kg"] - p * liquid["specific_volume_m3_kg"])
            + 0.6 * (vapor["enthalpy_j_kg"] - p * vapor["specific_volume_m3_kg"])
        )
    )
    attraction, _ = pure._a(220.0)
    p0 = R * 220 / (vm - pure.b) - attraction / (vm * vm + 2 * pure.b * vm - pure.b**2)
    geometry = SeparatorGeometry(0.1, 10 * vm / (math.pi * 0.1**2 / 4), "vertical")
    column = StratifiedPRColumn(f, geometry, quadrature_order=2, maximum_quadrature_order=8)
    cs = StratifiedPRState.from_dict(column.initialize(p0, 220.0, [0.0, 1.0])["state"])
    node = PREquilibriumNode(
        "pipe", math.pi * 0.001**2 / 4 * 0.1, "pipe_cell", length_m=0.1, diameter_m=0.001
    )
    t = PREquilibriumTransportNetwork(
        f,
        [node],
        gravity_m_s2=column.gravity_m_s2,
        dense_limit=768,
        initial_step_s=1.0,
        maximum_step_s=1.0,
    )
    ts = t.initialize(
        {"pipe": {"pressure_pa": p0, "temperature_k": 220.0, "mole_fractions": [0.0, 1.0]}}
    )
    m = PRGravityTransportNetwork(t, {"column": column}, [])
    s = m.initialize(ts, {"column": cs})
    mass = sum(n * c.molar_mass_kg_mol for n, c in zip(cs.component_moles, f.components))
    target += mass * column.gravity_m_s2 * geometry.height_m / 2
    cooled, r = m._step(s, 1.0, {}, {"column": target - cs.total_energy_j}, {"column": 170.0}, {})
    assert len(cooled.columns[0][1].phase_roots) == 2
    assert cooled.columns[0][1].component_moles == pytest.approx(cs.component_moles, abs=1e-10)
    assert cooled.columns[0][1].total_energy_j == pytest.approx(target, abs=1e-5)
    heated, r = m._step(
        cooled, 1.0, {}, {"column": cs.total_energy_j - target}, {"column": 300.0}, {}
    )
    assert len(heated.columns[0][1].phase_roots) == 1
    assert heated.columns[0][1].temperature_k == pytest.approx(cs.temperature_k, abs=1e-5)
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_independent_gravity_auditor_rejects_corrupted_energy(monkeypatch):
    import copy
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    runner = load("run_pr_gravity_transport")
    checker = load("check_pr_gravity_transport")
    config = json.loads((root / "configs/surface_pr_gravity_transport.json").read_text())
    config["schedule"] = [{"duration_s": 1e-7}]
    report = json.loads(json.dumps(runner.run(config)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime state solver called by independent auditor")

    monkeypatch.setattr(PRGravityTransportNetwork, "inspect", forbidden)
    monkeypatch.setattr(StratifiedPRColumn, "sample", forbidden)
    monkeypatch.setattr(StratifiedPRColumn, "_problem", forbidden)
    monkeypatch.setattr(PREquilibriumTransportNetwork, "_decode", forbidden)
    assert checker.audit(config, report)["accepted"]
    broken = copy.deepcopy(report)
    broken["intervals"][0]["steps"][0]["state"]["columns"]["separator"]["total_energy_j"] += 1.0
    with pytest.raises(ValueError):
        checker.audit(config, broken)


def test_one_column_recycle_two_ports_one_common_state_and_wall():
    from dataclasses import replace

    from surface.pr_equilibrium_transport import PREquilibriumLink
    from surface.vessel_wall import VesselWall

    original, s, _ = setup()
    t0 = original.transport
    node = t0.nodes[0]
    nodes = [replace(node, name="a"), replace(node, name="b")]
    t = PREquilibriumTransportNetwork(
        t0.fluid,
        nodes,
        [PREquilibriumLink("middle", "a", "b", node.diameter_m)],
        gravity_m_s2=t0.gravity_m_s2,
        dense_limit=768,
    )
    ps = t.initialize(
        {
            n.name: {
                "pressure_pa": 1e6,
                "temperature_k": 320.0,
                "mole_fractions": [0.5, 0.5],
                "velocity_m_s": 0.1,
            }
            for n in nodes
        }
    )
    col = original.columns["separator"]
    h = original.ports[0].height_m
    m = PRGravityTransportNetwork(
        t,
        {"separator": col},
        [
            PRColumnPort("inlet", "separator", "a", h, node.diameter_m),
            PRColumnPort("outlet", "separator", "b", h, node.diameter_m, "right"),
        ],
        walls={"separator": VesselWall(1000.0, 10.0, 1.0)},
    )
    state = m.initialize(ps, dict(s.columns), {"separator": 300.0})
    out, r = m._step(state, 1e-7, {"middle": True, "inlet": True, "outlet": True}, {}, {}, {})
    assert len(out.columns) == 1
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    assert r["columns"]["ports"]["inlet"]["flux"][0] > 0
    assert r["columns"]["ports"]["outlet"]["flux"][0] < 0
    assert out.walls[0][1] > 300.0
    assert r["entropy_production_j_k"] >= -1e-6


@pytest.mark.parametrize("matched", [False, True])
def test_heating_moves_interface_completely_out_of_gravitational_aperture(matched):
    m, s, commands = setup(temperature=196.2, diameter=0.00002)
    if matched:
        col = m.columns["separator"]
        cs = dict(s.columns)["separator"]
        local = col.sample(cs, m.ports[0].height_m, side="below")
        ps = m.transport.initialize(
            {
                "pipe": {
                    "pressure_pa": local["pressure_pa"],
                    "temperature_k": cs.temperature_k,
                    "mole_fractions": [0.5, 0.5],
                }
            }
        )
        s = m.initialize(ps, dict(s.columns))
    first, r0 = m._step(s, 1e-7, commands, {}, {}, {})
    initial = r0["columns"]["ports"]["port"]["area_fractions"]
    assert all(0 < v < 1 for v in initial)
    final, r = m._step(first, 0.5, commands, {"separator": 100.0}, {"separator": 300.0}, {})
    weights = r["columns"]["ports"]["port"]["area_fractions"]
    assert min(weights) < 1e-10
    assert max(weights) > 1 - 1e-10
    assert len(final.columns[0][1].phase_roots) == 2
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_synthetic_three_phase_column_and_pipe_share_finite_pickup():
    from test_surface_compositional_separator import ternary

    f = ternary()
    c = StratifiedPRColumn(
        f, SeparatorGeometry(0.1, 0.1, "vertical"), quadrature_order=2, maximum_quadrature_order=8
    )
    cs = StratifiedPRState.from_dict(c.initialize(1e6, 350.0, [0.4, 0.3, 0.3])["state"])
    assert len(cs.phase_roots) == 3
    h = cs.interface_heights_m[0]
    diameter = 0.0005
    node = PREquilibriumNode(
        "pipe",
        math.pi * diameter**2 / 4 * 0.1,
        "pipe_cell",
        elevation_m=h,
        length_m=0.1,
        diameter_m=diameter,
    )
    t = PREquilibriumTransportNetwork(f, [node], gravity_m_s2=c.gravity_m_s2, dense_limit=768)
    ts = t.initialize(
        {
            "pipe": {
                "pressure_pa": 990000.0,
                "temperature_k": 350.0,
                "mole_fractions": [0.4, 0.3, 0.3],
            }
        }
    )
    m = PRGravityTransportNetwork(
        t, {"separator": c}, [PRColumnPort("port", "separator", "pipe", h, diameter)]
    )
    state = m.initialize(ts, {"separator": cs})
    new, r = m._step(state, 1e-7, {"port": True}, {}, {}, {})
    assert len(new.columns[0][1].phase_roots) == 3
    weights = r["columns"]["ports"]["port"]["area_fractions"]
    assert len(weights) == 3 and weights[-1] == 0.0
    assert sum(weights) == pytest.approx(1.0, abs=1e-10)
    assert all(0 < v < 1 for v in weights[:-1])
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


@pytest.mark.parametrize("reverse,isolated", [(False, False), (True, False), (False, True)])
def test_exactly_absent_component_can_enter_column_or_pipe(reverse, isolated):
    f = binary()
    c = StratifiedPRColumn(
        f, SeparatorGeometry(0.1, 0.1, "vertical"), quadrature_order=2, maximum_quadrature_order=8
    )
    cs = StratifiedPRState.from_dict(c.initialize(1e6, 320.0, [1.0, 0.0])["state"])
    diameter = 0.0005
    node = PREquilibriumNode(
        "pipe",
        math.pi * diameter**2 / 4 * 0.1,
        "pipe_cell",
        elevation_m=0.05,
        length_m=0.1,
        diameter_m=diameter,
    )
    t = PREquilibriumTransportNetwork(f, [node], gravity_m_s2=c.gravity_m_s2, dense_limit=768)
    ts = t.initialize(
        {
            "pipe": {
                "pressure_pa": 990000.0 if reverse else 1010000.0,
                "temperature_k": 320.0,
                "mole_fractions": [0.0, 1.0],
            }
        }
    )
    m = PRGravityTransportNetwork(
        t, {"separator": c}, [PRColumnPort("port", "separator", "pipe", 0.05, diameter)]
    )
    s = m.initialize(ts, {"separator": cs})
    new, r = m._step(s, 1e-7, {"port": not isolated}, {}, {}, {})
    if isolated:
        assert new.columns[0][1].component_moles[1] == 0.0
        assert new.transport.nodes[0][1].component_moles[0] == 0.0
    elif reverse:
        assert new.transport.nodes[0][1].component_moles[0] > 0.0
        assert new.transport.nodes[0][1].component_moles[0] == pytest.approx(
            1e-7 * r["columns"]["ports"]["port"]["flux"][0], rel=1e-5, abs=1e-20
        )
    else:
        assert new.columns[0][1].component_moles[1] > 0.0
        assert new.columns[0][1].component_moles[1] == pytest.approx(
            -1e-7 * r["columns"]["ports"]["port"]["flux"][1], rel=1e-5, abs=1e-20
        )
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_water_can_enter_initially_water_free_hydrocarbon_column():
    from test_surface_compositional_separator import ternary

    f = ternary()
    c = StratifiedPRColumn(
        f, SeparatorGeometry(0.1, 0.1, "vertical"), quadrature_order=2, maximum_quadrature_order=8
    )
    cs = StratifiedPRState.from_dict(c.initialize(1e6, 350.0, [0.6, 0.4, 0.0])["state"])
    diameter = 0.0002
    h = 0.05
    node = PREquilibriumNode(
        "pipe",
        math.pi * diameter**2 / 4 * 0.1,
        "pipe_cell",
        elevation_m=h,
        length_m=0.1,
        diameter_m=diameter,
    )
    t = PREquilibriumTransportNetwork(
        f, [node], gravity_m_s2=c.gravity_m_s2, dense_limit=768, wave_speed_bound_m_s=10000.0
    )
    ts = t.initialize(
        {
            "pipe": {
                "pressure_pa": 1010000.0,
                "temperature_k": 350.0,
                "mole_fractions": [0.0, 0.0, 1.0],
            }
        }
    )
    m = PRGravityTransportNetwork(
        t, {"separator": c}, [PRColumnPort("water", "separator", "pipe", h, diameter)]
    )
    s = m.initialize(ts, {"separator": cs})
    new, r = m._step(s, 1e-7, {"water": True}, {}, {}, {})
    assert s.columns[0][1].component_moles[-1] == 0.0
    assert new.columns[0][1].component_moles[-1] > 0.0
    assert r["columns"]["ports"]["water"]["flux"][2] < 0.0
    assert new.columns[0][1].component_moles[-1] == pytest.approx(
        -1e-7 * r["columns"]["ports"]["water"]["flux"][2], rel=1e-5, abs=1e-20
    )
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
