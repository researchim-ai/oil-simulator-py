"""HEM phase equilibrium transport: fixed stocks, finite volume and graph topology."""

import json
import math

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)


def pipe(temperature=196.2, velocity=0.0, periodic=False, **options):
    f = binary()
    a = math.pi * 0.05**2 / 4
    nodes = [
        PREquilibriumNode(str(i), a * 0.5, "pipe_cell", length_m=0.5, diameter_m=0.05)
        for i in range(2)
    ]
    links = [PREquilibriumLink("internal", "0", "1", 0.05)]
    if periodic:
        links.append(PREquilibriumLink("return", "1", "0", 0.05))
    m = PREquilibriumTransportNetwork(f, nodes, links, **options)
    s = m.initialize(
        {
            str(i): {
                "pressure_pa": 1e6,
                "temperature_k": temperature,
                "mole_fractions": [0.5, 0.5],
                "velocity_m_s": velocity,
            }
            for i in range(2)
        }
    )
    return m, s


@pytest.mark.parametrize("temperature", [320.0, 196.2])
def test_hem_closed_and_moving_periodic_uniform_state(temperature):
    m, s = pipe(temperature, 2.0, True)
    before = s.to_dict()
    out, r = m._step(s, 1e-4, {c.name: True for c in m.links}, [0.0, 0.0], [temperature] * 2)
    for (_, a), (_, b) in zip(s.nodes, out.nodes):
        assert a.component_moles == pytest.approx(b.component_moles, abs=1e-10)
        assert a.total_energy_j == pytest.approx(b.total_energy_j, abs=1e-6)
        assert a.momentum_kg_m_s == pytest.approx(b.momentum_kg_m_s, abs=1e-9)
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    assert s.to_dict() == before


def test_closed_two_phase_pipe_stays_at_rest():
    m, s = pipe()
    assert all(len(v.phase_roots) == 2 for _, v in s.nodes)
    out, r = m._step(s, 1e-4, {"internal": True}, [0.0, 0.0], [196.2] * 2)
    assert all(abs(v.momentum_kg_m_s) < 1e-9 for _, v in out.nodes)
    assert r["entropy_production_j_k"] >= -1e-6


def test_joint_phase_momentum_energy_jacobian():
    m, s = pipe(velocity=0.1)
    seeds = [m._seed(i, v) for i, (_, v) in enumerate(s.nodes)]
    y, _, _, evaluate, _, _, _ = m._problem(s, seeds, 1e-4, {"internal": True}, [0.0, 0.0])
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-0.01, 0.01, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-6


def test_pure_condensation_and_evaporation_use_conserved_heat():
    from surface.peng_robinson import R

    f = binary()
    c = f.components[1]
    p = c.saturation_at_temperature(200)["pressure_pa"]
    l = c.state(p, 200, "liquid", derivatives=False)
    v = c.state(p, 200, "vapor", derivatives=False)
    mw = c.molar_mass_kg_mol
    vm = 0.4 * l["specific_volume_m3_kg"] * mw + 0.6 * v["specific_volume_m3_kg"] * mw
    target = 10 * (
        0.4 * (l["enthalpy_j_kg"] * mw - p * l["specific_volume_m3_kg"] * mw)
        + 0.6 * (v["enthalpy_j_kg"] * mw - p * v["specific_volume_m3_kg"] * mw)
    )
    attraction, _ = c._a(220)
    p0 = R * 220 / (vm - c.b) - attraction / (vm * vm + 2 * c.b * vm - c.b * c.b)
    area = math.pi * 0.05**2 / 4
    volume = 10 * vm
    node = PREquilibriumNode("cell", volume, "pipe_cell", length_m=volume / area, diameter_m=0.05)
    m = PREquilibriumTransportNetwork(f, [node], initial_step_s=1.0, maximum_step_s=1.0)
    s = m.initialize(
        {"cell": {"pressure_pa": p0, "temperature_k": 220.0, "mole_fractions": [0.0, 1.0]}}
    )
    assert len(s.nodes[0][1].phase_roots) == 1
    old_energy = s.nodes[0][1].total_energy_j
    r = m.advance(
        s, 1.0, heat_w={"cell": target - old_energy}, utility_temperature_k={"cell": 170.0}
    )
    s = PREquilibriumTransportState.from_dict(r["state"])
    assert len(s.nodes[0][1].phase_roots) == 2
    assert s.nodes[0][1].temperature_k == pytest.approx(200.0, abs=1e-5)
    assert s.nodes[0][1].pressure_pa == pytest.approx(p, rel=1e-7)
    assert s.nodes[0][1].total_energy_j == pytest.approx(target, abs=1e-5)
    r = m.advance(
        s, 1.0, heat_w={"cell": old_energy - target}, utility_temperature_k={"cell": 300.0}
    )
    final = PREquilibriumTransportState.from_dict(r["state"])
    assert len(final.nodes[0][1].phase_roots) == 1
    assert final.nodes[0][1].temperature_k == pytest.approx(220.0, abs=1e-5)


def test_density_slip_periodic_flow_conserves_both_phase_kinetic_energies():
    f = binary()
    area = math.pi * 0.05**2 / 4
    nodes = [
        PREquilibriumNode(
            str(i),
            area * 0.5,
            "pipe_cell",
            length_m=0.5,
            diameter_m=0.05,
            distribution_strength=0.01,
            dynamic_viscosity_pa_s=0.01,
        )
        for i in range(2)
    ]
    links = [PREquilibriumLink("a", "0", "1", 0.05), PREquilibriumLink("b", "1", "0", 0.05)]
    m = PREquilibriumTransportNetwork(f, nodes, links)
    s = m.initialize(
        {
            str(i): {
                "pressure_pa": 1e6,
                "temperature_k": 196.2,
                "mole_fractions": [0.5, 0.5],
                "velocity_m_s": 2.0,
            }
            for i in range(2)
        }
    )
    d = m._state_data(0, s.nodes[0][1])
    masses = d["phase_n"] * (d["x"] * f.molar_masses).sum(-1)
    assert abs(float((masses * d["drift"]).sum())) < 1e-12
    assert float(d["phase_velocity"].max() - d["phase_velocity"].min()) > 0.001
    old_kinetic = float(d["kinetic"])
    out, _r = m._step(s, 0.001, {c.name: True for c in links}, [0.0, 0.0], [196.2] * 2)
    new = m._state_data(0, out.nodes[0][1])
    assert float(new["kinetic"]) < old_kinetic
    assert sum(v.total_energy_j for _, v in out.nodes) == pytest.approx(
        sum(v.total_energy_j for _, v in s.nodes), abs=1e-6
    )
    assert float(new["internal"]) > float(d["internal"])


def test_branched_ring_and_parallel_pipes_share_one_conservative_solve():
    from surface.pr_transport_builder import (
        PREquilibriumPipe,
        build_equilibrium_transport,
        initialize_grouped,
    )

    f = binary()
    vessels = [PREquilibriumNode(n, 0.002) for n in ("a", "b", "c")]
    paths = [
        PREquilibriumPipe("ab", "a", "b", 1.0, 0.02, 2),
        PREquilibriumPipe("parallel", "a", "b", 1.0, 0.02, 2),
        PREquilibriumPipe("bc", "b", "c", 1.0, 0.02, 2),
        PREquilibriumPipe("ca", "c", "a", 1.0, 0.02, 2),
    ]
    m = build_equilibrium_transport(f, vessels, paths)
    initial = {
        name: {
            "pressure_pa": 1e6 + (100.0 if name == "a" else -100.0 if name == "b" else 0.0),
            "temperature_k": 196.2,
            "mole_fractions": [0.5, 0.5],
        }
        for name in [*("a", "b", "c"), *m.pipe_paths]
    }
    s = initialize_grouped(m, initial)
    out, r = m._step(
        s, 1e-4, {c.name: True for c in m.links}, [0.0] * len(m.nodes), [196.2] * len(m.nodes)
    )
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    assert r["links"]["ab__face_0000"]["flux"][0] > 0
    assert r["links"]["ca__face_0002"]["flux"][0] < 0
    restored = PREquilibriumTransportNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(PREquilibriumTransportState.from_dict(json.loads(json.dumps(out.to_dict()))))


def test_component_arrives_from_exact_zero_and_closed_link_preserves_zero():
    f = binary()
    nodes = [PREquilibriumNode("a", 0.001), PREquilibriumNode("b", 0.001)]
    m = PREquilibriumTransportNetwork(f, nodes, [PREquilibriumLink("feed", "b", "a", 0.01)])
    s = m.initialize(
        {
            "a": {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [1.0, 0.0]},
            "b": {"pressure_pa": 1001000.0, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]},
        }
    )
    closed, _ = m._step(s, 1e-4, {"feed": False}, [0.0, 0.0], [320.0] * 2)
    assert dict(closed.nodes)["a"].component_moles[1] == 0.0
    out, r = m._step(s, 1e-4, {"feed": True}, [0.0, 0.0], [320.0] * 2)
    assert dict(out.nodes)["a"].component_moles[1] > 0.0
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_three_equilibrium_phases_transport_in_a_pipe():
    from test_surface_compositional_separator import ternary

    f = ternary()
    area = math.pi * 0.05**2 / 4
    nodes = [
        PREquilibriumNode(str(i), area * 0.5, "pipe_cell", length_m=0.5, diameter_m=0.05)
        for i in range(2)
    ]
    links = [PREquilibriumLink("a", "0", "1", 0.05), PREquilibriumLink("b", "1", "0", 0.05)]
    m = PREquilibriumTransportNetwork(f, nodes, links)
    s = m.initialize(
        {
            str(i): {
                "pressure_pa": 1e6,
                "temperature_k": 350.0,
                "mole_fractions": [0.4, 0.3, 0.3],
                "velocity_m_s": 0.2,
            }
            for i in range(2)
        }
    )
    assert all(len(v.phase_roots) == 3 for _, v in s.nodes)
    out, r = m._step(s, 1e-4, {c.name: True for c in links}, [0.0, 0.0], [350.0] * 2)
    assert all(len(v.phase_roots) == 3 for _, v in out.nodes)
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_pickup_area_and_donor_change_with_phase_holdup():
    from surface.phase_geometry import phase_layout
    from surface.separator_geometry import SeparatorGeometry

    f = binary()
    geo = SeparatorGeometry(0.1, 0.1, "vertical")
    base = PREquilibriumTransportNetwork(f, [PREquilibriumNode("a", geo.volume_m3, geometry=geo)])
    bs = base.initialize(
        {"a": {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]}}
    )
    bd = base._state_data(0, bs.nodes[0][1])
    layout = phase_layout(bd["phase_n"], bd["q"], geo)
    boundary = float(layout["phase_tops_m"].min())
    diameter = min(0.001, boundary * 0.5)
    nodes = [
        PREquilibriumNode("a", geo.volume_m3, geometry=geo),
        PREquilibriumNode("b", geo.volume_m3 * 10),
    ]
    link = PREquilibriumLink("pickup", "a", "b", diameter, source_height_m=boundary)
    m = PREquilibriumTransportNetwork(f, nodes, [link], initial_step_s=0.01, maximum_step_s=0.1)
    s = m.initialize(
        {
            "a": {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]},
            "b": {"pressure_pa": 990000.0, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]},
        }
    )
    _out, r = m._step(s, 0.01, {"pickup": True}, [100.0, 0.0], [300.0, 196.2])
    weights = r["links"]["pickup"]["source_area_fractions"]
    assert sum(weights) == pytest.approx(1.0, abs=1e-10)
    assert abs(weights[0] - 0.5) > 1e-5
    assert r["links"]["pickup"]["flux"][0] > 0
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_buoyancy_counterflow_and_elevation_datum_invariance():
    f = binary()
    area = math.pi * 0.05**2 / 4

    def make(offset):
        nodes = [
            PREquilibriumNode(
                str(i),
                area * 0.5,
                "pipe_cell",
                elevation_m=offset + 0.25 + 0.5 * i,
                length_m=0.5,
                diameter_m=0.05,
                elevation_change_m=0.5,
                buoyancy_relaxation_s=1e-4,
            )
            for i in range(2)
        ]
        m = PREquilibriumTransportNetwork(
            f,
            nodes,
            [PREquilibriumLink("up", "0", "1", 0.05)],
            gravity_m_s2=9.80665,
            reference_elevation_m=offset,
        )
        s = m.initialize(
            {
                str(i): {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]}
                for i in range(2)
            }
        )
        return m, s

    m, s = make(0.0)
    d = m._state_data(0, s.nodes[0][1])
    assert float(d["phase_velocity"].min()) < 0 < float(d["phase_velocity"].max())
    out, r = m._step(s, 1e-5, {"up": True}, [0.0, 0.0], [196.2] * 2)
    shifted, t = make(10000.0)
    other, _ = shifted._step(t, 1e-5, {"up": True}, [0.0, 0.0], [196.2] * 2)
    for (_, a), (_, b) in zip(out.nodes, other.nodes):
        assert a.total_energy_j == pytest.approx(b.total_energy_j, abs=1e-7)
        assert a.component_moles == pytest.approx(b.component_moles, abs=1e-10)
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_finite_resistance_has_exact_implicit_velocity_and_retains_dissipated_energy():
    m, s = pipe(velocity=2.0, periodic=True)
    dt = 0.001
    losses = {n.name: 4.0 for n in m.nodes}
    out, r = m._step(s, dt, {c.name: True for c in m.links}, [0.0] * 2, [196.2] * 2, losses)
    a = dt * 4.0 / (2 * 0.5)
    expected = 2 * 2.0 / (1 + math.sqrt(1 + 4 * a * 2.0))
    for i, (_, v) in enumerate(out.nodes):
        d = m._state_data(i, v)
        assert float(d["u"]) == pytest.approx(expected, rel=1e-8)
        assert v.total_energy_j == pytest.approx(s.nodes[i][1].total_energy_j, abs=1e-6)
    assert r["entropy_production_j_k"] > 0


def test_slip_flux_is_sum_of_actual_phase_enthalpy_and_kinetic_fluxes():
    f = binary()
    area = math.pi * 0.05**2 / 4
    nodes = [
        PREquilibriumNode(
            str(i),
            area * 0.5,
            "pipe_cell",
            length_m=0.5,
            diameter_m=0.05,
            distribution_strength=0.01,
        )
        for i in range(2)
    ]
    links = [PREquilibriumLink("a", "0", "1", 0.05), PREquilibriumLink("b", "1", "0", 0.05)]
    m = PREquilibriumTransportNetwork(f, nodes, links)
    s = m.initialize(
        {
            str(i): {
                "pressure_pa": 1e6,
                "temperature_k": 196.2,
                "mole_fractions": [0.5, 0.5],
                "velocity_m_s": 2.0,
            }
            for i in range(2)
        }
    )
    d = [m._state_data(i, v) for i, (_, v) in enumerate(s.nodes)]
    _, records, _ = m._fluxes(d, {c.name: True for c in links})
    v = d[0]["phase_velocity"]
    q = d[0]["q"]
    pn = d[0]["phase_n"]
    volume = nodes[0].volume_m3
    mw = (d[0]["x"] * f.molar_masses).sum(-1)
    expected_n = area * (pn[:, None] / volume * d[0]["x"] * v[:, None]).sum(0)
    expected_m = area * ((pn * mw / volume * v.square()).sum() + d[0]["p"])
    expected_e = (
        area * (pn / volume * (q["molar_enthalpy_j_mol"] + 0.5 * mw * v.square()) * v).sum()
    )
    assert records[0]["flux"].cpu().tolist() == pytest.approx(
        [*expected_n.tolist(), float(expected_m), float(expected_e)], rel=1e-10, abs=1e-8
    )


@pytest.mark.parametrize("temperature", [320.0, 196.2])
def test_equilibrium_acoustic_characteristics_against_independent_pt_flash(temperature):
    import numpy as np

    from surface.peng_robinson_mixture import PengRobinsonMixture

    m, s = pipe(temperature, 2.0, True)
    d = m._state_data(0, s.nodes[0][1])
    active = torch.tensor([0, 1])
    actual = sorted(m._characteristics(0, d, active).tolist())
    reference = PengRobinsonMixture(m.fluid.components, m.fluid.binary_interactions)
    mw = np.array([c.molar_mass_kg_mol for c in m.fluid.components]) @ np.array([0.5, 0.5])

    def state(p, t):
        q = reference.flash_pt(p, t, np.array([0.5, 0.5]))
        vm = sum(v["molar_fraction"] * v["molar_volume_m3_mol"] for v in q["phases"])
        entropy = sum(v["molar_fraction"] * v["molar_entropy_j_mol_k"] for v in q["phases"])
        return mw / vm, entropy

    dp = 10.0
    dt = 0.001
    a, b = state(1e6 + dp, temperature), state(1e6 - dp, temperature)
    c, e = state(1e6, temperature + dt), state(1e6, temperature - dt)
    rp = (a[0] - b[0]) / (2 * dp)
    rt = (c[0] - e[0]) / (2 * dt)
    sp = (a[1] - b[1]) / (2 * dp)
    st = (c[1] - e[1]) / (2 * dt)
    sound = (rp - rt * sp / st) ** -0.5
    assert actual == pytest.approx([2.0 - sound, 2.0, 2.0, 2.0 + sound], rel=3e-5, abs=1e-5)


def test_coflow_donor_follows_absolute_phase_velocity():
    f = binary()
    area = math.pi * 0.05**2 / 4
    nodes = [
        PREquilibriumNode(
            str(i),
            area * 0.5,
            "pipe_cell",
            length_m=0.5,
            diameter_m=0.05,
            distribution_strength=0.01,
        )
        for i in range(2)
    ]
    m = PREquilibriumTransportNetwork(f, nodes, [PREquilibriumLink("a", "0", "1", 0.05)])
    initial = {
        str(i): {
            "pressure_pa": 1e6,
            "temperature_k": 196.2,
            "mole_fractions": [0.5, 0.5] if i == 0 else [0.6, 0.4],
            "velocity_m_s": 2.0,
        }
        for i in range(2)
    }
    s = m.initialize(initial)
    d = [m._state_data(i, v) for i, (_, v) in enumerate(s.nodes)]
    assert all(float(v["phase_velocity"].min()) > 0 for v in d)
    _, records, _ = m._fluxes(d, {"a": True})
    source = d[0]
    v = source["phase_velocity"]
    expected = area * (
        source["phase_n"][:, None] / nodes[0].volume_m3 * source["x"] * v[:, None]
    ).sum(0)
    assert records[0]["flux"][:-2].tolist() == pytest.approx(
        expected.tolist(), rel=1e-10, abs=1e-10
    )


def test_finite_wall_retains_total_energy_and_restart_is_exact():
    from surface.vessel_wall import VesselWall

    node = PREquilibriumNode("cell", 0.001, wall=VesselWall(100.0, 20.0))
    m = PREquilibriumTransportNetwork(binary(), [node])
    equilibrium = m.initialize(
        {"cell": {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]}}
    )
    assert equilibrium.nodes[0][1].wall_temperature_k == 196.2
    s = m.initialize(
        {
            "cell": {
                "pressure_pa": 1e6,
                "temperature_k": 196.2,
                "mole_fractions": [0.5, 0.5],
                "wall_temperature_k": 230.0,
            }
        }
    )
    original = s.to_dict()
    r = m.advance(s, 1e-4)
    out = PREquilibriumTransportState.from_dict(r["state"])
    assert abs(r["conservation_residuals"][-1]) < 1e-6
    assert out.nodes[0][1].temperature_k > s.nodes[0][1].temperature_k
    restored = PREquilibriumTransportNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(PREquilibriumTransportState.from_dict(json.loads(json.dumps(out.to_dict()))))
    a = m.advance(out, 1e-5)
    b = restored.advance(out, 1e-5)
    assert a["state"] == b["state"]
    assert s.to_dict() == original


def test_invalid_wave_bound_and_failed_report_do_not_change_checkpoint():
    from surface.liquid import HydraulicConvergenceError, PropertyModelError

    with pytest.raises(PropertyModelError, match="wave bound"):
        pipe(wave_speed_bound_m_s=101.0, maximum_velocity_m_s=100.0, temperature=320.0)
    m, s = pipe(maximum_attempts=1)
    before = s.to_dict()
    with pytest.raises(HydraulicConvergenceError):
        m.advance(s, 0.001)
    assert s.to_dict() == before


def test_prepared_newton_residual_does_not_call_flash(monkeypatch):
    m, s = pipe(velocity=0.1)
    seeds = [m._seed(i, v) for i, (_, v) in enumerate(s.nodes)]
    y, _, _, evaluate, _, _, _ = m._problem(s, seeds, 1e-4, {"internal": True}, [0.0, 0.0])

    def forbidden(*args, **kwargs):
        raise AssertionError("Nested flash in coupled Newton residual")

    monkeypatch.setattr(type(m.fluid), "flash_pt", forbidden)
    monkeypatch.setattr(type(m.fluid), "flash_uv", forbidden)
    residual = evaluate(y)[0]
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    assert torch.isfinite(residual).all()
    assert torch.isfinite(jac).all()


def test_independent_auditor_rejects_corrupted_transport(monkeypatch):
    import copy
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.syspath_prepend(str(root / "scripts"))
    runner = load("run_pr_equilibrium_transport")
    checker = load("check_pr_equilibrium_transport")
    config = json.loads((root / "configs/surface_pr_transport_slip.json").read_text())
    config["schedule"] = [{"duration_s": 1e-5}]
    original = copy.deepcopy(config)
    report = json.loads(json.dumps(runner.run(config)))
    assert config == original

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent auditor called runtime state solver")

    monkeypatch.setattr(PREquilibriumTransportNetwork, "_decode", forbidden)
    monkeypatch.setattr(PREquilibriumTransportNetwork, "_step", forbidden)
    monkeypatch.setattr(PREquilibriumTransportNetwork, "inspect", forbidden)
    assert checker.audit(config, report)["accepted"]
    broken = copy.deepcopy(report)
    step = broken["intervals"][0]["steps"][0]
    first = next(iter(step["state"]["nodes"].values()))
    first["total_energy_j"] += 1.0
    with pytest.raises(ValueError):
        checker.audit(config, broken)


def test_sparse_graph_newton_matches_direct_and_support_is_exact(monkeypatch):
    from dataclasses import replace

    m, s = pipe(velocity=0.1)
    s = replace(
        s, nodes=tuple((name, replace(v, pressure_pa=v.pressure_pa)) for name, v in s.nodes)
    )
    seed = [m._seed(i, v) for i, (_, v) in enumerate(s.nodes)]
    y, _, _, evaluate, offsets, _, _ = m._problem(s, seed, 1e-5, {"internal": True}, [0.0, 0.0])
    pattern = m._sparsity(offsets, {"internal": True})
    j = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    mask = torch.zeros_like(j, dtype=torch.bool)
    mask[pattern[0], pattern[1]] = True
    if bool((~mask).any()):
        assert float(j[~mask].abs().max()) == 0.0
    direct, _ = m._step(s, 1e-5, {"internal": True}, [0.0, 0.0], [196.2, 196.2])
    import surface.pr_equilibrium_transport as implementation
    from surface.torch_linear import TorchNewton

    def sparse(*args, **kwargs):
        kwargs["dense_limit"] = 0
        return TorchNewton(*args, **kwargs)

    # Force the public model to prepare the sparse pattern as well.
    m.dense_limit = 0
    m.model_id = implementation._digest(m.to_dict())
    s = replace(s, model_id=m.model_id)
    monkeypatch.setattr(implementation, "TorchNewton", sparse)
    result, _ = m._step(s, 1e-5, {"internal": True}, [0.0, 0.0], [196.2, 196.2])
    for (_, a), (_, b) in zip(direct.nodes, result.nodes):
        assert a.pressure_pa == pytest.approx(b.pressure_pa, abs=1e-3)
        assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-7)


def test_graph_above_previous_dense_budget_uses_local_pattern_and_conserves():
    from dataclasses import replace

    base, initial = pipe(temperature=320.0)
    count = 160
    nodes = [replace(base.nodes[0], name=f"cell_{i:03d}") for i in range(count)]
    links = [
        PREquilibriumLink(f"face_{i:03d}", nodes[i].name, nodes[(i + 1) % count].name, 0.05)
        for i in range(count)
    ]
    m = PREquilibriumTransportNetwork(base.fluid, nodes, links)
    s = PREquilibriumTransportState(
        tuple((n.name, initial.nodes[0][1]) for n in nodes), 0.0, m.initial_step_s, m.model_id
    )
    seeds = [m._seed(i, v) for i, (_, v) in enumerate(s.nodes)]
    commands = {c.name: True for c in links}
    y, _, _, _, offsets, _, _ = m._problem(s, seeds, 1e-6, commands, [0.0] * count)
    assert len(y) > 768
    rows, _cols = m._sparsity(offsets, commands)
    assert len(rows) < 30 * len(y)
    result, r = m._step(s, 1e-6, commands, [0.0] * count, [320.0] * count)
    assert result.time_s == 1e-6
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_named_drift_flux_closure_transports_energy_and_roundtrips():
    from dataclasses import replace

    from surface.drift_flux_closure import ZuberFindlayClosure

    old, _ = pipe(velocity=0.1, periodic=True)
    closure = ZuberFindlayClosure(1.00001, 1e-5, maximum_gas_volume_fraction=0.999)
    m = PREquilibriumTransportNetwork(
        old.fluid, [replace(n, drift_flux=closure) for n in old.nodes], old.links
    )
    s = m.initialize(
        {
            n.name: {
                "pressure_pa": 1e6,
                "temperature_k": 196.2,
                "mole_fractions": [0.5, 0.5],
                "velocity_m_s": 0.1,
            }
            for n in m.nodes
        }
    )
    final, r = m._step(s, 1e-5, {c.name: True for c in m.links}, [0.0, 0.0], [196.2, 196.2])
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    restored = PREquilibriumTransportNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(PREquilibriumTransportState.from_dict(json.loads(json.dumps(final.to_dict()))))
