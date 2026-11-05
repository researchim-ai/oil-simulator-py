"""Quasi-1-D changing area shares N/M/E graph, geometry source and audits."""

import json
import math

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork
from surface.pr_transport_builder import PREquilibriumNozzle, build_equilibrium_transport


def setup(temperature=320.0, wave_fan="fixed_bound"):
    f = binary()
    nodes = [PREquilibriumNode("source", 0.01), PREquilibriumNode("target", 0.01)]
    nozzle = PREquilibriumNozzle(
        "nozzle", "source", "target", (0.02, 0.015, 0.01, 0.015, 0.02), (0.02,) * 4
    )
    m = build_equilibrium_transport(f, nodes, [nozzle], wave_fan=wave_fan)
    s = m.initialize(
        {
            n.name: {"pressure_pa": 1e6, "temperature_k": temperature, "mole_fractions": [0.5, 0.5]}
            for n in m.nodes
        }
    )
    return m, s


@pytest.mark.parametrize("temperature", [320.0, 196.2])
def test_static_nozzle_exactly_balances_stationary_wall_pressure(temperature):
    m, s = setup(temperature)
    commands = {c.name: True for c in m.links}
    out, r = m._step(s, 1e-5, commands, [0.0] * len(m.nodes), [temperature] * len(m.nodes))
    for (_, a), (_, b) in zip(s.nodes, out.nodes):
        assert b.pressure_pa == pytest.approx(a.pressure_pa, abs=1e-3)
        assert b.momentum_kg_m_s == pytest.approx(0.0, abs=1e-10)
        assert b.total_energy_j == pytest.approx(a.total_energy_j, abs=1e-6)
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_geometry_and_joint_jacobian_use_face_areas_not_centre_area():
    m, s = setup(wave_fan="local_single_phase")
    n = m.nodes[m.names.index("nozzle__0000")]
    expected = math.pi * 0.02 * (0.02**2 + 0.02 * 0.015 + 0.015**2) / 12
    assert n.volume_m3 == pytest.approx(expected, rel=1e-15)
    assert n.face_area("left") == pytest.approx(math.pi * 0.02**2 / 4)
    seeds = [m._seed(i, v) for i, (_, v) in enumerate(s.nodes)]
    y, _, _, evaluate, _, _, _ = m._problem(
        s, seeds, 1e-5, {c.name: True for c in m.links}, [0.0] * len(m.nodes)
    )
    direction = torch.linspace(-0.001, 0.001, len(y), dtype=y.dtype)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    eps = 1e-5
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-6
    restored = PREquilibriumTransportNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    assert restored.model_id == m.model_id


def test_invalid_frustum_cannot_silently_change_volume_or_face_matching():
    with pytest.raises(ValueError, match="frustum"):
        PREquilibriumNode(
            "bad",
            1.0,
            "pipe_cell",
            length_m=0.1,
            diameter_m=0.015,
            left_diameter_m=0.02,
            right_diameter_m=0.01,
        )
    with pytest.raises(ValueError, match="Both face"):
        PREquilibriumNode(
            "bad", 1.0, "pipe_cell", length_m=0.1, diameter_m=0.015, left_diameter_m=0.02
        )


@pytest.mark.parametrize("wave_fan", ["fixed_bound", "local_single_phase"])
def test_nozzle_pressure_pulse_and_independent_geometry_audit(monkeypatch, wave_fan):
    import copy
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    runner = load("run_pr_equilibrium_transport")
    checker = load("check_pr_equilibrium_transport")
    m, _ = setup(wave_fan=wave_fan)
    initial = {
        n.name: {
            "pressure_pa": 1e6
            + (100.0 if n.name == "source" else -100.0 if n.name == "target" else 0.0),
            "temperature_k": 320.0,
            "mole_fractions": [0.5, 0.5],
        }
        for n in m.nodes
    }
    config = {
        "model": m.to_dict(),
        "initial": {"nodes": initial},
        "schedule": [{"duration_s": 1e-6}],
    }
    config = json.loads(json.dumps(config))
    report = json.loads(json.dumps(runner.run(config)))
    assert checker.audit(config, report)["accepted"]
    step = report["intervals"][0]["steps"][-1]
    assert step["links"]["nozzle__face_0000"]["flux"][0] > 0.0
    broken = copy.deepcopy(report)
    broken["intervals"][0]["steps"][0]["nodes"]["nozzle__0000"]["force_impulse_kg_m_s"] += 1.0
    with pytest.raises(ValueError):
        checker.audit(config, broken)


def test_whole_nozzle_wall_is_distributed_by_actual_lateral_area():
    from surface.vessel_wall import VesselWall

    f = binary()
    vessels = [PREquilibriumNode("a", 0.01), PREquilibriumNode("b", 0.01)]
    nozzle = PREquilibriumNozzle(
        "n", "a", "b", (0.02, 0.015, 0.01), (0.02, 0.04), wall=VesselWall(1000.0, 10.0, 1.0)
    )
    m = build_equilibrium_transport(f, vessels, [nozzle])
    cells = [n for n in m.nodes if n.kind == "pipe_cell"]
    areas = [
        math.pi * 0.0175 * math.hypot(0.02, 0.0025),
        math.pi * 0.0125 * math.hypot(0.04, 0.0025),
    ]
    for cell, area in zip(cells, areas):
        assert cell.wall.heat_capacity_j_k == pytest.approx(1000.0 * area / sum(areas))
    assert sum(n.wall.heat_capacity_j_k for n in cells) == pytest.approx(1000.0, abs=1e-12)


def isentropic_nozzle(cells, reservoir_ports="riemann"):
    """NASA ideal-gas A/A* reference, approached by dilute real-PR methane.

    Only this independent fixture uses SciPy's scalar algebraic root finder.
    """
    import numpy as np
    from scipy.optimize import brentq

    from surface.equilibrium_properties import EquilibriumFluid
    from surface.peng_robinson import R

    cfg = binary().to_dict()
    cfg["min_pressure_pa"] = 1.0
    f = EquilibriumFluid.from_dict(cfg)
    component = f.components[0]
    gamma = component.ideal_cp_j_mol_k / (component.ideal_cp_j_mol_k - R)
    rgas = R / component.molar_mass_kg_mol
    p0 = 1000.0
    t0 = 400.0
    length = 1.0
    throat = 0.01
    grid = np.linspace(0.0, 1.0, cells + 1)
    diameters = throat * np.sqrt(1 + 3 * (2 * grid - 1) ** 2)
    nozzle = PREquilibriumNozzle(
        "nozzle", "source", "target", tuple(diameters), tuple([length / cells] * cells)
    )
    m = build_equilibrium_transport(
        f,
        [PREquilibriumNode("source", 1.0), PREquilibriumNode("target", 1.0)],
        [nozzle],
        wave_fan="local_single_phase",
        reservoir_ports=reservoir_ports,
        maximum_velocity_m_s=1200.0,
        wave_speed_bound_m_s=2000.0,
        initial_step_s=1e-6,
        maximum_step_s=1e-4,
    )

    def area_ratio(mach):
        return ((2 / (gamma + 1)) * (1 + (gamma - 1) * mach * mach / 2)) ** (
            (gamma + 1) / (2 * (gamma - 1))
        ) / mach

    initial = {
        "source": {"pressure_pa": p0, "temperature_k": t0, "mole_fractions": [1.0, 0.0]},
        "target": {"pressure_pa": p0 * 0.1, "temperature_k": t0, "mole_fractions": [1.0, 0.0]},
    }
    for i in range(cells):
        x = (i + 0.5) / cells
        ratio = 1 + 3 * (2 * x - 1) ** 2
        mach = (
            brentq(lambda value, ratio=ratio: area_ratio(value) - ratio, 1e-7, 1.0)
            if x < 0.5
            else brentq(lambda value, ratio=ratio: area_ratio(value) - ratio, 1.0, 10.0)
        )
        temp = t0 / (1 + (gamma - 1) * mach * mach / 2)
        pressure = p0 * (temp / t0) ** (gamma / (gamma - 1))
        velocity = mach * math.sqrt(gamma * rgas * temp)
        initial[f"nozzle__{i:04d}"] = {
            "pressure_pa": pressure,
            "temperature_k": temp,
            "mole_fractions": [1.0, 0.0],
            "velocity_m_s": velocity,
        }
    s = m.initialize(initial)
    expected = (
        math.pi
        * throat**2
        / 4
        * p0
        / math.sqrt(t0)
        * math.sqrt(gamma / rgas)
        * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    )
    return m, s, expected, initial


def test_transonic_isentropic_reference_flux_converges_with_mesh():
    errors = []
    for cells in [16, 32, 64]:
        m, s, expected, _ = isentropic_nozzle(cells)
        data = [m._state_data(i, v) for i, (_, v) in enumerate(s.nodes)]
        _, records, valid = m._fluxes(data, {c.name: True for c in m.links})
        assert bool(valid)
        row = next(r for r in records if r["name"] == f"nozzle__face_{cells // 2:04d}")
        actual = float((row["flux"][:-2] * m.fluid.molar_masses).sum())
        errors.append(abs(actual / expected - 1.0))
    assert errors[1] < 0.8 * errors[0]
    assert errors[2] < 0.8 * errors[1]
    assert errors[-1] < 0.05


@pytest.mark.parametrize("ports", ["riemann", "stagnation_single_phase"])
@pytest.mark.parametrize("fan", ["local_single_phase", "pressure_single_phase"])
def test_transonic_reference_advances_without_creating_material_or_energy(ports, fan):
    m, s, _, initial = isentropic_nozzle(8, reservoir_ports=ports)
    config = m.to_dict()
    config["wave_fan"] = fan
    m = type(m).from_dict(config)
    s = m.initialize(initial)
    out, r = m._step(
        s,
        1e-6,
        {c.name: True for c in m.links},
        [0.0] * len(m.nodes),
        [v.temperature_k for _, v in s.nodes],
    )
    assert out.time_s == 1e-6
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    throat = r["links"]["nozzle__face_0004"]["flux"]
    assert throat[0] > 0.0
