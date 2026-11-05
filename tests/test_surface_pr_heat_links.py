"""Finite thermal circuits share the physical Newton without mixing material."""

import copy
import json
from pathlib import Path

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork
from surface.pr_heat_link import PRHeatLink


def setup(dense_limit=128, different_components=False):
    model = PREquilibriumTransportNetwork(
        binary(),
        [PREquilibriumNode("cold", 0.01), PREquilibriumNode("hot", 0.01)],
        heat_links=[PRHeatLink("exchanger", "hot", "cold", 1.0)],
        dense_limit=dense_limit,
    )
    state = model.initialize(
        {
            "cold": {
                "pressure_pa": 1000.0,
                "temperature_k": 300.0,
                "mole_fractions": [0.0, 1.0] if different_components else [1.0, 0.0],
            },
            "hot": {"pressure_pa": 1000.0, "temperature_k": 350.0, "mole_fractions": [1.0, 0.0]},
        }
    )
    return model, state


@pytest.mark.parametrize("dense_limit", [1, 128])
@pytest.mark.parametrize("different_components", [False, True])
def test_heat_exchange_conserves_energy_without_material_exchange(
    dense_limit, different_components
):
    m, old = setup(dense_limit, different_components)
    dt = 1e-4
    new, record = m._step(old, dt, {"exchanger": True}, [0.0] * 2, [300.0, 350.0])
    a, b = dict(old.nodes), dict(new.nodes)
    assert 300 < b["cold"].temperature_k < b["hot"].temperature_k < 350
    q = record["heat_links"]["exchanger"]["heat_rate_w"]
    for name, sign in [("cold", 1), ("hot", -1)]:
        assert b[name].component_moles == pytest.approx(a[name].component_moles, abs=1e-15)
        assert b[name].total_energy_j - a[name].total_energy_j == pytest.approx(
            sign * dt * q, abs=1e-10
        )
        assert [v == 0 for v in b[name].component_moles] == [
            v == 0 for v in a[name].component_moles
        ]
    assert record["heat_links"]["exchanger"]["entropy_production_w_k"] > 0
    assert record["entropy_production_j_k"] > 0
    assert max(map(abs, record["conservation_residuals"])) < 1e-10
    if not different_components:
        # Independent constant-Cv backward-Euler limit; dilute PR departures
        # are retained by the model, without replacing its EOS by ideal gas.
        capacity = {}
        for name, body in old.nodes:
            q0 = m.fluid.homogeneous(body.pressure_pa, body.temperature_k, [1.0, 0.0])
            capacity[name] = sum(body.component_moles) * float(q0["homogeneous_molar_cv_j_mol_k"])
        difference = 50 / (1 + dt * (1 / capacity["cold"] + 1 / capacity["hot"]))
        expected = 300 + dt * difference / capacity["cold"]
        assert b["cold"].temperature_k == pytest.approx(expected, abs=1e-8)


def test_isolation_restart_and_thermal_jacobian_support():
    m, old = setup()
    seeds = [m._seed(i, body) for i, (_, body) in enumerate(old.nodes)]
    y, _, _, ev, offsets, _, _ = m._problem(old, seeds, 1e-4, {"exchanger": True}, [0.0] * 2)
    jac = torch.func.jacfwd(lambda z: ev(z)[0])(y)
    direction = torch.linspace(-0.001, 0.001, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (ev(y + eps * direction)[0] - ev(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-5, atol=1e-8)
    assert bool((jac[: offsets[1], offsets[1] :].abs() > 1e-10).any())
    off = m._sparsity(offsets, {"exchanger": False})
    assert not any(a < offsets[1] <= b for a, b in zip(*off))
    closed, r = m._step(old, 1e-4, {"exchanger": False}, [0.0] * 2, [300.0, 350.0])
    assert r["heat_links"]["exchanger"]["heat_rate_w"] == 0.0
    for (_, before), (_, after) in zip(old.nodes, closed.nodes):
        assert after.temperature_k == pytest.approx(before.temperature_k, abs=1e-9)
    restored = type(m).from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(closed)


def test_thermal_registry_rejects_invalid_topology():
    with pytest.raises(ValueError):
        PRHeatLink("bad", "a", "a", 1)
    with pytest.raises(ValueError):
        PREquilibriumTransportNetwork(
            binary(),
            [PREquilibriumNode("a", 0.01)],
            heat_links=[PRHeatLink("bad", "a", "missing", 1)],
        )


def test_finite_thermal_circuits_condense_then_evaporate_without_external_heat():
    from surface.peng_robinson import R

    f = binary()
    fluid = f.components[1]
    p = fluid.saturation_at_temperature(200)["pressure_pa"]
    liquid = fluid.state(p, 200, "liquid", derivatives=False)
    vapor = fluid.state(p, 200, "vapor", derivatives=False)
    mw = fluid.molar_mass_kg_mol
    vm = (0.4 * liquid["specific_volume_m3_kg"] + 0.6 * vapor["specific_volume_m3_kg"]) * mw
    target = 10 * (
        0.4 * (liquid["enthalpy_j_kg"] * mw - p * liquid["specific_volume_m3_kg"] * mw)
        + 0.6 * (vapor["enthalpy_j_kg"] * mw - p * vapor["specific_volume_m3_kg"] * mw)
    )
    attraction, _ = fluid._a(220)
    p0 = R * 220 / (vm - fluid.b) - attraction / (vm * vm + 2 * fluid.b * vm - fluid.b**2)
    old_energy = float(f.homogeneous(p0, 220, [0, 1])["molar_internal_energy_j_mol"]) * 10
    drop = old_energy - target
    m = PREquilibriumTransportNetwork(
        f,
        [
            PREquilibriumNode("process", 10 * vm),
            PREquilibriumNode("cold", 10),
            PREquilibriumNode("hot", 10),
        ],
        heat_links=[
            PRHeatLink("cool", "process", "cold", drop / 30),
            PRHeatLink("warm", "hot", "process", drop / 80),
        ],
    )
    old = m.initialize(
        {
            "process": {"pressure_pa": p0, "temperature_k": 220, "mole_fractions": [0, 1]},
            "cold": {"pressure_pa": 1e6, "temperature_k": 170, "mole_fractions": [1, 0]},
            "hot": {"pressure_pa": 1e6, "temperature_k": 300, "mole_fractions": [1, 0]},
        }
    )
    condensed, a = m._step(
        old, 1.0, {"cool": True, "warm": False}, [0.0] * 3, [170.0, 300.0, 220.0]
    )
    final, b = m._step(
        condensed, 1.0, {"cool": False, "warm": True}, [0.0] * 3, [170.0, 300.0, 220.0]
    )
    assert len(dict(old.nodes)["process"].phase_roots) == 1
    assert len(dict(condensed.nodes)["process"].phase_roots) == 2
    assert len(dict(final.nodes)["process"].phase_roots) == 1
    assert dict(condensed.nodes)["process"].temperature_k == pytest.approx(200, abs=0.1)
    assert dict(final.nodes)["process"].temperature_k == pytest.approx(220, abs=0.1)
    for record in (a, b):
        assert record["external_heat_j"] == 0
        assert record["entropy_production_j_k"] > 0
        assert max(map(abs, record["conservation_residuals"])) < 1e-6
    for (name, before), (_, after) in zip(old.nodes, final.nodes):
        assert after.component_moles == pytest.approx(before.component_moles, abs=1e-10)
        if name == "process":
            assert after.component_moles[0] == 0


def test_gravitational_layer_disappears_and_reappears_with_finite_thermal_circuits():
    from surface.pr_gravity_transport import PRGravityTransportNetwork, _ColumnCoupling
    from surface.separator_geometry import SeparatorGeometry
    from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState

    f = binary()
    transport = PREquilibriumTransportNetwork(
        f,
        [PREquilibriumNode("hot", 10), PREquilibriumNode("cold", 10)],
        gravity_m_s2=9.80665,
        maximum_step_s=100,
    )
    column = StratifiedPRColumn(
        f,
        SeparatorGeometry(0.1, 0.1, "vertical"),
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    model = PRGravityTransportNetwork(
        transport,
        {"column": column},
        [],
        heat_links=[
            PRHeatLink("heat", "hot", "column", 10),
            PRHeatLink("cool", "column", "cold", 10),
        ],
    )
    ts = transport.initialize(
        {
            "hot": {"pressure_pa": 1e6, "temperature_k": 320, "mole_fractions": [1, 0]},
            "cold": {"pressure_pa": 1e6, "temperature_k": 170, "mole_fractions": [1, 0]},
        }
    )
    cs = StratifiedPRState.from_dict(column.initialize(1e6, 196.2, [0.5, 0.5])["state"])
    old = model.initialize(ts, {"column": cs})
    commands = {"heat": True, "cool": False}
    frozen = old.to_dict()
    predictor = _ColumnCoupling(model, old, 100, 2, commands, {}, {})
    assert cs.temperature_k < predictor.thermal_guesses["column"] < 320
    assert old.to_dict() == frozen
    # These deliberately large implicit steps exercise branch changes, not
    # a claim of temporally resolved heating/cooling trajectories.
    hot, a = model._step(old, 100, commands, {}, {}, {})
    cold, b = model._step(hot, 10, {"heat": False, "cool": True}, {}, {}, {})
    assert len(cs.phase_roots) == 2
    assert len(dict(hot.columns)["column"].phase_roots) == 1
    assert len(dict(cold.columns)["column"].phase_roots) == 2
    for state, record in [(hot, a), (cold, b)]:
        assert dict(state.columns)["column"].component_moles == pytest.approx(
            cs.component_moles, abs=1e-12
        )
        assert record["total_heat_j"] == 0
        assert record["entropy_production_j_k"] > 0
        assert max(map(abs, record["conservation_residuals"])) < 1e-6


@pytest.mark.parametrize("column_target", [False, True])
@pytest.mark.parametrize("dense_limit", [1, 128])
def test_column_heat_exchange_and_independent_graph_audit(column_target, dense_limit, monkeypatch):
    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.separator_geometry import SeparatorGeometry
    from surface.stratified_pr import StratifiedPRColumn

    f = binary()
    transport = PREquilibriumTransportNetwork(
        f,
        [PREquilibriumNode("receiver", 0.01)],
        gravity_m_s2=9.80665,
        dense_limit=dense_limit,
    )
    column = StratifiedPRColumn(
        f,
        SeparatorGeometry(0.1, 0.1, "vertical"),
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    columns = {"hot": column}
    if column_target:
        columns["cold"] = column
    m = PRGravityTransportNetwork(
        transport,
        columns,
        [],
        heat_links=[PRHeatLink("exchanger", "hot", "cold" if column_target else "receiver", 10.0)],
    )
    config = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {
                    "transport": {
                        "receiver": {
                            "pressure_pa": 1e6,
                            "temperature_k": 300.0,
                            "mole_fractions": [0.5, 0.5],
                        }
                    },
                    "columns": {
                        name: {
                            "pressure_pa": 1e6,
                            "temperature_k": 340.0 if name == "hot" else 300.0,
                            "mole_fractions": [0.5, 0.5],
                        }
                        for name in columns
                    },
                    "wall_temperatures": {},
                },
                "schedule": [
                    {"duration_s": 1e-6},
                    {"duration_s": 1e-6, "openings": {"exchanger": False}},
                ],
            }
        )
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_gravity_transport import audit
    from run_pr_gravity_transport import run

    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    first = report["intervals"][0]["steps"][0]
    assert first["columns"]["heat_links"]["exchanger"]["heat_rate_w"] > 0
    assert first["entropy_production_j_k"] > 0
    assert (
        report["intervals"][1]["steps"][0]["columns"]["heat_links"]["exchanger"]["heat_rate_w"] == 0
    )
    assert max(map(abs, report["conservation_residuals"])) < 1e-6
    broken = copy.deepcopy(report)
    broken["intervals"][0]["steps"][0]["columns"]["heat_links"]["exchanger"]["heat_rate_w"] += 1
    with pytest.raises(ValueError):
        audit(config, broken)
