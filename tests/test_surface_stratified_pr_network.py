"""Implicit finite-aperture network: conservation, upwind and cross derivatives."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
from surface.stratified_pr_flow import StratifiedPRFlow
from surface.stratified_pr_network import (
    StratifiedConnection,
    StratifiedNetworkState,
    StratifiedPRNetwork,
)
from surface.vessel_wall import VesselWall

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def setup(device="cpu", reverse=False, two=False, **options):
    cfg = json.loads(
        (
            ROOT
            / (
                "configs/surface_stratified_pr.json"
                if two
                else "configs/surface_stratified_pr_gas.json"
            )
        ).read_text()
    )
    cfg["system"]["geometry"]["length_m"] = 2.0
    vessels = {}
    states = {}
    for name, p in [("a", 990000.0 if reverse else 1e6), ("b", 1e6 if reverse else 990000.0)]:
        c = StratifiedPRColumn.from_dict(cfg["system"], device=device)
        q = c.initialize(**{**cfg["initial"], "pressure_pa": p})
        s = StratifiedPRState.from_dict(q["state"])
        v = StratifiedPRFlow(c, VesselWall(1000.0, 0.0))
        vessels[name] = v
        states[name] = v.initialize(s, s.temperature_k)
    height = states["a"].column.interface_heights_m[0] if two else 1.0
    connection = StratifiedConnection("link", "a", "b", height, height, 0.05, 1e-10, **options)
    model = StratifiedPRNetwork(vessels, [connection], initial_step_s=0.02)
    return model, model.initialize(states)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reverse", [False, True])
def test_bidirectional_transfer_conserves_each_component_and_energy(device, reverse):
    m, s = setup(device, reverse=reverse)
    final, report = m._step(s, 0.02, m._commands(None))
    rate = report["transfers"][0]["component_rates_mol_s"]
    assert all(r < 0 if reverse else r > 0 for r in rate)
    old, new = dict(s.vessels), dict(final.vessels)
    for i, r in enumerate(rate):
        assert new["a"].column.component_moles[i] - old["a"].column.component_moles[
            i
        ] == pytest.approx(-0.02 * r, abs=1e-10)
        assert new["b"].column.component_moles[i] - old["b"].column.component_moles[
            i
        ] == pytest.approx(0.02 * r, abs=1e-10)
    assert max(map(abs, report["component_residuals_mol"])) < 1e-10
    assert abs(report["energy_residual_j"]) < 1e-6
    assert report["entropy_production_j_k"] >= -1e-7
    assert sum(p["fraction"] for p in report["transfers"][0]["phase_pair_areas"]) == pytest.approx(
        1.0
    )
    restored = StratifiedPRNetwork.from_dict(json.loads(json.dumps(m.to_dict())), device=device)
    restored.inspect(StratifiedNetworkState.from_dict(final.to_dict()))


def test_joint_jacobian_matches_directional_difference_and_has_cross_blocks():
    m, s = setup()
    states = dict(s.vessels)
    seeds = m._seeds(states)
    y, _lo, _hi, evaluate, problems = m._problem(states, seeds, 0.02, 4, m._commands(None))
    jac = torch.func.jacfwd(lambda x: evaluate(x)[0])(y)
    direction = torch.linspace(-0.02, 0.02, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert torch.isfinite(jac).all()
    assert (jac @ direction - fd).abs().max() < 1e-6
    first = len(problems["a"][0])
    second = len(problems["b"][0])
    # A's conservation rows depend on the connector unknowns from both ends.
    assert jac[:first, first + second :].abs().max() > 1e-8


def test_check_valve_and_commanded_stop_are_exactly_closed():
    m, s = setup(reverse=True, allow_reverse=False)
    final, report = m._step(s, 0.02, m._commands(None))
    assert report["transfers"][0]["component_rates_mol_s"] == [0.0, 0.0]
    assert (
        dict(final.vessels)["a"].column.component_moles
        == dict(s.vessels)["a"].column.component_moles
    )
    m, s = setup()
    final, report = m._step(s, 0.02, m._commands({"link": 0.0}))
    assert report["transfers"][0]["total_energy_rate_w"] == 0.0
    assert (
        dict(final.vessels)["b"].column.component_moles
        == dict(s.vessels)["b"].column.component_moles
    )


def test_public_interval_and_restart_are_atomic():
    from surface.liquid import HydraulicConvergenceError

    m, s = setup()
    before = s.to_dict()
    result = m.advance(s, 0.02)
    final = StratifiedNetworkState.from_dict(result["state"])
    assert final.time_s == 0.02
    assert s.to_dict() == before
    m.inspect(final)
    assert abs(result["energy_residual_j"]) < 1e-6
    stopped = m.advance(final, 0.01, openings={"link": 0.0})
    assert stopped["state"]["time_s"] == 0.03
    with pytest.raises(ValueError, match="Unknown connection"):
        m.advance(s, 0.02, openings={"missing": 0.0})
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        limited = StratifiedPRNetwork(
            m.vessels, m.connections, initial_step_s=0.02, maximum_attempts=1
        )
        limited.advance(limited.initialize(dict(s.vessels)), 0.1)
    assert s.to_dict() == before


def test_two_phase_overlapping_apertures():
    m, s = setup(two=True)
    final, report = m._step(s, 0.005, m._commands(None))
    assert len(dict(final.vessels)["a"].column.phase_roots) == 2
    fractions = report["transfers"][0]["phase_pair_areas"]
    assert len(fractions) == 4
    assert sum(v["fraction"] for v in fractions) == pytest.approx(1.0, abs=1e-8)
    assert sum(v["fraction"] > 1e-6 for v in fractions) >= 2
    assert abs(report["energy_residual_j"]) < 1e-6


def test_ideal_gas_pair_matches_independent_scalar_implicit_solution():
    from scipy.optimize import brentq

    from surface.peng_robinson import R

    cfg = json.loads((ROOT / "configs/surface_stratified_pr_gas.json").read_text())
    cfg["system"]["geometry"]["length_m"] = 1.0
    cfg["system"]["gravity_m_s2"] = 0.0
    cfg["system"]["fluid"]["min_pressure_pa"] = 1e-7
    vessels = {}
    states = {}
    for name, p in [("a", 0.01), ("b", 0.0099)]:
        c = StratifiedPRColumn.from_dict(cfg["system"])
        s = StratifiedPRState.from_dict(c.initialize(p, 320.0, [0.5, 0.5])["state"])
        v = StratifiedPRFlow(c, VesselWall(1000.0, 0.0))
        vessels[name] = v
        states[name] = v.initialize(s, 320.0)
    link = StratifiedConnection("ideal", "a", "b", 0.5, 0.5, 0.05, 10.0)
    model = StratifiedPRNetwork(vessels, [link])
    old = model.initialize(states)
    result, report = model._step(old, 0.1, model._commands(None))
    na = sum(states["a"].column.component_moles)
    nb = sum(states["b"].column.component_moles)
    cv = sum(0.5 * (component.ideal_cp_j_mol_k - R) for component in model.fluid.components)
    volume = vessels["a"].column.geometry.volume_m3

    def values(fraction):
        loss = na * fraction
        ta = na * cv * 320 / (na * cv + loss * R)
        tb = (nb * cv * 320 + loss * (cv + R) * ta) / ((nb + loss) * cv)
        pa = (na - loss) * R * ta / volume
        pb = (nb + loss) * R * tb / volume
        return ta, tb, pa, pb

    def residual(fraction):
        ta, _tb, pa, pb = values(fraction)
        return fraction - 0.1 * 10.0 * (pa - pb) * pa / (R * ta * na)

    fraction = brentq(residual, 0.0, 0.05, xtol=1e-14)
    ta, tb, _pa, _pb = values(fraction)
    actual = dict(result.vessels)
    assert actual["a"].column.temperature_k == pytest.approx(ta, abs=2e-6)
    assert actual["b"].column.temperature_k == pytest.approx(tb, abs=2e-6)
    assert sum(actual["a"].column.component_moles) / na == pytest.approx(1 - fraction, abs=1e-8)
    assert sum(report["transfers"][0]["component_rates_mol_s"]) > 0


def test_shared_height_and_component_support_are_explicit_constraints():
    from surface.liquid import PropertyModelError

    m, s = setup()
    with pytest.raises(ValueError, match="absolute height"):
        StratifiedPRNetwork(m.vessels, [replace(m.connections[0], target_height_m=0.8)])
    children = dict(s.vessels)
    b = m.vessels["b"]
    pure = StratifiedPRState.from_dict(b.column.initialize(990000.0, 320.0, [1.0, 0.0])["state"])
    children["b"] = b.initialize(pure, 320.0)
    with pytest.raises(PropertyModelError, match="component support"):
        m.initialize(children)


def test_heated_receiver_reverses_flow_with_same_connection():
    m, s = setup()
    vessels = dict(m.vessels)
    b = vessels["b"]
    vessels["b"] = StratifiedPRFlow(
        b.column, VesselWall(1e6, 1000.0, max_temperature_change_k=10.0)
    )
    children = dict(s.vessels)
    children["b"] = vessels["b"].initialize(children["b"].column, 350.0)
    m = StratifiedPRNetwork(vessels, m.connections)
    state = m.initialize(children)
    forward, one = m._step(state, 0.005, m._commands(None))
    reverse, two = m._step(forward, 0.05, m._commands(None))
    assert sum(one["transfers"][0]["component_rates_mol_s"]) > 0
    assert sum(two["transfers"][0]["component_rates_mol_s"]) < 0
    assert abs(one["energy_residual_j"]) < 1e-6
    assert abs(two["energy_residual_j"]) < 1e-6
    assert reverse.time_s == pytest.approx(0.055)


def test_network_thermal_phase_disappearance_with_closed_connector():
    m, s = setup(two=True)
    vessels = {
        name: StratifiedPRFlow(v.column, VesselWall(1e6, 1000.0, max_temperature_change_k=100.0))
        for name, v in m.vessels.items()
    }
    m = StratifiedPRNetwork(vessels, [replace(m.connections[0], opening=0.0)])
    old = m.initialize(
        {name: vessels[name].initialize(child.column, 320.0) for name, child in s.vessels}
    )
    final, report = m._step(old, 100.0, m._commands(None))
    assert all(len(child.column.phase_roots) == 1 for _, child in final.vessels)
    assert abs(report["energy_residual_j"]) < 1e-6
    assert report["transfers"][0]["component_rates_mol_s"] == [0.0, 0.0]
    assert report["entropy_production_j_k"] >= 0


def test_independent_network_audit_rejects_modified_transfer():
    from dataclasses import asdict

    sys.path.insert(0, str(ROOT / "scripts"))
    from check_stratified_pr_network import audit

    m, s = setup()
    final, step = m._step(s, 0.02, m._commands(None))
    config = {
        "vessels": {name: {"model": v.to_dict()} for name, v in m.vessels.items()},
        "connections": [asdict(p) for p in m.connections],
        "schedule": [{"duration_s": 0.02}],
    }
    interval = {**step, "state": final.to_dict(), "steps": [step]}
    payload = {"config": config, "initial": s.to_dict(), "intervals": [interval]}
    assert audit(config, payload)["accepted"]
    broken = json.loads(json.dumps(payload))
    broken["intervals"][0]["steps"][0]["transfers"][0]["total_energy_rate_w"] += 10.0
    with pytest.raises(ValueError, match="audit mismatch"):
        audit(config, broken)
    broken = json.loads(json.dumps(payload))
    broken["intervals"][0]["energy_residual_j"] += 1.0
    with pytest.raises(ValueError, match="audit mismatch"):
        audit(config, broken)


def test_three_vessel_chain_closes_global_balances():
    m, s = setup()
    vessels = dict(m.vessels)
    states = dict(s.vessels)
    description = vessels["b"].to_dict()
    third = StratifiedPRFlow.from_dict(description)
    column = StratifiedPRState.from_dict(
        third.column.initialize(980000.0, 320.0, [0.5, 0.5])["state"]
    )
    vessels["c"] = third
    states["c"] = third.initialize(column, column.temperature_k)
    connections = [
        m.connections[0],
        StratifiedConnection("second", "b", "c", 1.0, 1.0, 0.05, 1e-10),
    ]
    network = StratifiedPRNetwork(vessels, connections)
    final, report = network._step(network.initialize(states), 0.02, network._commands(None))
    assert len(final.vessels) == 3
    assert all(sum(p["component_rates_mol_s"]) > 0 for p in report["transfers"])
    assert max(map(abs, report["component_residuals_mol"])) < 1e-10
    assert abs(report["energy_residual_j"]) < 1e-6


def test_connection_energy_is_invariant_under_elevation_gauge():
    m, s = setup()
    baseline, one = m._step(s, 0.02, m._commands(None))
    shifted = {}
    children = {}
    delta = 10000.0
    for name, v in m.vessels.items():
        cfg = v.to_dict()
        cfg["column"]["elevation_m"] += delta
        moved = StratifiedPRFlow.from_dict(cfg)
        shifted[name] = moved
        old = dict(s.vessels)[name]
        mass = sum(
            n * c.molar_mass_kg_mol for n, c in zip(old.column.component_moles, m.fluid.components)
        )
        column = replace(
            old.column,
            total_energy_j=old.column.total_energy_j + mass * v.column.gravity_m_s2 * delta,
            model_id=moved.column.model_id,
        )
        children[name] = moved.initialize(column, old.wall_temperature_k)
    other = StratifiedPRNetwork(shifted, m.connections)
    after, two = other._step(other.initialize(children), 0.02, other._commands(None))
    for name in shifted:
        a, b = dict(baseline.vessels)[name], dict(after.vessels)[name]
        assert a.column.temperature_k == pytest.approx(b.column.temperature_k, abs=1e-7)
        assert a.column.component_moles == pytest.approx(b.column.component_moles, abs=1e-9)
    assert one["transfers"][0]["component_rates_mol_s"] == pytest.approx(
        two["transfers"][0]["component_rates_mol_s"], abs=1e-10
    )
    assert abs(two["energy_residual_j"]) < 1e-6


def test_external_boundaries_and_distinct_vessel_base_elevations():
    from surface.component_energy import ComponentEnergyStream
    from surface.compositional_separator import EquilibriumOutlet
    from surface.stratified_pr_flow import StratifiedFeed

    m, s = setup()
    a = m.vessels["a"]
    feed = StratifiedFeed(
        1.0,
        ComponentEnergyStream.from_pt(
            m.fluid, [1e-4, 2e-4], 1010000.0, 320.0, elevation_m=a.column.elevation_m + 1.0
        ),
    )
    source = StratifiedPRFlow(a.column, VesselWall(1000.0, 2.0), feeds=[feed])
    description = m.vessels["b"].column.to_dict()
    description["elevation_m"] += 0.5
    target_column = StratifiedPRColumn.from_dict(description)
    export = EquilibriumOutlet("export", 980000.0, 1e-10, height_m=0.5, diameter_m=0.05)
    target = StratifiedPRFlow(target_column, VesselWall(1000.0, 2.0), outlets=[export])
    b = StratifiedPRState.from_dict(target_column.initialize(990000.0, 320.0, [0.5, 0.5])["state"])
    old_a = dict(s.vessels)["a"].column
    network = StratifiedPRNetwork(
        {"a": source, "b": target}, [replace(m.connections[0], target_height_m=0.5)]
    )
    initial = network.initialize(
        {"a": source.initialize(old_a, 330.0), "b": target.initialize(b, 330.0)}
    )
    final, report = network._step(initial, 0.02, network._commands(None))
    assert report["feed_energy_j"] != 0
    assert report["outlet_energy_j"] != 0
    assert abs(report["energy_residual_j"]) < 1e-6
    assert max(map(abs, report["component_residuals_mol"])) < 1e-10
    assert report["entropy_production_j_k"] >= -1e-7
    network.inspect(final)
