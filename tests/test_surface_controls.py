"""Non-return complementarity, equipment transitions and pressure control."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import Liquid, LiquidNetwork, Node, Pipe, Pump

WATER = Liquid(1000, 0.001)


def audit(r):
    assert max(map(abs, r.node_mass_residuals_kg_s.values()), default=0) < 1e-8
    assert max(map(abs, r.pipe_equation_residuals_pa.values()), default=0) < 1e-3
    assert max(map(abs, r.pump_equation_residuals_pa.values()), default=0) < 1e-3
    for name, driving in r.check_valve_driving_pressure_pa.items():
        rates = dict(r.pipe_volume_rates_m3_s, **r.pump_volume_rates_m3_s)
        assert rates[name] >= 0
        assert driving <= 1e-3
        if r.branch_states[name] == "blocked":
            assert rates[name] == 0
    assert abs(r.power_balance_error_w) < 1e-5


def test_check_valve_reverses_state_and_reopens_from_saved_state():
    net = LiquidNetwork(
        WATER,
        [Node("a", pressure_pa=3e5), Node("b", pressure_pa=2e5)],
        [Pipe("cv", "a", "b", 100, 0.1, check_valve=True)],
    )
    opened = net.solve()
    assert opened.pipe_mass_rates_kg_s["cv"] > 0
    blocked = net.solve(boundary_pressures_pa={"a": 1e5}, initial=opened)
    assert blocked.pipe_mass_rates_kg_s["cv"] == 0
    assert blocked.branch_states["cv"] == "blocked"
    reopened = net.solve(initial=blocked)
    assert reopened.pipe_mass_rates_kg_s["cv"] == pytest.approx(opened.pipe_mass_rates_kg_s["cv"])
    for r in (opened, blocked, reopened):
        audit(r)


def test_blocked_branch_with_live_bypass_does_not_leak():
    net = LiquidNetwork(
        WATER,
        [Node("a", pressure_pa=2e5), Node("b"), Node("out", pressure_pa=3e5)],
        [Pipe("cv", "a", "b", 100, 0.1, check_valve=True), Pipe("bypass", "out", "b", 100, 0.1)],
    )
    r = net.solve()
    assert r.pipe_mass_rates_kg_s["cv"] == 0
    assert r.pressures_pa["b"] == pytest.approx(3e5)
    audit(r)


def test_pump_against_excess_backpressure_blocks_and_restarts():
    pump = Pump("p", "a", "b", 20, 1e6, 0.0044, check_valve=True)
    net = LiquidNetwork(WATER, [Node("a", pressure_pa=2e5), Node("b", pressure_pa=5e5)], [], [pump])
    blocked = net.solve()
    assert blocked.pump_mass_rates_kg_s["p"] == 0
    assert blocked.pump_shaft_power_w == {}
    opened = net.solve(boundary_pressures_pa={"b": 3e5}, initial=blocked)
    assert opened.pump_mass_rates_kg_s["p"] > 0
    for r in (blocked, opened):
        audit(r)


def test_parallel_pumps_weaker_pump_does_not_reverse():
    strong = Pump("strong", "a", "b", 40, 1e6, 0.006, check_valve=True)
    weak = replace(strong, name="weak", shutoff_head_m=10, max_flow_m3_s=0.003)
    net = LiquidNetwork(
        WATER,
        [Node("a", pressure_pa=2e5), Node("b"), Node("out", pressure_pa=5e5)],
        [Pipe("line", "b", "out", 100, 0.1)],
        [strong, weak],
    )
    r = net.solve()
    assert r.branch_states["weak"] == "blocked"
    assert r.pump_mass_rates_kg_s["weak"] == 0
    assert r.pump_mass_rates_kg_s["strong"] > 0
    audit(r)


def pressure_control_model():
    # Laminar pipe to a sink with a prescribed 0.85 kg/s withdrawal:
    # q=0.001 m3/s, so controlled pressure is an analytic function of speed.
    fluid = Liquid(850, 0.1)
    pump = Pump("pump", "supply", "delivery", 20, 1e6, 0.0044)
    return LiquidNetwork(
        fluid,
        [Node("supply", pressure_pa=2e5), Node("delivery", source_mass_rate_kg_s=-0.85)],
        [],
        [pump],
    )


def test_pressure_controller_target_limits_and_restart():
    from surface import PumpPressureControl

    net = pressure_control_model()
    target = 2e5 + 850 * 9.80665 * (0.9**2 * 20 - 1)
    control = PumpPressureControl("pump", "delivery", target, 0.5, 1.2, pressure_tolerance_pa=0.01)
    r = control.solve(net)
    assert r.status == "target"
    assert r.speed_ratio == pytest.approx(0.9, abs=1e-7)
    assert abs(r.pressure_error_pa) <= 0.01
    assert r.to_dict()["network"]["pump_mass_rates_kg_s"]["pump"] == pytest.approx(0.85)
    again = control.solve(net, initial=r.network)
    assert again.speed_ratio == r.speed_ratio
    low = replace(control, target_pressure_pa=2e5).solve(net)
    high = replace(control, target_pressure_pa=8e5).solve(net)
    assert low.status == "min_speed" and low.pressure_error_pa > 0
    assert high.status == "max_speed" and high.pressure_error_pa < 0
    assert net.pumps[0].speed_ratio == 1


def test_pressure_regulator_rejects_fixed_node_and_preserves_safety_errors():
    from surface import EquipmentOperatingError, PumpPressureControl

    net = pressure_control_model()
    with pytest.raises(ValueError, match="fixed pressure"):
        PumpPressureControl("pump", "supply", 3e5, 0.5, 1.2).solve(net)
    restricted = LiquidNetwork(
        net.fluid, net.nodes, [], [replace(net.pumps[0], max_shaft_power_w=1)]
    )
    with pytest.raises(EquipmentOperatingError, match="power limit"):
        PumpPressureControl("pump", "delivery", 3e5, 0.5, 1.2).solve(restricted)


def test_isolated_zero_demand_does_not_get_an_invented_pressure():
    from surface import EquipmentOperatingError

    net = LiquidNetwork(
        WATER,
        [Node("a", pressure_pa=2e5), Node("b")],
        [Pipe("cv", "a", "b", 100, 0.1, check_valve=True)],
    )
    with pytest.raises(EquipmentOperatingError, match="undetermined"):
        net.solve()


def test_independent_parallel_pump_states_against_epanet():
    import json

    fixture = json.loads(
        (Path(__file__).parent / "data/surface/epanet_22_parallel_pumps.json").read_text()
    )
    net = LiquidNetwork.from_dict(fixture["input"])
    r = net.solve()
    rates = dict(r.pipe_mass_rates_kg_s, **r.pump_mass_rates_kg_s)
    for name, reference in fixture["epanet_flows_lps"].items():
        assert rates[name] == pytest.approx(reference, abs=3e-5)
        assert (r.branch_states[name] == "open") == bool(fixture["epanet_link_status"][name])
    for node in net.nodes:
        head = node.elevation_m + r.pressures_pa[node.name] / (1000 * 9.80665)
        assert head == pytest.approx(fixture["epanet_heads_m"][node.name], abs=0.001)
    assert fixture["epanet_warnings"] == [{"function": "EN_solveH", "code": 4}]
    audit(r)


def test_pressure_regulator_starts_with_blocked_pump():
    from surface import PumpPressureControl

    pump = Pump("p", "a", "b", 20, 1e6, 0.0044, check_valve=True)
    net = LiquidNetwork(
        WATER,
        [Node("a", pressure_pa=2e5), Node("b"), Node("out", pressure_pa=3e5)],
        [Pipe("line", "b", "out", 100, 0.1)],
        [pump],
    )
    r = PumpPressureControl("p", "b", 301000, 0.5, 1.0, pressure_tolerance_pa=0.01).solve(net)
    assert r.status == "target"
    assert r.network.branch_states["p"] == "open"
    assert abs(r.pressure_error_pa) <= 0.01
    audit(r.network)


def test_nonreturn_valve_uses_total_head_including_elevation():
    net = LiquidNetwork(
        WATER,
        [Node("hill", elevation_m=20, pressure_pa=1e5), Node("valley", pressure_pa=2e5)],
        [Pipe("cv", "hill", "valley", 100, 0.1, check_valve=True)],
    )
    r = net.solve()
    assert r.pipe_mass_rates_kg_s["cv"] > 0
    assert r.branch_states["cv"] == "open"
    audit(r)


def test_pressure_regulator_iteration_limit_is_an_error():
    from surface import HydraulicConvergenceError, PumpPressureControl

    net = pressure_control_model()
    target = 2e5 + 850 * 9.80665 * (0.9**2 * 20 - 1)
    control = PumpPressureControl(
        "pump", "delivery", target, 0.5, 1.2, pressure_tolerance_pa=0.001, max_evaluations=3
    )
    with pytest.raises(HydraulicConvergenceError, match="evaluation limit"):
        control.solve(net)


def test_impossible_demand_behind_check_valve_never_passes():
    from surface import HydraulicConvergenceError

    net = LiquidNetwork(
        WATER,
        [Node("supply", pressure_pa=2e5), Node("reverse_source", source_mass_rate_kg_s=1)],
        [Pipe("cv", "supply", "reverse_source", 100, 0.1, check_valve=True)],
    )
    with pytest.raises(HydraulicConvergenceError):
        net.solve()
