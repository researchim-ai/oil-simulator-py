"""Independent equipment operating points and valve/orifice limits."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    EquipmentCurve,
    EquipmentOperatingError,
    Liquid,
    LiquidNetwork,
    Node,
    Pump,
    ThrottleValve,
)

WATER = Liquid(1000, 0.001, 2300)


def mapped_pump():
    return Pump(
        "p",
        "a",
        "b",
        20,
        1e6,
        0.004,
        head_curve=EquipmentCurve((0, 0.002, 0.004), (20, 16, 8)),
        efficiency_curve=EquipmentCurve((0, 0.002, 0.004), (0.5, 0.8, 0.6)),
        npsh_curve=EquipmentCurve((0, 0.002, 0.004), (1, 2, 4)),
        inlet_diameter_m=0.1,
    )


def test_maps_determine_head_power_npsh_and_serialize():
    p = mapped_pump()
    # 16 m gain at nominal flow 0.002; homologous point at speed 1.5.
    net = LiquidNetwork(
        WATER,
        [Node("a", pressure_pa=2e5), Node("b", pressure_pa=2e5 + 1000 * 9.80665 * 16 * 1.5**2)],
        [],
        [replace(p, speed_ratio=1.5)],
    )
    r = net.solve()
    assert r.pump_volume_rates_m3_s["p"] == pytest.approx(0.003, abs=1e-12)
    assert r.pump_shaft_power_w["p"] == pytest.approx(1000 * 9.80665 * 0.003 * 16 * 1.5**2 / 0.8)
    restored = LiquidNetwork.from_dict(json.loads(json.dumps(net.to_dict())))
    assert restored.solve(initial=r).iterations == 0
    low_suction = replace(p, npsh_curve=EquipmentCurve((0, 0.004), (100, 100)))
    with pytest.raises(EquipmentOperatingError, match="NPSH"):
        LiquidNetwork(WATER, net.nodes, [], [replace(low_suction, speed_ratio=1.5)]).solve()


@pytest.mark.parametrize("q", [-0.003, -0.001, 0, 0.001, 0.003])
def test_map_pressure_derivative(q):
    p = mapped_pump()
    h = 1e-9
    _, grad = p.loss_and_gradient(q, WATER)
    fd = (p.loss_and_gradient(q + h, WATER)[0] - p.loss_and_gradient(q - h, WATER)[0]) / (2 * h)
    assert grad == pytest.approx(fd, rel=1e-7)


@pytest.mark.parametrize("opening", [0.1, 0.5, 1])
def test_orifice_flow_against_bernoulli_and_opening_control(opening):
    valve = ThrottleValve("v", "a", "b", 0.02, opening=opening)
    net = LiquidNetwork(
        WATER, [Node("a", pressure_pa=3e5), Node("b", pressure_pa=2e5)], [], valves=[valve]
    )
    r = net.solve()
    q = 0.62 * math.pi * 0.02**2 / 4 * opening * math.sqrt(2 * 1e5 / 1000)
    assert r.valve_volume_rates_m3_s["v"] == pytest.approx(q, abs=1e-12)
    assert r.valve_dissipation_w["v"] == pytest.approx(q * 1e5, rel=1e-8)
    assert abs(r.power_balance_error_w) < 1e-5
    closed = net.solve(valve_openings={"v": 0}, initial=r)
    assert closed.valve_mass_rates_kg_s["v"] == 0
    assert closed.branch_states["v"] == "closed"
    assert net.valves[0].opening == opening
    assert (
        LiquidNetwork.from_dict(json.loads(json.dumps(net.to_dict()))).solve(initial=r).iterations
        == 0
    )
    with pytest.raises(ValueError, match="Unknown valve"):
        net.solve(valve_openings={"typo": 0.5})


def test_orifice_reverse_and_nonreturn_states():
    v = ThrottleValve("v", "a", "b", 0.02)
    net = LiquidNetwork(
        WATER, [Node("a", pressure_pa=2e5), Node("b", pressure_pa=3e5)], [], valves=[v]
    )
    assert net.solve().valve_mass_rates_kg_s["v"] < 0
    r = LiquidNetwork(WATER, net.nodes, [], valves=[replace(v, check_valve=True)]).solve()
    assert r.valve_mass_rates_kg_s["v"] == 0 and r.branch_states["v"] == "blocked"


@pytest.mark.parametrize(
    "make",
    [
        lambda: EquipmentCurve((0, 0), (1, 2)),
        lambda: EquipmentCurve((0, 1), (1, float("nan"))),
        lambda: replace(mapped_pump(), efficiency_curve=EquipmentCurve((0, 0.004), (0.5, 1.1))),
        lambda: replace(mapped_pump(), head_curve=EquipmentCurve((0, 0.004), (20, 21))),
        lambda: replace(
            mapped_pump(), npsh_curve=EquipmentCurve((0, 0.004), (1, 2)), inlet_diameter_m=None
        ),
        lambda: ThrottleValve("v", "a", "b", 0.1, opening=1.1),
    ],
)
def test_bad_maps_and_valves(make):
    with pytest.raises((ValueError, TypeError)):
        make()


def test_mapped_pump_valve_and_efficiency_against_epanet():
    fixture = json.loads(
        (Path(__file__).parent / "data/surface/epanet_22_mapped_pump_valve.json").read_text()
    )
    net = LiquidNetwork.from_dict(fixture["input"])
    r = net.solve()
    rates = dict(r.pipe_mass_rates_kg_s, **r.pump_mass_rates_kg_s, **r.valve_mass_rates_kg_s)
    for name, value in fixture["epanet_flows_lps"].items():
        assert rates[name] == pytest.approx(value, abs=1e-10)
    for node in net.nodes:
        assert r.pressures_pa[node.name] / (1000 * 9.80665) + node.elevation_m == pytest.approx(
            fixture["epanet_heads_m"][node.name], abs=0.01
        )
    for name, power in fixture["epanet_shaft_power_w"].items():
        assert r.pump_shaft_power_w[name] == pytest.approx(power, rel=0.002)
