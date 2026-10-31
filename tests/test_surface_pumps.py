"""Pump/network operating points, affinity scaling and explicit envelope failures."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import EquipmentOperatingError, Liquid, LiquidNetwork, NetworkResult, Node, Pipe, Pump

G = 9.80665
FLUID = Liquid(850, 0.1, vapor_pressure_pa=2000)


def model(pump=None, outlet_pressure=2e5):
    return LiquidNetwork(
        FLUID,
        [
            Node("suction", pressure_pa=2e5),
            Node("discharge"),
            Node("outlet", pressure_pa=outlet_pressure),
        ],
        [Pipe("line", "discharge", "outlet", 100, 0.1)],
        [pump or Pump("booster", "suction", "discharge", 20, 1e6, 0.0044)],
    )


def audit(r):
    assert max(map(abs, r.node_mass_residuals_kg_s.values())) < 1e-8
    assert max(map(abs, r.pump_equation_residuals_pa.values()), default=0) < 1e-3
    assert max(map(abs, r.pipe_equation_residuals_pa.values()), default=0) < 1e-3
    volume = sum(map(abs, r.pipe_volume_rates_m3_s.values())) + sum(
        map(abs, r.pump_volume_rates_m3_s.values())
    )
    assert abs(r.power_balance_error_w) < 1e-3 * volume + 1e-8
    assert sum(r.boundary_mass_rates_kg_s.values()) == pytest.approx(0, abs=1e-8)


def test_pump_intersection_with_laminar_pipeline():
    net = model()
    result = net.solve()
    resistance = 128 * 0.1 * 100 / (math.pi * 0.1**4)
    a = 850 * G * 1e6
    b = resistance
    # Independent positive quadratic root of rho*g*(H0-aq^2)=R*q.
    q = 2 * 850 * G * 20 / (b + math.sqrt(b * b + 4 * a * 850 * G * 20))
    assert 4 * 850 * q / (math.pi * 0.1 * 0.1) < 2000
    assert result.pump_volume_rates_m3_s["booster"] == pytest.approx(q, abs=5e-11)
    assert result.pipe_volume_rates_m3_s["line"] == pytest.approx(q, abs=5e-11)
    hydraulic = 850 * G * q * (20 - 1e6 * q * q)
    assert result.pump_hydraulic_power_w["booster"] == pytest.approx(hydraulic, rel=1e-8)
    assert result.pump_shaft_power_w["booster"] == pytest.approx(hydraulic / 0.75, rel=1e-8)
    audit(result)


def test_speed_affinity_for_fixed_head_boundaries():
    base = Pump("pump", "a", "b", 20, 1e6, 0.004)
    rates = []
    for speed in (0.5, 1.0, 1.5):
        p = replace(base, speed_ratio=speed)
        net = LiquidNetwork(
            FLUID, [Node("a", pressure_pa=2e5), Node("b", pressure_pa=2e5)], [], [p]
        )
        # Equal pressures require zero pump head, at sqrt(H0/a) outside max flow;
        # instead a scaled differential gives equal homologous operating points.
        r = net.solve(boundary_pressures_pa={"b": 2e5 + 850 * G * speed**2 * 12})
        assert r.pump_volume_rates_m3_s["pump"] == pytest.approx(speed * math.sqrt(8e-6), abs=5e-11)
        rates.append(r.pump_shaft_power_w["pump"] / speed**3)
        audit(r)
    assert rates == pytest.approx([rates[0]] * 3, rel=1e-8)


@pytest.mark.parametrize("q", [-0.003, 0, 0.002])
def test_analytic_pump_gradient(q):
    p = Pump("p", "a", "b", 20, 1e6, 0.004)
    h = 1e-11 if q == 0 else 1e-10
    value, grad = p.loss_and_gradient(q, FLUID)
    fd = (p.loss_and_gradient(q + h, FLUID)[0] - p.loss_and_gradient(q - h, FLUID)[0]) / (2 * h)
    assert grad == pytest.approx(fd, rel=2e-8, abs=1)
    assert math.isfinite(value)


def test_json_warm_start_and_legacy_result_compatibility():
    net = model()
    r = net.solve()
    saved = json.loads(json.dumps(r.to_dict()))
    restored = LiquidNetwork.from_dict(json.loads(json.dumps(net.to_dict())))
    warm = restored.solve(initial=NetworkResult.from_dict(saved))
    assert warm.iterations == 0
    assert warm.pump_mass_rates_kg_s == r.pump_mass_rates_kg_s
    for k in list(saved):
        if k.startswith("pump_"):
            saved.pop(k)
    assert NetworkResult.from_dict(saved).pump_mass_rates_kg_s == {}
    audit(warm)


@pytest.mark.parametrize(
    "override,pressure,message",
    [
        ({"max_shaft_power_w": 1.0}, 2e5, "power limit"),
        ({"max_flow_m3_s": 0.0001}, 2e5, "outside forward"),
        ({"npsh_required_m": 30, "inlet_diameter_m": 0.1}, 2e5, "NPSH"),
        ({}, 5e5, "outside forward"),
    ],
)
def test_outside_operating_envelope_is_not_silently_clamped(override, pressure, message):
    p = replace(model().pumps[0], **override)
    with pytest.raises(EquipmentOperatingError, match=message):
        model(p, pressure).solve()


def test_suction_total_head_npsh_and_closed_pump():
    p = replace(model().pumps[0], npsh_required_m=1, inlet_diameter_m=0.1)
    r = model(p).solve()
    q = r.pump_volume_rates_m3_s["booster"]
    v = q / (math.pi * 0.1**2 / 4)
    assert r.pump_npsh_available_m["booster"] == pytest.approx(
        (2e5 - 2000) / (850 * G) + v * v / (2 * G)
    )
    net = LiquidNetwork(
        FLUID,
        [Node("suction", pressure_pa=2e5), Node("discharge", pressure_pa=8e5)],
        [],
        [replace(p, open=False)],
    )
    closed = net.solve()
    assert closed.pump_mass_rates_kg_s == {"booster": 0.0}
    assert closed.pump_shaft_power_w == {}
    audit(closed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"efficiency": 1.1},
        {"max_flow_m3_s": 0.01},
        {"speed_ratio": 0},
        {"npsh_required_m": 1},
        {"max_shaft_power_w": -1},
        {"open": "false"},
    ],
)
def test_bad_equipment_data_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        replace(model().pumps[0], **kwargs)


def test_parallel_pumps_and_series_pipeline():
    base = model().pumps[0]
    net = model()
    twins = LiquidNetwork(FLUID, net.nodes, net.pipes, [base, replace(base, name="parallel")])
    r = twins.solve()
    q = r.pump_volume_rates_m3_s["booster"]
    resistance = 128 * 0.1 * 100 / (math.pi * 0.1**4)
    a = 850 * G * 1e6
    expected = (
        2
        * 850
        * G
        * 20
        / (2 * resistance + math.sqrt((2 * resistance) ** 2 + 4 * a * 850 * G * 20))
    )
    assert q == pytest.approx(expected, abs=5e-11)
    assert r.pump_volume_rates_m3_s["parallel"] == pytest.approx(q, abs=1e-12)
    assert r.pipe_volume_rates_m3_s["line"] == pytest.approx(2 * q, abs=1e-12)
    audit(r)


def test_shutoff_zero_flow_retains_head_without_false_power():
    pump = model().pumps[0]
    net = LiquidNetwork(FLUID, [Node("suction", pressure_pa=2e5), Node("discharge")], [], [pump])
    r = net.solve()
    assert r.pressures_pa["discharge"] == pytest.approx(2e5 + 850 * G * 20, abs=1e-7)
    assert r.pump_volume_rates_m3_s["booster"] == 0
    # Hydraulic-efficiency model excludes real motor idle/mechanical losses.
    assert r.pump_shaft_power_w["booster"] == 0
    audit(r)


def test_shared_names_across_pipe_and_pump_are_rejected():
    net = model()
    with pytest.raises(ValueError, match="Duplicate branch"):
        LiquidNetwork(FLUID, net.nodes, net.pipes, [replace(net.pumps[0], name="line")])


def test_independent_epanet_pump_and_pipe_reference():
    fixture = json.loads(
        (Path(__file__).parent / "data/surface/epanet_22_booster.json").read_text()
    )
    network = LiquidNetwork.from_dict(fixture["input"])
    r = network.solve()
    rates = dict(r.pipe_mass_rates_kg_s, **r.pump_mass_rates_kg_s)
    for name, flow_lps in fixture["epanet_flows_lps"].items():
        assert rates[name] == pytest.approx(flow_lps * 850 / 1000, abs=2e-4)
    for node in network.nodes:
        head = node.elevation_m + r.pressures_pa[node.name] / (850 * G)
        assert head == pytest.approx(fixture["epanet_heads_m"][node.name], abs=0.002)
    audit(r)


def test_operating_controls_are_transactional_and_validate_disconnection():
    net = model()
    config = net.to_dict()
    r = net.solve(pump_speed_ratios={"booster": 0.8})
    assert net.to_dict() == config
    assert r.pump_volume_rates_m3_s["booster"] < net.solve().pump_volume_rates_m3_s["booster"]
    warm = net.solve(pump_speed_ratios={"booster": 0.8}, initial=r)
    assert warm.iterations == 0
    audit(warm)
    with pytest.raises(ValueError, match="Unknown pump"):
        net.solve(pump_speed_ratios={"typo": 1})
    closed = net.solve(pump_status={"booster": False})
    assert closed.pump_mass_rates_kg_s == {"booster": 0.0}
    assert closed.pipe_mass_rates_kg_s["line"] == pytest.approx(0, abs=1e-10)
    assert net.to_dict() == config
    with pytest.raises(ValueError, match="No pressure boundary"):
        LiquidNetwork(
            FLUID, [Node("suction", pressure_pa=2e5), Node("discharge")], [], net.pumps
        ).solve(pump_status={"booster": False})
