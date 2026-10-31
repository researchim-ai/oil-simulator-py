"""Analytic equipment/control tangents against independently perturbed solves."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surface import Liquid, LiquidNetwork, Pump, PumpPressureControl


def network():
    c = json.loads((ROOT / "configs/fim_water_surface_active.json").read_text())
    return LiquidNetwork.from_dict(c["simulation"]["physical_surface_network"]["network"])


def control(target=6e6):
    return PumpPressureControl(
        "booster",
        "header",
        target,
        0.5,
        1.2,
        pressure_tolerance_pa=1e-4,
        method="newton",
        response_direction="auto",
    )


OPTIONS = {
    "source_mass_rates_kg_s": {"bottom_1": 0.4, "bottom_2": 0.6},
    "pressure_tolerance_pa": 1e-6,
    "mass_tolerance_kg_s": 1e-10,
}


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("q", [-0.001, 0.0, 0.001])
def test_pump_speed_partial_matches_finite_difference(mapped, q):
    data = {
        "name": "p",
        "start": "a",
        "end": "b",
        "shutoff_head_m": 300.0,
        "curve_coefficient_s2_m5": 2e6,
        "max_flow_m3_s": 0.01,
        "speed_ratio": 0.8,
    }
    if mapped:
        data["head_curve"] = {"flow_m3_s": [0.0, 0.005, 0.01], "values": [300.0, 210.0, 100.0]}
    pump = Pump(**data)
    fluid = Liquid(1000, 0.001)
    h = 1e-5
    fd = (
        replace(pump, speed_ratio=0.8 + h).loss_and_gradient(q, fluid)[0]
        - replace(pump, speed_ratio=0.8 - h).loss_and_gradient(q, fluid)[0]
    ) / (2 * h)
    assert pump.loss_speed_gradient(q, fluid) == pytest.approx(fd, rel=1e-9)


def test_effective_controls_are_reused_by_source_and_speed_tangents():
    net = network()
    kwargs = {"pump_speed_ratios": {"booster": 0.9}, "valve_openings": {"export_valve": 0.6}}
    result = net.solve(**OPTIONS, **kwargs)
    assert result.operating_controls["pump_speed_ratios"] == {"booster": 0.9}
    assert net.pumps[0].speed_ratio == 1.0
    names = ["bottom_1", "bottom_2", "header"]
    K = net.pressure_sensitivities(result, ["bottom_1", "bottom_2"], names)
    for j, node in enumerate(["bottom_1", "bottom_2"]):
        h = 1e-4
        plus = dict(OPTIONS["source_mass_rates_kg_s"])
        minus = dict(plus)
        plus[node] += h
        minus[node] -= h
        a = net.solve(**{**OPTIONS, "source_mass_rates_kg_s": plus}, **kwargs)
        b = net.solve(**{**OPTIONS, "source_mass_rates_kg_s": minus}, **kwargs)
        fd = np.array([(a.pressures_pa[n] - b.pressures_pa[n]) / (2 * h) for n in names])
        np.testing.assert_allclose(K[:, j], fd, rtol=2e-7, atol=0.03)
    h = 1e-5
    a = net.solve(**OPTIONS, **{**kwargs, "pump_speed_ratios": {"booster": 0.9 + h}})
    b = net.solve(**OPTIONS, **{**kwargs, "pump_speed_ratios": {"booster": 0.9 - h}})
    fd = np.array([(a.pressures_pa[n] - b.pressures_pa[n]) / (2 * h) for n in names])
    np.testing.assert_allclose(
        net.pump_speed_sensitivities(result, ["booster"], names)[:, 0], fd, rtol=2e-8, atol=0.02
    )


@pytest.mark.parametrize("target,status", [(6e6, "target"), (9e6, "min_speed"), (2e6, "max_speed")])
def test_regulated_and_saturated_tangents_include_correct_speed_response(target, status):
    net = network()
    c = control(target)
    r = c.solve(net, **OPTIONS)
    assert r.status == status and r.evaluations <= 15
    names = ["bottom_1", "bottom_2", "header"]
    K = c.pressure_sensitivities(net, r, ["bottom_1", "bottom_2"], names)
    for j, node in enumerate(["bottom_1", "bottom_2"]):
        h = 1e-3
        plus = dict(OPTIONS["source_mass_rates_kg_s"])
        minus = dict(plus)
        plus[node] += h
        minus[node] -= h
        a = c.solve(net, **{**OPTIONS, "source_mass_rates_kg_s": plus})
        b = c.solve(net, **{**OPTIONS, "source_mass_rates_kg_s": minus})
        fd = np.array(
            [(a.network.pressures_pa[n] - b.network.pressures_pa[n]) / (2 * h) for n in names]
        )
        np.testing.assert_allclose(K[:, j], fd, rtol=4e-5, atol=0.3)
    if status == "target":
        assert abs(K[-1]).max() < 1e-5
        fixed = net.pressure_sensitivities(r.network, ["bottom_1", "bottom_2"], names)
        assert abs(fixed[-1]).max() > 1000  # omitting speed response would be physically wrong


def test_newton_regulator_needs_fewer_network_solves_than_bisection():
    net = network()
    fast = control().solve(net, **OPTIONS)
    slow = replace(control(), method="bisect").solve(net, **OPTIONS)
    assert fast.evaluations < slow.evaluations / 2
    assert abs(fast.pressure_error_pa) <= 1e-4 and abs(slow.pressure_error_pa) <= 1e-4


def test_closed_pump_between_boundaries_has_zero_speed_sensitivity():
    from surface import Node

    p = Pump("off", "a", "b", 20.0, 1e6, 0.004, open=False)
    net = LiquidNetwork(
        Liquid(1000, 0.001), [Node("a", pressure_pa=2e5), Node("b", pressure_pa=3e5)], [], [p]
    )
    r = net.solve()
    np.testing.assert_array_equal(
        net.pump_speed_sensitivities(r, ["off"], ["a", "b"]), np.zeros((2, 1))
    )
