"""Local liquid loss: analytic limit, PVT tangents and strict phase/safety gates."""

import math
from dataclasses import replace

import numpy as np
import pytest
from test_surface_blackoil_properties import DENSITIES, model, pvt_data

from surface import (
    BlackOilLiquidValve,
    BlackOilMixture,
    BlackOilPressureControl,
    BlackOilProductionNetwork,
    ComponentStream,
    EquipmentOperatingError,
    ProductionNode,
    ProductionPipe,
    ProductionValve,
    PropertyModelError,
)


def valve(**kwargs):
    return BlackOilLiquidValve("trim", 0.05, 50, 3000, 0.8, **kwargs)


def feeds():
    return {
        "A": ComponentStream(0.0007, 0.0014, 0.0105),
        "B": ComponentStream(0.0003, 0.0006, 0.0045),
    }


def network(outlet=1.2e6):
    return BlackOilProductionNetwork(
        model(),
        [
            ProductionNode("A"),
            ProductionNode("B"),
            ProductionNode("header"),
            ProductionNode("valve_out"),
            ProductionNode("receiver", pressure_pa=outlet),
        ],
        [
            ProductionPipe("a", "A", "header", 100, 0.06),
            ProductionPipe("b", "B", "header", 100, 0.06),
            ProductionPipe("export", "valve_out", "receiver", 100, 0.06),
        ],
        [ProductionValve("trim", "header", "valve_out", 0.05, 50, 3000, 0.8, 0.8)],
    )


def test_water_analytic_inverse_and_hydraulic_power():
    data = pvt_data()
    data["Bw"] = [1, 1, 1]
    m = BlackOilMixture(data, DENSITIES)
    v = valve()
    r = v.required_inlet(1.2e6, [0, 2, 0], m)
    dp = v.loss_coefficient * (2 / (math.pi * v.diameter_m**2 / 4)) ** 2 / (2 * 1000)
    assert r["inlet_pressure_pa"] == pytest.approx(1.2e6 + dp, abs=1e-6)
    assert r["friction_dissipation_w"] == pytest.approx(dp * 0.002, rel=1e-12)
    assert r["pressure_derivatives"]["loss_coefficient"] == pytest.approx(dp / 50, rel=1e-12)
    assert r["pressure_derivatives"]["water_kg_s"] == pytest.approx(dp, rel=1e-12)
    back = v.run(r["inlet_pressure_pa"], [0, 2, 0], m)
    assert back["outlet_pressure_pa"] == pytest.approx(1.2e6, abs=1e-6)
    assert r["vapor_pressure_margin_pa"] > 0


@pytest.mark.parametrize("inverse", [False, True])
def test_port_and_component_tangents(inverse):
    v = valve()
    m = model()
    rates = np.array([0.85, 2.0, 0.012])
    solve = v.required_inlet if inverse else v.run
    result = solve(1.5e6, rates, m)
    port = "inlet_pressure_pa" if inverse else "outlet_pressure_pa"
    for j, c in enumerate(("oil", "water", "gas")):
        h = 1e-5 if j < 2 else 1e-7
        a, b = rates.copy(), rates.copy()
        a[j] += h
        b[j] -= h
        fd = (solve(1.5e6, a, m)[port] - solve(1.5e6, b, m)[port]) / (2 * h)
        assert result["pressure_derivatives"][c + "_kg_s"] == pytest.approx(fd, rel=1e-5, abs=0.01)
    fd = (solve(1.5e6 + 2, rates, m)[port] - solve(1.5e6 - 2, rates, m)[port]) / 4
    assert result["pressure_derivatives"]["boundary_pressure_pa"] == pytest.approx(fd, rel=1e-7)


def test_free_gas_flashing_and_cavitation_are_not_silently_modeled():
    m = model()
    with pytest.raises(PropertyModelError, match="free gas|flashing"):
        valve().required_inlet(1.2e6, [0.85, 2, 0.12], m)
    # Dissolved at the inlet, liberated at the outlet.
    with pytest.raises(PropertyModelError, match="free gas|flashing"):
        replace(valve(), loss_coefficient=500).run(1.2e6, [0.85, 2, 0.024], m)
    with pytest.raises(EquipmentOperatingError, match="vena-contracta"):
        replace(valve(), pressure_recovery_factor=0.05).run(1.2e6, [0, 2, 0], m)
    with pytest.raises(PropertyModelError, match="positive total"):
        valve().run(1.2e6, [0, 0, 0], m)
    with pytest.raises(ValueError):
        replace(valve(), pressure_recovery_factor=1.1)


@pytest.mark.parametrize("regulated", [False, True])
def test_tree_and_regulator_component_and_opening_derivatives(regulated):
    n = network()
    s = feeds()
    target = n.with_controls({"trim": 0.7}).solve(s)["node_pressures_pa"]["header"]
    control = BlackOilPressureControl("trim", "header", target, 0.6, 1.0)
    solve = lambda net, sources: control.solve(net, sources) if regulated else net.solve(sources)
    r = solve(n, s)
    for source in s:
        base = np.array(list(s[source].mass_rates(DENSITIES).values()))
        for k, c in enumerate(("oil", "water", "gas")):
            h = 1e-5 if k < 2 else 1e-7
            a, b = base.copy(), base.copy()
            a[k] += h
            b[k] -= h
            sa, sb = dict(s), dict(s)
            sa[source] = ComponentStream.from_mass_rates(*a, DENSITIES)
            sb[source] = ComponentStream.from_mass_rates(*b, DENSITIES)
            ra, rb = solve(n, sa), solve(n, sb)
            for node in ("A", "B", "header"):
                fd = (ra["node_pressures_pa"][node] - rb["node_pressures_pa"][node]) / (2 * h)
                assert r["pressure_derivatives_per_kg_s"][node][source][c] == pytest.approx(
                    fd, rel=2e-5, abs=0.03
                )
    if not regulated:
        h = 1e-5
        a = solve(n.with_controls({"trim": 0.8 + h}), s)
        b = solve(n.with_controls({"trim": 0.8 - h}), s)
        for node in ("A", "B", "header"):
            fd = (a["node_pressures_pa"][node] - b["node_pressures_pa"][node]) / (2 * h)
            assert r["pressure_derivatives_valve_opening"][node]["trim"] == pytest.approx(
                fd, rel=1e-5, abs=0.01
            )
    else:
        assert r["pressure_control"]["status"] == "target"
    saved = n.to_dict()
    assert BlackOilProductionNetwork.from_dict(saved).to_dict() == saved
    assert abs(r["mechanical_balance_error_w"]) < 1e-5
    with pytest.raises(ValueError):
        n.with_controls({"trim": 0})
    assert n.to_dict() == saved


def test_valve_ports_must_have_equal_elevation():
    n = network()
    with pytest.raises(PropertyModelError, match="equal elevation"):
        BlackOilProductionNetwork(
            n.model,
            [replace(p, elevation_m=1) if p.name == "header" else p for p in n.nodes],
            [p for p in n.pipes if isinstance(p, ProductionPipe)],
            [p for p in n.valves if isinstance(p, ProductionValve)],
        )


def test_flash_inside_vena_is_rejected_even_when_both_ports_are_liquid():
    m = model()
    rates = [0.85, 2, 0.024]
    safe = valve().run(1.2e6, rates, m)
    assert safe["inlet_state"]["phase_mass_rates_kg_s"][2] == 0
    assert safe["outlet_state"]["phase_mass_rates_kg_s"][2] == 0
    with pytest.raises(PropertyModelError, match="free gas|flashing"):
        replace(valve(), pressure_recovery_factor=0.3).run(1.2e6, rates, m)


def test_local_density_approximation_has_a_strict_domain_guard():
    data = pvt_data()
    data["Bw"] = [1, 0.8, 0.2]
    m = BlackOilMixture(data, DENSITIES)
    with pytest.raises(PropertyModelError, match="density change"):
        replace(valve(), loss_coefficient=500).run(1.5e6, [0, 2, 0], m)
