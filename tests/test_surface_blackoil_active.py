"""Distributed resistance, opening regulation and full component tangents."""

import math
from dataclasses import replace

import numpy as np
import pytest
from test_surface_blackoil_network import feeds, network
from test_surface_blackoil_properties import DENSITIES, pvt_data

from surface import BlackOilPressureControl
from surface.blackoil_network import BlackOilProductionNetwork
from surface.blackoil_pipe import BlackOilPipe
from surface.blackoil_properties import BlackOilMixture
from surface.liquid import HydraulicConvergenceError
from surface.process import ComponentStream


def active_network(outlet=1.2e6):
    n = network(outlet)
    return BlackOilProductionNetwork(
        n.model,
        n.nodes,
        [
            replace(p, loss_coefficient=80 if p.name == "export" else 10, valve_opening=0.8)
            for p in n.pipes
        ],
    )


def controller(n, sources=None, target=None):
    sources = feeds() if sources is None else sources
    if target is None:
        target = n.with_controls({"export": 0.7}).solve(sources)["node_pressures_pa"]["header"]
    return BlackOilPressureControl("export", "header", target, 0.5, 1.0)


def test_constant_water_resistance_matches_analytic_energy_and_derivative():
    data = pvt_data()
    data["Bw"] = [1.0] * len(data["Bw"])
    m = BlackOilMixture(data, DENSITIES)
    a = BlackOilPipe("water", 100, 0.08)
    b = replace(a, loss_coefficient=15)
    rates = [0, 2.0, 0]
    ra, rb = [pipe.required_inlet(1.2e6, rates, m) for pipe in (a, b)]
    flux = 2.0 / (math.pi * 0.08**2 / 4)
    dpdk = flux**2 / (2 * DENSITIES.water_kg_m3)
    assert rb["inlet_pressure_pa"] - ra["inlet_pressure_pa"] == pytest.approx(15 * dpdk, abs=1e-5)
    assert rb["pressure_derivatives"]["loss_coefficient"] == pytest.approx(dpdk, rel=1e-9)
    assert abs(rb["mechanical_balance_error_w"]) < 1e-6
    forward = b.run(rb["inlet_pressure_pa"], rates, m)
    assert forward["outlet_pressure_pa"] == pytest.approx(1.2e6, abs=1e-4)


@pytest.mark.parametrize("regulated", [False, True])
def test_component_and_opening_tangents_against_resolved_finite_differences(regulated):
    n = active_network()
    s = feeds()
    control = controller(n) if regulated else None
    solve = lambda net, sources: (
        net.solve(sources) if control is None else control.solve(net, sources)
    )
    r = solve(n, s)
    for source in s:
        base = np.array(list(s[source].mass_rates(DENSITIES).values()))
        for k, component in enumerate(("oil", "water", "gas")):
            h = 1e-5 if k < 2 else 1e-7
            plus, minus = base.copy(), base.copy()
            plus[k] += h
            minus[k] -= h
            sa, sb = dict(s), dict(s)
            sa[source] = ComponentStream.from_mass_rates(*plus, DENSITIES)
            sb[source] = ComponentStream.from_mass_rates(*minus, DENSITIES)
            a, b = solve(n, sa), solve(n, sb)
            for node in ("A", "B", "header"):
                fd = (a["node_pressures_pa"][node] - b["node_pressures_pa"][node]) / (2 * h)
                assert r["pressure_derivatives_per_kg_s"][node][source][component] == pytest.approx(
                    fd, rel=2e-5, abs=0.03
                )
    # A second manually controlled branch remains independent of the regulator.
    for valve in ("a",) if regulated else ("a", "export"):
        h = 1e-5
        a = solve(n.with_controls({valve: 0.8 + h}), s)
        b = solve(n.with_controls({valve: 0.8 - h}), s)
        for node in ("A", "B", "header"):
            fd = (a["node_pressures_pa"][node] - b["node_pressures_pa"][node]) / (2 * h)
            assert r["pressure_derivatives_valve_opening"][node][valve] == pytest.approx(
                fd, rel=2e-5, abs=0.01
            )
    a, b = solve(active_network(1.2e6 + 2), s), solve(active_network(1.2e6 - 2), s)
    for node in ("A", "B", "header"):
        fd = (a["node_pressures_pa"][node] - b["node_pressures_pa"][node]) / 4
        assert r["pressure_derivatives_receiving_pressure"][node] == pytest.approx(
            fd, rel=2e-5, abs=1e-5
        )
    assert abs(r["mechanical_balance_error_w"]) < 1e-5


def test_bounds_serialization_atomic_controls_and_failure_budget():
    n = active_network()
    saved = n.to_dict()
    assert BlackOilProductionNetwork.from_dict(saved).to_dict() == saved
    for target, status in ((0.9e6, "max_opening"), (4.5e6, "min_opening")):
        r = controller(n, target=target).solve(n, feeds())
        assert r["pressure_control"]["status"] == status
        assert abs(r["pressure_derivatives_per_kg_s"]["header"]["A"]["water"]) > 1
    for value in (0, -1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            n.with_controls({"export": value})
    with pytest.raises(ValueError, match="Unknown"):
        n.with_controls({"missing": 1})
    with pytest.raises(ValueError, match="upstream"):
        replace(controller(n), valve="a").solve(n, feeds())
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        replace(controller(n), max_evaluations=3).solve(n, feeds())
    assert n.to_dict() == saved


def test_gathering_schedule_controls_and_json_restart():
    import json

    from test_surface_gathering import config as gathering_config
    from test_surface_gathering import load_gathering

    from surface import FacilityPeriod, run_facility_schedule
    from surface.operation import facility_fingerprint

    c = gathering_config()
    for pipe in c["gathering"]["pipes"]:
        pipe["loss_coefficient"] = 50
    plant = load_gathering(c)
    periods = [
        FacilityPeriod("open", 120),
        FacilityPeriod("restrict", 120, gathering_valve_openings={"export": 0.6}),
        FacilityPeriod("stop", 60, shutdown=True),
        FacilityPeriod("restart", 120),
    ]
    full = run_facility_schedule(plant, periods)
    partial = run_facility_schedule(plant, periods, stop_after=2)
    resumed = run_facility_schedule(
        plant, periods, checkpoint=json.loads(json.dumps(partial["checkpoint"]))
    )
    assert full["checkpoint"] == resumed["checkpoint"]
    assert full["totals"] == resumed["totals"]
    ledger = full["checkpoint"]["ledger"]
    pa = ledger[0]["facility"]["gathering_network"]["node_pressures_pa"]["header"]
    pb = ledger[1]["facility"]["gathering_network"]["node_pressures_pa"]["header"]
    assert pb > pa
    assert (
        ledger[3]["facility"]["gathering_network"]["operating_controls"]["valve_openings"]["export"]
        == 0.6
    )
    assert (
        full["totals"]["accepted_standard_volumes_m3"]
        == full["totals"]["product_standard_volumes_m3"]
    )
    with pytest.raises(ValueError, match="Unknown"):
        run_facility_schedule(
            plant,
            [FacilityPeriod("bad", 10, shutdown=True, gathering_valve_openings={"missing": 0.5})],
        )
    target = (pa + pb) / 2
    c["pressure_control"] = {
        "valve": "export",
        "node": "header",
        "target_pressure_pa": target,
        "min_opening": 0.4,
    }
    controlled = load_gathering(c)
    assert facility_fingerprint(controlled) != facility_fingerprint(plant)
    result = controlled.run()
    assert result["gathering_network"]["pressure_control"]["status"] == "target"
    assert result["gathering_network"]["node_pressures_pa"]["header"] == pytest.approx(
        target, abs=1e-4
    )
    changed = controlled.run(gathering_target_pressure_pa=pb)
    assert changed["gathering_network"]["node_pressures_pa"]["header"] == pytest.approx(
        pb, abs=1e-4
    )
    with pytest.raises(ValueError, match="conflicts"):
        controlled.run(gathering_valve_openings={"export": 0.8})
    with pytest.raises(ValueError, match="require"):
        run_facility_schedule(
            plant.facility,
            [FacilityPeriod("bad", 10, shutdown=True, gathering_valve_openings={"export": 0.8})],
        )
