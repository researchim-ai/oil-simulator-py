"""Wall/fluid energy conservation, analytic RC limits, restart and protection."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    ConservationError,
    DynamicVessel,
    DynamicVesselNetwork,
    NetworkTrip,
    PengRobinsonFluid,
    PengRobinsonMixture,
    PropertyModelError,
    VesselState,
    VesselWall,
    advance_protected_network,
    run_network_schedule,
)
from surface.peng_robinson import R


def setup(ambient_ua=0.0):
    d = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_thermal_mixture.json").read_text()
    )["fluid"]
    f = PengRobinsonMixture(
        tuple(PengRobinsonFluid(**c) for c in d["components"]), d["binary_interactions"]
    )
    v = DynamicVessel("tank", 1, 1, 1000)
    wall = VesselWall(10.0, 0.1, ambient_ua, 280.0)
    net = DynamicVesselNetwork({"tank": v}, walls={"tank": wall})
    s = net.initialize(
        f,
        {
            "tank": {
                "pressure_pa": 100,
                "temperature_k": 300,
                "mole_fractions": [0.5, 0.5],
                "wall_temperature_k": 350,
            }
        },
    )
    return f, v, wall, net, s


def total(s):
    return s.internal_energy_j + (s.wall_energy_j or 0.0)


def test_closed_pair_transfers_heat_without_creating_energy():
    f, _v, wall, net, s = setup()
    r = net.advance(s, f, 1, initial_dt_s=1)
    final = VesselState(**r["states"]["tank"])
    assert 300 < final.temperature_k < final.wall_temperature_k < 350
    assert total(final) == pytest.approx(total(s["tank"]), abs=1e-6)
    assert r["steps"][0]["heat_j"] == 0
    assert r["steps"][0]["vessels"]["tank"]["wall_to_fluid_heat_j"] > 0
    assert r["steps"][0]["vessels"]["tank"]["entropy_production_j_k"] >= -1e-6
    assert final.wall_energy_j == pytest.approx(wall.energy(final.wall_temperature_k), abs=1e-9)


def test_implicit_step_matches_independent_two_body_ideal_limit():
    f, _v, wall, net, s = setup()
    cf = sum(s["tank"].component_moles) * (sum(c.ideal_cp_j_mol_k for c in f.components) / 2 - R)
    dt = 1.0
    difference = 50 / (1 + dt * wall.fluid_ua_w_k * (1 / cf + 1 / wall.heat_capacity_j_k))
    tf = 300 + wall.heat_capacity_j_k / (cf + wall.heat_capacity_j_k) * (50 - difference)
    tw = tf + difference
    r = net.advance(s, f, dt, initial_dt_s=dt)
    assert r["states"]["tank"]["temperature_k"] == pytest.approx(tf, abs=1e-4)
    assert r["states"]["tank"]["wall_temperature_k"] == pytest.approx(tw, abs=1e-4)


def test_ambient_energy_and_first_order_time_refinement():
    f, _v, wall, net, s = setup(0.2)
    cf = sum(s["tank"].component_moles) * (sum(c.ideal_cp_j_mol_k for c in f.components) / 2 - R)
    a = np.array(
        [
            [-wall.fluid_ua_w_k / cf, wall.fluid_ua_w_k / cf],
            [wall.fluid_ua_w_k / 10, -(wall.fluid_ua_w_k + 0.2) / 10],
        ]
    )
    target = 280 + expm(a * 2) @ np.array([20, 70])
    errors = []
    for dt in (0.5, 0.25, 0.125):
        r = net.advance(s, f, 2, initial_dt_s=dt)
        end = VesselState(**r["states"]["tank"])
        errors.append(abs(end.temperature_k - target[0]))
        heat = sum(step["ambient_heat_j"] for step in r["steps"])
        assert total(end) - total(s["tank"]) == pytest.approx(heat, abs=1e-6)
        assert heat < 0
    assert errors[1] < 0.65 * errors[0] and errors[2] < 0.65 * errors[1]


def test_zero_fluid_ua_leaves_fluid_unchanged_but_wall_evolves():
    f, v, wall, _net, _s = setup(0.2)
    wall = replace(wall, fluid_ua_w_k=0)
    net = DynamicVesselNetwork({"tank": v}, walls={"tank": wall})
    s = net.initialize(
        f,
        {
            "tank": {
                "pressure_pa": 100,
                "temperature_k": 300,
                "mole_fractions": [0.5, 0.5],
                "wall_temperature_k": 350,
            }
        },
    )
    r = net.advance(s, f, 1, initial_dt_s=1)
    end = r["states"]["tank"]
    assert end["temperature_k"] == pytest.approx(300, abs=1e-7)
    assert end["wall_temperature_k"] == pytest.approx((10 * 350 + 0.2 * 280) / (10 + 0.2), abs=1e-8)
    assert r["steps"][0]["vessels"]["tank"]["wall_to_fluid_heat_j"] == 0


def test_wall_model_and_energy_mismatch_rejected_not_silently_discarded():
    f, v, wall, net, s = setup()
    with pytest.raises(ConservationError):
        net.advance({"tank": replace(s["tank"], wall_energy_j=s["tank"].wall_energy_j + 1)}, f, 1)
    with pytest.raises(PropertyModelError):
        DynamicVesselNetwork({"tank": v}).advance(s, f, 1)
    with pytest.raises(PropertyModelError):
        v.advance(s["tank"], f, 1)
    with pytest.raises(PropertyModelError):
        DynamicVesselNetwork(
            {"tank": v}, walls={"tank": replace(wall, heat_capacity_j_k=11)}
        ).advance(s, f, 1)


def test_mixed_walled_and_unwalled_nodes_preserve_all_energy():
    f, v, wall, _net, _s = setup()
    other = DynamicVessel("other", 1, 1, 1000)
    net = DynamicVesselNetwork({"tank": v, "other": other}, walls={"tank": wall})
    specs = {
        name: {"pressure_pa": 100, "temperature_k": 300, "mole_fractions": [0.5, 0.5]}
        for name in ("tank", "other")
    }
    specs["tank"]["wall_temperature_k"] = 350
    s = net.initialize(f, specs)
    r = net.advance(s, f, 1, initial_dt_s=1)
    assert r["states"]["other"]["wall_temperature_k"] is None
    assert r["states"]["other"]["temperature_k"] == pytest.approx(300, abs=1e-7)
    assert sum(total(VesselState(**q)) for q in r["states"].values()) == pytest.approx(
        sum(total(q) for q in s.values()), abs=1e-6
    )


def test_two_phase_fluid_wall_coupling_closes_total_energy():
    f, _v, _wall, _net, _s = setup()
    v = DynamicVessel("tank", 0.1, 1e5, 5e6)
    wall = VesselWall(100.0, 1.0)
    net = DynamicVesselNetwork({"tank": v}, walls={"tank": wall})
    s = net.initialize(
        f,
        {
            "tank": {
                "pressure_pa": 3e6,
                "temperature_k": 220,
                "mole_fractions": [0.5, 0.5],
                "wall_temperature_k": 230,
            }
        },
    )
    r = net.advance(s, f, 0.1, initial_dt_s=0.1)
    assert v.inspect(VesselState(**r["states"]["tank"]), f)["equilibrium"]["phase_count"] == 2
    assert total(VesselState(**r["states"]["tank"])) == pytest.approx(total(s["tank"]), abs=1e-4)
    assert r["states"]["tank"]["temperature_k"] > 220


def test_wall_heat_soak_continues_after_direct_heater_trip():
    f, _v, _wall, net, s = setup()
    trip = NetworkTrip(
        "high-p", "tank", "pressure_pa", "above", 100.1, 1e-5, stop_heating_vessels=("tank",)
    )
    r = advance_protected_network(
        net,
        s,
        f,
        1,
        [trip],
        heat={"tank": {"heat_w": 0.1, "utility_temperature_k": 400}},
        initial_dt_s=0.5,
    )
    assert len(r["events"]) == 1 and r["latched_trips"] == ["high-p"]
    assert r["states"]["tank"]["pressure_pa"] > 100.1 + 0.01
    external = sum(step["heat_j"] for step in r["steps"])
    assert external == pytest.approx(0.1 * r["events"][0]["time_s"], abs=1e-6)
    assert total(VesselState(**r["states"]["tank"])) - total(s["tank"]) == pytest.approx(
        external, abs=1e-6
    )


def test_wall_observer_and_schedule_ambient_commands_restart():
    f, _v, _wall, net, s = setup(0.2)
    observer = NetworkTrip(
        "wall-cold",
        "tank",
        "wall_temperature_k",
        "below",
        349.8,
        1e-5,
        stop_heating_vessels=("tank",),
    )
    r = net.advance(s, f, 2, initial_dt_s=0.5, stop_conditions=[observer])
    assert 0 < r["duration_s"] < 2
    assert r["states"]["tank"]["wall_temperature_k"] == pytest.approx(349.8, abs=1e-5)
    periods = [{"duration_s": 0.5}, {"duration_s": 0.5, "ambient_temperatures": {"tank": 370}}]
    full = run_network_schedule(f, net, s, periods, stepper={"initial_dt_s": 0.5})
    partial = run_network_schedule(f, net, s, periods, stepper={"initial_dt_s": 0.5}, stop_after=1)
    saved = copy.deepcopy(partial)
    resumed = run_network_schedule(
        f,
        net,
        s,
        periods,
        stepper={"initial_dt_s": 0.5},
        checkpoint=json.loads(json.dumps(partial)),
    )
    assert resumed == full and partial == saved
    assert full["records"][0]["steps"][0]["ambient_heat_j"] < 0
    assert full["records"][1]["steps"][0]["ambient_heat_j"] > 0


def test_bad_wall_initialization_and_ambient_command_rejected():
    f, v, _wall, net, _s = setup()
    with pytest.raises(ValueError):
        net.initialize(
            f, {"tank": {"pressure_pa": 100, "temperature_k": 300, "mole_fractions": [0.5, 0.5]}}
        )
    bare = DynamicVesselNetwork({"tank": v})
    s = bare.initialize(
        f, {"tank": {"pressure_pa": 100, "temperature_k": 300, "mole_fractions": [0.5, 0.5]}}
    )
    with pytest.raises(ValueError):
        run_network_schedule(
            f, bare, s, [{"duration_s": 1, "ambient_temperatures": {"tank": 350}}], stop_after=0
        )
