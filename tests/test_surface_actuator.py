"""Actuator trajectories, conservative transient flow, trips and checked restart."""

import copy
import json
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from test_surface_vessel_network import setup

from surface import (
    ConnectionActuator,
    DynamicVesselNetwork,
    HydraulicConvergenceError,
    NetworkTrip,
    advance_protected_network,
    run_network_schedule,
)
from surface.operation import _digest


def model(**kwargs):
    return ConnectionActuator(
        **{
            "opening_rate_per_s": 0.4,
            "closing_rate_per_s": 0.25,
            "fail_position": 0.0,
            "max_position_change": 1.0,
            **kwargs,
        }
    )


@pytest.mark.parametrize("x,target,dt", [(0, 1, 1), (0.9, 0, 4), (0.1, 0.8, 2), (0.8, 0.8, 3)])
def test_exact_law_matches_independent_ode_and_subdivision(x, target, dt):
    a = model(time_constant_s=0.7)

    def ode(t, y):
        return [
            np.clip(
                (target - y[0]) / a.time_constant_s, -a.closing_rate_per_s, a.opening_rate_per_s
            )
        ]

    reference = solve_ivp(ode, [0, dt], [x], rtol=1e-11, atol=1e-12, max_step=0.01).y[0, -1]
    result = a.advance(x, target, dt)["final_position"]
    first = a.advance(x, target, dt * 0.3)["final_position"]
    assert result == pytest.approx(reference, abs=2e-9)
    assert a.advance(first, target, dt * 0.7)["final_position"] == pytest.approx(result, abs=1e-14)
    assert min(x, target) <= result <= max(x, target)


def test_slew_limits_modes_and_exact_stops():
    a = model()
    assert a.advance(0, 1, 1)["final_position"] == 0.4
    assert a.advance(1, 0, 1)["final_position"] == 0.75
    assert a.advance(0, 1, 100)["final_position"] == 1
    assert a.advance(1, 1, 4, "fail_safe")["final_position"] == 0
    assert a.advance(0.7, 0, 100, "stuck")["final_position"] == 0.7
    fail_open = replace(a, fail_position=1)
    r = fail_open.advance(0, 0, 1, "fail_safe")
    assert r["command"] == 0 and r["effective_target"] == 1 and r["final_position"] == 0.4
    exponential = model(time_constant_s=2, opening_rate_per_s=10)
    assert exponential.advance(0, 1, 2)["final_position"] == pytest.approx(1 - math.exp(-1))


@pytest.mark.parametrize(
    "params",
    [
        {"opening_rate_per_s": 0},
        {"closing_rate_per_s": -1},
        {"time_constant_s": -1},
        {"max_position_change": 1.1},
        {"fail_position": 2},
        {"time_constant_s": float("nan")},
    ],
)
def test_bad_models_rejected(params):
    with pytest.raises(ValueError):
        model(**params)


def test_initial_state_and_commands_are_explicit_and_validated():
    f, v, s, c = setup()
    net = DynamicVesselNetwork(v, (c,), actuators={"line": model()})
    for positions in (None, {}, {"other": 0.5}, {"line": -0.1}):
        with pytest.raises(ValueError):
            net.advance(s, f, 1, actuator_positions=positions)
    with pytest.raises(ValueError):
        net.advance(s, f, 1, actuator_positions={"line": 0}, actuator_modes={"line": "magic"})
    with pytest.raises(ValueError):
        DynamicVesselNetwork(v, (c,), actuators={"typo": model()})


def test_new_layer_actual_position_controls_flux_and_conserves_closed_network():
    f, v, s, c = setup()
    a = model()
    net = DynamicVesselNetwork(v, (c,), actuators={"line": a})
    r = net.advance(s, f, 1, actuator_positions={"line": 0}, initial_dt_s=1)
    assert r["actuator_positions"] == {"line": 0.4}
    record = r["steps"][0]
    p = r["states"]
    expected = 0.4 * c.molar_conductance_mol_s_pa * (p["a"]["pressure_pa"] - p["b"]["pressure_pa"])
    assert record["connections"][0]["molar_rate_mol_s"] == pytest.approx(expected, abs=1e-12)
    for i in range(2):
        assert sum(x["component_moles"][i] for x in p.values()) == pytest.approx(
            sum(x.component_moles[i] for x in s.values()), abs=1e-9
        )
    assert abs(record["energy_residual_j"]) < 1e-7


def test_failed_trial_does_not_advance_actuator_and_motion_guard_cuts_step(monkeypatch):
    f, v, s, c = setup()
    a = model(max_position_change=0.1)
    net = DynamicVesselNetwork(v, (c,), actuators={"line": a})
    original = DynamicVesselNetwork._step
    failed = [False]

    def once(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not failed[0]:
            failed[0] = True
            raise HydraulicConvergenceError("simulated rejected physical trial")
        return result

    monkeypatch.setattr(DynamicVesselNetwork, "_step", once)
    positions = {"line": 0.0}
    r = net.advance(s, f, 1, actuator_positions=positions, initial_dt_s=1)
    assert positions == {"line": 0.0} and s["a"].time_s == 0
    assert r["failed_steps"] >= 2
    assert r["actuator_positions"]["line"] == pytest.approx(0.4)
    assert all(
        abs(x["actuators"]["line"]["final_position"] - x["actuators"]["line"]["initial_position"])
        <= 0.1 + 1e-12
        for x in r["steps"]
    )


def test_trip_requests_closure_but_flow_continues_during_physical_stroke():
    f, v, s, c = setup()
    net = DynamicVesselNetwork(v, (c,), actuators={"line": model()})
    trip = NetworkTrip(
        "high-p", "b", "pressure_pa", "above", 100.1, 1e-5, close_connections=("line",)
    )
    r = advance_protected_network(
        net, s, f, 5, [trip], actuator_positions={"line": 1}, initial_dt_s=0.5
    )
    assert len(r["events"]) == 1
    after = [x for x in r["steps"] if x["actuators"]["line"]["command"] == 0]
    assert after[0]["actuators"]["line"]["final_position"] > 0
    assert after[0]["connections"][0]["molar_rate_mol_s"] > 0
    assert r["actuator_positions"]["line"] == 0
    assert r["states"]["b"]["pressure_pa"] > trip.threshold + 0.1
    assert sum(x["dt_s"] for x in r["steps"]) == pytest.approx(5)
    assert max(abs(x["energy_residual_j"]) for x in r["steps"]) < 1e-7


def test_schedule_failure_modes_reset_and_restart_are_exact():
    f, v, s, c = setup()
    net = DynamicVesselNetwork(v, (c,), actuators={"line": model(fail_position=1)})
    trip = NetworkTrip(
        "high-p", "b", "pressure_pa", "above", 100.1, 1e-5, close_connections=("line",)
    )
    periods = [
        {"duration_s": 0.5},
        {"duration_s": 0.5, "actuator_modes": {"line": "stuck"}},
        {"duration_s": 0.5, "actuator_modes": {"line": "fail_safe"}},
        {"duration_s": 0.5, "reset_trips": ["high-p"]},
    ]
    kw = {
        "initial_actuator_positions": {"line": 0},
        "protections": [trip],
        "stepper": {"initial_dt_s": 0.25},
    }
    full = run_network_schedule(f, net, s, periods, **kw)
    partial = run_network_schedule(f, net, s, periods, stop_after=1, **kw)
    saved = copy.deepcopy(partial)
    resumed = run_network_schedule(
        f, net, s, periods, checkpoint=json.loads(json.dumps(partial)), **kw
    )
    assert resumed == full and partial == saved
    stuck = full["records"][1]["steps"]
    assert all(
        x["actuators"]["line"]["final_position"] == x["actuators"]["line"]["initial_position"]
        for x in stuck
    )
    failed = full["records"][2]["steps"]
    assert all(x["actuators"]["line"]["effective_target"] == 1 for x in failed)
    assert full["records"][3]["events"][0]["initially_triggered"]
    damaged = copy.deepcopy(partial)
    damaged["actuator_positions"]["line"] += 0.01
    damaged.pop("checksum")
    damaged["checksum"] = _digest(damaged)
    with pytest.raises(ValueError, match="positions"):
        run_network_schedule(f, net, s, periods, checkpoint=damaged, **kw)
    damaged = copy.deepcopy(partial)
    damaged["records"][0]["actuator_positions"]["line"] += 0.01
    damaged.pop("checksum")
    damaged["checksum"] = _digest(damaged)
    with pytest.raises(ValueError, match="period actuator"):
        run_network_schedule(f, net, s, periods, checkpoint=damaged, **kw)
    damaged = copy.deepcopy(partial)
    damaged["records"][0]["steps"][0]["actuators"]["line"]["command"] = 0.9
    damaged.pop("checksum")
    damaged["checksum"] = _digest(damaged)
    with pytest.raises(ValueError, match="trajectory"):
        run_network_schedule(f, net, s, periods, checkpoint=damaged, **kw)


def test_hydraulic_transient_converges_when_substeps_are_refined():
    f, v, s, c = setup()
    net = DynamicVesselNetwork(
        v, (c,), actuators={"line": model(opening_rate_per_s=2, time_constant_s=0.3)}
    )
    outputs = []
    for dt in (0.25, 0.125, 0.0625):
        r = net.advance(s, f, 0.5, actuator_positions={"line": 0}, initial_dt_s=dt, maximum_dt_s=dt)
        outputs.append(r["states"]["b"]["pressure_pa"])
    assert abs(outputs[2] - outputs[1]) < 0.7 * abs(outputs[1] - outputs[0])


@pytest.mark.parametrize("kind", ["pr", "hem"])
def test_nozzle_connections_use_actual_actuator_opening(kind):
    from test_surface_hem_connection import case

    from surface import PRNozzleConnection

    f, v, s, connection = case(300 if kind == "pr" else 220)
    if kind == "pr":
        connection = PRNozzleConnection("pr", "a", "b", 1e-8, 3e6, 0.1, minimum_pressure_ratio=0.3)
    a = model(opening_rate_per_s=1)
    net = DynamicVesselNetwork(v, (connection,), actuators={connection.name: a})
    result = net.advance(s, f, 0.01, initial_dt_s=0.01, actuator_positions={connection.name: 0})
    step = result["steps"][0]
    end = result["states"]
    from test_surface_hem_connection import values

    from surface import VesselState

    states = {name: VesselState(**x) for name, x in end.items()}
    actual = connection.flow(
        {name: x.pressure_pa for name, x in states.items()},
        factor=0.01,
        values=values(f, v, states),
        fluid=f,
    )
    assert step["connections"][0]["molar_rate_mol_s"] == pytest.approx(actual, rel=1e-10)
    assert abs(step["energy_residual_j"]) < 1e-5
