"""Physical actuator oracle, conservative bypass and sampled recycle/restart."""

import copy
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surface import LiquidTransientNetwork, LiquidTransientState
from surface.liquid_recycle import RecyclingLiquidState, RecyclingLiquidStation
from surface.liquid_recycle_schedule import run_recycling_liquid_schedule
from surface.operation import _digest


def draining(dt=0.01, **updates):
    data = {
        "network": {
            "model": "incompressible_liquid",
            "fluid": {"density_kg_m3": 1000, "viscosity_pa_s": 0.001},
            "nodes": [{"name": "a"}, {"name": "out", "pressure_pa": 500000}],
            "pipes": [],
            "pumps": [],
            "valves": [
                {
                    "name": "v",
                    "start": "a",
                    "end": "out",
                    "diameter_m": 0.01,
                    "opening": 0,
                    "check_valve": True,
                }
            ],
        },
        "storage_volumes_m3": {"a": 1},
        "bulk_modulus_pa": 1e7,
        "reference_pressure_pa": 500000,
        "maximum_dt_s": dt,
        "temporal_error_control": False,
        "actuated_valves": {
            "v": {
                "opening_rate_per_s": 100,
                "closing_rate_per_s": 100,
                "fail_position": 1,
                "time_constant_s": 0.05,
                "max_position_change": 1,
            }
        },
    }
    data.update(updates)
    return LiquidTransientNetwork.from_dict(data)


def test_midpoint_moving_valve_matches_independent_closed_form_and_second_order():
    errors = []
    for dt in (0.02, 0.01, 0.005):
        m = draining(dt)
        s = m.initialize({"a": 600000}, actuator_positions={"v": 0.2})
        r = m.advance(s, 0.2, valve_openings={"v": 1})
        end = LiquidTransientState(**r["state"])
        valve = m.network.valves[0]
        cv = valve.discharge_coefficient * math.pi * valve.diameter_m**2 / 4
        integrated_opening = 0.2 - 0.8 * 0.05 * (-math.expm1(-0.2 / 0.05))
        root = math.sqrt(100000) - cv * math.sqrt(2 / 1000) / (2e-7) * integrated_opening
        exact = 500000 + root**2
        errors.append(abs(dict(end.pressures_pa)["a"] - exact))
        assert dict(end.actuator_positions)["v"] == pytest.approx(1 - 0.8 * math.exp(-4), abs=2e-15)
        assert all(
            abs(x["mass_residual_kg"]) < 1e-8 and abs(x["energy_residual_j"]) < 1e-4
            for x in r["steps"]
        )
        first = r["steps"][0]["valve_actuators"]["v"]
        assert first["midpoint_position"] == pytest.approx(
            1 - 0.8 * math.exp(-r["steps"][0]["dt_s"] / 2 / 0.05)
        )
    assert errors[0] / errors[1] > 3.8 and errors[1] / errors[2] > 3.8, errors
    assert errors[-1] < 0.5


@pytest.mark.parametrize(
    "mode,target,expected",
    [("stuck", 1, 0), ("fail_safe", 0, -math.expm1(-4)), ("normal", 1, -math.expm1(-4))],
)
def test_actuator_modes_and_restart_position(mode, target, expected):
    m = draining()
    s = m.initialize({"a": 600000})
    a = m.advance(s, 0.1, valve_openings={"v": target}, actuator_modes={"v": mode})
    middle = LiquidTransientState(**json.loads(json.dumps(a["state"])))
    b = m.advance(middle, 0.1, valve_openings={"v": target}, actuator_modes={"v": mode})
    assert b["state"]["actuator_positions"]["v"] == pytest.approx(expected, abs=2e-15)
    if mode == "stuck":
        assert b["state"]["pressures_pa"]["a"] == pytest.approx(600000, abs=1e-4)


def test_motion_guard_retries_without_double_advancing_actuator():
    data = draining(0.1).to_dict()
    data["actuated_valves"]["v"]["max_position_change"] = 0.02
    m = LiquidTransientNetwork.from_dict(data)
    s = m.initialize({"a": 600000})
    r = m.advance(s, 0.2, valve_openings={"v": 1})
    assert r["failed_steps"] > 0
    assert all(
        abs(
            x["valve_actuators"]["v"]["final_position"]
            - x["valve_actuators"]["v"]["initial_position"]
        )
        <= 0.0200000001
        for x in r["steps"]
    )
    assert r["state"]["actuator_positions"]["v"] == pytest.approx(-math.expm1(-4), abs=2e-15)


@pytest.mark.parametrize("modes", [{"missing": "normal"}, {"v": "teleport"}])
def test_bad_modes_rejected(modes):
    m = draining()
    with pytest.raises(ValueError):
        m.advance(m.initialize(), 0.1, actuator_modes=modes)


def station():
    d = json.loads((ROOT / "configs/surface_recycling_liquid.json").read_text())
    return RecyclingLiquidStation.from_dict(d["system"])


def test_recycle_model_and_state_roundtrip():
    m = station()
    assert (
        RecyclingLiquidStation.from_dict(json.loads(json.dumps(m.to_dict()))).model_id == m.model_id
    )
    s = m.initialize()
    assert RecyclingLiquidState.from_dict(json.loads(json.dumps(s.to_dict()))) == s


def test_sample_clock_continues_through_partial_reports_and_tracks_actual_position():
    m = station()
    s = m.initialize()
    cmd = {
        "mode": "torque",
        "motor_torques_nm": {"p1": 2},
        "recycle_mode": "manual",
        "recycle_openings": {"recycle": 0.7},
    }
    a = m.advance(s, 0.013, **cmd)
    b = m.advance(RecyclingLiquidState.from_dict(a["state"]), 0.087, **cmd)
    assert [x["time_s"] for x in a["recycle_samples"] + b["recycle_samples"]] == pytest.approx(
        [0, 0.05]
    )
    actual = b["recycle_samples"][0]["actual_position"]
    assert 0 < actual < 0.7
    assert b["state"]["protected"]["controlled"]["physical"]["actuator_positions"]["recycle"] > 0.0


def test_recycle_restart_exact_and_tampering_rejected():
    m = station()
    s = m.initialize()
    periods = [
        {"duration_s": 0.013, "mode": "torque", "motor_torques_nm": {"p1": 2}},
        {
            "duration_s": 0.057,
            "mode": "torque",
            "motor_torques_nm": {"p1": 2},
            "actuator_modes": {"recycle": "fail_safe"},
        },
    ]
    full = run_recycling_liquid_schedule(m, s, periods)
    partial = run_recycling_liquid_schedule(m, s, periods, stop_after=1)
    resumed = run_recycling_liquid_schedule(m, s, periods, checkpoint=partial)
    assert full == resumed
    altered = copy.deepcopy(partial)
    altered["records"][0]["state"]["targets"]["recycle"] = 0.123
    altered["checksum"] = _digest({k: v for k, v in altered.items() if k != "checksum"})
    with pytest.raises((ValueError, RuntimeError)):
        run_recycling_liquid_schedule(m, s, periods, checkpoint=altered)


@pytest.mark.parametrize(
    "update",
    [
        {"valve_openings": {"recycle": 0.2}},
        {"recycle_mode": "manual"},
        {"recycle_openings": {"missing": 0}},
        {"max_recycle_samples": False},
    ],
)
def test_conflicting_or_invalid_commands(update):
    m = station()
    with pytest.raises(ValueError):
        m.advance(m.initialize(), 0.01, **update)


def test_nearly_closed_actuator_does_not_make_hydraulic_equation_singular():
    m = draining(0.02)
    s = m.initialize({"a": 600000}, actuator_positions={"v": 1e-30})
    r = m.advance(s, 0.1, valve_openings={"v": 0})
    assert len(r["steps"]) <= 10
    assert r["failed_steps"] == 0
    assert r["state"]["actuator_positions"]["v"] > 0


def test_closed_export_recycles_real_mass_without_artificial_production():
    m = station()
    q, dp = 0.0013, 250000
    m.physics.network.pumps[0]
    rho = m.physics.network.fluid.density_kg_m3
    # Independent quadratic map and orifice steady solution.
    gravity = 9.80665
    omega = 100 * math.sqrt((dp / (rho * gravity) + 1e6 * q * q) / 20)
    ratio = omega / 100
    opening = 2 * q / (0.7 * math.pi * 0.02**2 / 4 * math.sqrt(2 * dp / rho))
    torque = (dp * q / 0.8 + 10 * ratio**3) / omega + 0.005 * omega
    s = m.initialize(
        {
            "controller_initial": {
                "physical_initial": {
                    "pressures_pa": {"h": 650000, "t": 650000},
                    "speeds_rad_s": {"p1": omega, "p2": omega},
                    "actuator_positions": {"recycle": opening},
                },
                "running_pumps": ["p1", "p2"],
                "applied_torques_nm": {"p1": torque, "p2": torque},
            }
        }
    )
    r = m.advance(
        s,
        0.2,
        mode="torque",
        motor_torques_nm={"p1": torque, "p2": torque},
        valve_openings={"export": 0},
        recycle_mode="manual",
        recycle_openings={"recycle": opening},
    )
    assert not r["latched_trips"]
    assert r["recycled_volume_m3"]["recycle"] == pytest.approx(2 * q * 0.2, rel=1e-6)
    for step in r["steps"]:
        flows = step["midpoint_branch_volume_rates_m3_s"]
        assert flows["export"] == pytest.approx(0, abs=1e-10)
        assert flows["recycle"] == pytest.approx(flows["p1"] + flows["p2"], abs=1e-9)
        assert abs(sum(step["boundary_mass_transfers_in_kg"].values())) < 1e-8
        assert abs(step["energy_residual_j"]) < 1e-4
    assert r["state"]["protected"]["controlled"]["physical"]["pressures_pa"]["h"] == pytest.approx(
        650000, abs=0.1
    )


def test_stuck_closed_recycle_cannot_fake_minimum_flow_and_existing_guard_trips():
    m = station()
    omega = 100 * math.sqrt((250000 / (1000 * 9.80665)) / 20)
    torque = 10 * (omega / 100) ** 3 / omega + 0.005 * omega
    s = m.initialize(
        {
            "controller_initial": {
                "physical_initial": {
                    "pressures_pa": {"h": 650000, "t": 650000},
                    "speeds_rad_s": {"p2": omega},
                },
                "running_pumps": ["p2"],
                "applied_torques_nm": {"p2": torque},
            }
        }
    )
    r = m.advance(
        s,
        1.6,
        mode="torque",
        motor_torques_nm={"p2": torque},
        valve_openings={"export": 0},
        actuator_modes={"recycle": "stuck"},
    )
    assert "minflow-p2" in r["latched_trips"]
    assert r["recycled_volume_m3"]["recycle"] == pytest.approx(0, abs=1e-10)
    assert r["state"]["protected"]["controlled"]["physical"]["actuator_positions"]["recycle"] == 0
    assert max(x["target_position"] for x in r["recycle_samples"]) > 0.1
    assert "p2" in r["state"]["protected"]["controlled"]["physical"]["unavailable"]


@pytest.mark.parametrize(
    "mutation", ["reverse", "unactuated", "duplicate", "wrong_planning", "wrong_pid"]
)
def test_recycle_rejects_nonphysical_or_ambiguous_topology(mutation):
    data = station().to_dict()
    if mutation == "reverse":
        v = data["protected"]["controller"]["physics"]["network"]["valves"][-1]
        v["start"], v["end"] = v["end"], v["start"]
    elif mutation == "unactuated":
        data["protected"]["controller"]["physics"]["actuated_valves"] = {}
    elif mutation == "duplicate":
        data["loops"].append(copy.deepcopy(data["loops"][0]))
    elif mutation == "wrong_planning":
        data["protected"]["controller"]["planning_valve_openings"] = {}
    else:
        data["loops"][0]["controller"]["direction"] = -1
    with pytest.raises(ValueError):
        RecyclingLiquidStation.from_dict(data)


def test_adaptive_half_steps_use_correct_actuator_trajectory():
    m = draining(0.02, temporal_error_control=True)
    r = m.advance(
        m.initialize({"a": 600000}, actuator_positions={"v": 0.2}), 0.2, valve_openings={"v": 1}
    )
    previous = 0.2
    for record in r["steps"]:
        actuator = record["valve_actuators"]["v"]
        assert actuator["initial_position"] == pytest.approx(previous, abs=1e-15)
        assert actuator["midpoint_position"] == pytest.approx(
            1 - (1 - previous) * math.exp(-record["dt_s"] / 2 / 0.05), abs=1e-15
        )
        previous = actuator["final_position"]
    assert previous == pytest.approx(1 - 0.8 * math.exp(-4), abs=1e-14)


def test_recycle_budget_covers_whole_reporting_interval():
    m = station()
    with pytest.raises(RuntimeError, match="sample budget"):
        m.advance(m.initialize(), 0.12, mode="off", max_recycle_samples=1)
