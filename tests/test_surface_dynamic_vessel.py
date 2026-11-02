"""Rigid-vessel balances, independent ideal limits and phase-selective dynamics."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    ConservationError,
    DynamicVessel,
    HydraulicConvergenceError,
    PengRobinsonFluid,
    PengRobinsonMixture,
    PropertyModelError,
    ThermalStream,
    VesselState,
    run_vessel_schedule,
)
from surface.peng_robinson import R


def fluid():
    spec = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_thermal_mixture.json").read_text()
    )["fluid"]
    return PengRobinsonMixture(
        tuple(PengRobinsonFluid(**c) for c in spec["components"]), spec["binary_interactions"]
    )


def dilute():
    f = fluid()
    v = DynamicVessel("dilute", 1.0, 1.0, 1e7)
    return f, v, v.initialize(f, 100.0, 300.0, [0.5, 0.5])


def test_initialization_is_consistent_and_tampered_energy_rejected():
    f, v, s = dilute()
    q = v.inspect(s, f)
    assert abs(q["volume_residual_m3"]) < 1e-10
    assert q["liquid_volume_fraction"] is None
    assert VesselState(**json.loads(json.dumps(s.to_dict()))) == s
    with pytest.raises(ConservationError):
        v.inspect(replace(s, internal_energy_j=s.internal_energy_j + 1), f)
    with pytest.raises(PropertyModelError):
        v.inspect(replace(s, model_id="wrong"), f)


def test_isolated_tank_and_exact_interval():
    f, v, s = dilute()
    r = v.advance(s, f, 0.7, initial_dt_s=0.3)
    assert r["state"]["component_moles"] == pytest.approx(s.component_moles, abs=1e-11)
    assert r["state"]["internal_energy_j"] == pytest.approx(s.internal_energy_j, abs=1e-7)
    assert r["state"]["time_s"] == 0.7
    assert sum(x["dt_s"] for x in r["steps"]) == pytest.approx(0.7)


def test_rigid_tank_heating_uses_cv_not_cp():
    f, v, s = dilute()
    cp = sum(c.ideal_cp_j_mol_k for c in f.components) / 2
    heat = 0.1
    duration = 2
    r = v.advance(s, f, duration, heat_w=heat, utility_temperature_k=400, initial_dt_s=1)
    actual = r["state"]["temperature_k"] - s.temperature_k
    expected = heat * duration / (sum(s.component_moles) * (cp - R))
    assert actual == pytest.approx(expected, rel=3e-5)
    assert r["state"]["internal_energy_j"] - s.internal_energy_j == pytest.approx(
        heat * duration, abs=1e-7
    )


def test_filling_includes_inlet_flow_work():
    f, v, s = dilute()
    feed = ThermalStream.from_pt(f, 1000, 300, 0.0001, [0.5, 0.5])
    duration = 2
    r = v.advance(s, f, duration, feeds=[feed], initial_dt_s=1)
    n0 = sum(s.component_moles)
    added = duration * feed.molar_rate_mol_s
    cp = sum(c.ideal_cp_j_mol_k for c in f.components) / 2
    expected = 300 * (n0 * (cp - R) + added * cp) / ((n0 + added) * (cp - R))
    assert r["state"]["temperature_k"] == pytest.approx(expected, rel=2e-6)
    assert sum(r["state"]["component_moles"]) == pytest.approx(n0 + added, abs=1e-10)
    assert r["state"]["internal_energy_j"] - s.internal_energy_j == pytest.approx(
        duration * feed.enthalpy_rate_w, abs=1e-7
    )


def test_blowdown_time_refinement_approaches_independent_solution():
    f, v, s = dilute()
    duration = 2
    q = 0.001
    cp = sum(c.ideal_cp_j_mol_k for c in f.components) / 2
    target = 300 * ((sum(s.component_moles) - duration * q) / sum(s.component_moles)) ** (
        R / (cp - R)
    )
    errors = []
    for dt in (1.0, 0.5, 0.25):
        r = v.advance(s, f, duration, withdrawals={"bulk": q}, initial_dt_s=dt)
        errors.append(abs(r["state"]["temperature_k"] - target))
        assert sum(r["state"]["component_moles"]) == pytest.approx(
            sum(s.component_moles) - duration * q, abs=1e-9
        )
    assert errors[1] < 0.6 * errors[0] and errors[2] < 0.6 * errors[1]


def test_two_phase_withdrawal_changes_inventory_composition_and_energy():
    f = fluid()
    v = DynamicVessel("separator", 0.1, 1e5, 1e7)
    s = v.initialize(f, 3e6, 220, [0.5, 0.5])
    r = v.advance(s, f, 0.1, withdrawals={"phase_1": 0.1}, initial_dt_s=0.1)
    final = r["state"]
    out = np.sum([step["outlet_component_moles"] for step in r["steps"]], axis=0)
    assert out[0] / sum(out) > 0.7
    assert np.array(final["component_moles"]) + out == pytest.approx(s.component_moles, abs=1e-7)
    assert final["internal_energy_j"] - s.internal_energy_j == pytest.approx(
        -sum(x["outlet_enthalpy_j"] for x in r["steps"]), abs=1e-5
    )
    assert r["final"]["equilibrium"]["phase_count"] == 2
    assert 0 < r["final"]["liquid_volume_fraction"] < 1


def test_level_pressure_and_empty_phase_guards():
    f, v, s = dilute()
    with pytest.raises(PropertyModelError, match="absent initially"):
        v.advance(s, f, 1, withdrawals={"phase_1": 1})
    with pytest.raises(PropertyModelError, match="resolved liquid/vapor"):
        replace(v, max_liquid_volume_fraction=0.9).inspect(s, f)
    two = DynamicVessel("level", 0.1, 1e5, 1e7)
    state = two.initialize(f, 3e6, 220, [0.5, 0.5])
    with pytest.raises(PropertyModelError, match="volume fraction"):
        replace(two, max_liquid_volume_fraction=0.1).inspect(state, f)
    with pytest.raises(ValueError):
        replace(two, min_liquid_volume_fraction=0.9, max_liquid_volume_fraction=0.1)


def test_failed_interval_does_not_mutate_input_and_pressure_guard_cuts_step():
    f, v, s = dilute()
    before = s.to_dict()
    narrow = replace(v, max_pressure_pa=105.0)
    with pytest.raises(HydraulicConvergenceError):
        narrow.advance(
            s, f, 1, heat_w=100, utility_temperature_k=450, initial_dt_s=0.2, minimum_dt_s=0.01
        )
    assert s.to_dict() == before
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        v.advance(s, f, 1, initial_dt_s=0.1, max_substeps=1)
    assert s.to_dict() == before


def test_adaptive_inventory_change_guard_covers_whole_interval():
    f, v, s = dilute()
    v = replace(v, max_fractional_inventory_change=0.01)
    r = v.advance(s, f, 1, withdrawals={"bulk": 0.001}, initial_dt_s=1)
    assert r["failed_steps"] > 0 and len(r["steps"]) > 1
    assert r["state"]["time_s"] == 1
    assert sum(r["state"]["component_moles"]) == pytest.approx(
        sum(s.component_moles) - 0.001, abs=1e-9
    )


def test_dynamic_schedule_json_resume_and_global_inventory_balance():
    f, v, s = dilute()
    periods = [
        {"duration_s": 1, "heat_w": 0.1, "utility_temperature_k": 400},
        {"duration_s": 1, "withdrawals": {"bulk": 0.0001}},
        {"duration_s": 0.5},
    ]
    full = run_vessel_schedule(f, v, s, periods, stepper={"initial_dt_s": 0.5})
    partial = run_vessel_schedule(f, v, s, periods, stepper={"initial_dt_s": 0.5}, stop_after=1)
    saved = copy.deepcopy(partial)
    resumed = run_vessel_schedule(
        f, v, s, periods, stepper={"initial_dt_s": 0.5}, checkpoint=json.loads(json.dumps(partial))
    )
    assert full == resumed and partial == saved
    assert full["state"]["time_s"] == 2.5
    assert np.array(full["state"]["component_moles"]) + full["totals"][
        "outlet_component_moles"
    ] == pytest.approx(s.component_moles, abs=1e-9)
    assert full["state"]["internal_energy_j"] - s.internal_energy_j == pytest.approx(
        full["totals"]["heat_j"] - full["totals"]["outlet_enthalpy_j"], abs=1e-6
    )
    corrupt = copy.deepcopy(partial)
    corrupt["totals"]["heat_j"] += 1
    with pytest.raises(ValueError, match="checksum"):
        run_vessel_schedule(f, v, s, periods, stepper={"initial_dt_s": 0.5}, checkpoint=corrupt)


def test_invalid_passive_feed_and_heat_direction_are_not_silent():
    f, v, s = dilute()
    feed = ThermalStream.from_pt(f, 50, 300, 0.001, [0.5, 0.5])
    with pytest.raises(PropertyModelError, match="supply pressure"):
        v.advance(s, f, 1, feeds=[feed])
    with pytest.raises(HydraulicConvergenceError, match="reservoir"):
        v.advance(s, f, 0.01, heat_w=0.1, utility_temperature_k=290, minimum_dt_s=0.001)


def test_schedule_rejects_invalid_stepper_before_any_period():
    f, v, s = dilute()
    with pytest.raises(ValueError):
        run_vessel_schedule(f, v, s, [{"duration_s": 1}], stepper={"initial_dt_s": 0}, stop_after=0)
