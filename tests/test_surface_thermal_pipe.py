"""Conservative total energy, independent caloric limits and thermal restart."""

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
    PropertyModelError,
    ThermalHomogeneousPipe,
    ThermalPipeInventory,
    run_pipe_schedule,
)
from surface.operation import _digest


def pipe(**kw):
    return ThermalHomogeneousPipe(
        **{
            "length_m": 10,
            "diameter_m": 0.1,
            "cells": 4,
            "liquid_density_kg_m3": 850,
            "gas_constant_j_kg_k": 500,
            "temperature_k": 300,
            "distribution_parameter": 1,
            "drift_velocity_m_s": 0,
            "darcy_factor": 0.02,
            "liquid_cv_j_kg_k": 2000,
            "gas_cv_j_kg_k": 1500,
            **kw,
        }
    )


def boundary(t=300, p=1e6, v=0.1):
    return {
        "pressure_pa": p,
        "gas_void_fraction": 0.2,
        "mixture_velocity_m_s": v,
        "temperature_k": t,
    }


def test_eos_calorics_energy_flux_and_characteristic_speed():
    p = pipe()
    u = p.primitive(1e6, 0.2, 0.3, 320)
    rho = u[0] + u[1]
    cv = u[0] * 2000 + u[1] * 1500
    assert u[3] == pytest.approx(cv * 320 + 0.5 * rho * 0.3**2)
    assert p._flux(u)[3] == pytest.approx((u[3] + 1e6) * 0.3)
    gamma = 1 + u[1] * 500 / cv
    assert p._wave_speed(u) == pytest.approx(0.3 + np.sqrt(gamma * 1e6 / (rho * 0.2)))
    # Independent conservative flux Jacobian: two contacts and two acoustic waves.
    jac = np.empty((4, 4))
    for k in range(4):
        h = max(1e-5, abs(u[k]) * 1e-6)
        plus, minus = u.copy(), u.copy()
        plus[k] += h
        minus[k] -= h
        jac[:, k] = (p._flux(plus) - p._flux(minus)) / (2 * h)
    eigen = np.sort(np.linalg.eigvals(jac).real)
    sound = p._wave_speed(u) - 0.3
    assert eigen == pytest.approx([0.3 - sound, 0.3, 0.3, 0.3 + sound], abs=1e-5)


@pytest.mark.parametrize("ambient", [250, 350])
def test_closed_uniform_heat_exchange_matches_two_independent_energy_formulas(ambient):
    p = pipe(heat_transfer_w_m_k=100, ambient_temperature_k=ambient)
    initial = p.initialize(1e6, 0.2)
    dt = 0.1
    result = p.advance(initial, dt, initial_dt_s=dt)
    end = ThermalPipeInventory(**result["state"])
    cv = sum(initial.liquid_mass_kg) * 2000 + sum(initial.gas_mass_kg) * 1500
    ua = p.heat_transfer_w_m_k * p.length_m
    expected = (cv * 300 + dt * ua * ambient) / (cv + dt * ua)
    assert result["properties"]["temperature_k"] == pytest.approx([expected] * 4, abs=1e-9)
    assert result["properties"]["pressure_pa"] == pytest.approx(
        [1e6 * expected / 300] * 4, rel=1e-10
    )
    heat = sum(s["heat_j"] for s in result["steps"])
    assert heat == pytest.approx(cv * (expected - 300), abs=1e-5)
    assert sum(end.total_energy_j) - sum(initial.total_energy_j) == pytest.approx(heat, abs=1e-5)
    entropy_change = sum(result["properties"]["entropy_j_k"]) - sum(
        p.inspect(initial)["entropy_j_k"]
    )
    assert entropy_change - heat / ambient >= -1e-7


def test_friction_heats_fluid_without_deleting_total_energy():
    p = pipe(darcy_factor=0.1)
    initial = p.initialize(1e6, 0.2, 10)
    before = p.inspect(initial)
    result = p.advance(initial, 0.01, initial_dt_s=0.001)
    end = ThermalPipeInventory(**result["state"])
    after = result["properties"]
    assert sum(end.total_energy_j) == pytest.approx(sum(initial.total_energy_j), abs=2e-5)
    assert sum(after["kinetic_energy_j"]) < sum(before["kinetic_energy_j"])
    assert sum(after["internal_energy_j"]) > sum(before["internal_energy_j"])
    assert sum(after["entropy_j_k"]) >= sum(before["entropy_j_k"]) - 1e-7
    assert all(s["heat_j"] == 0 for s in result["steps"])


def test_uniform_open_transport_preserves_state_and_boundary_pressure_work():
    p = pipe(darcy_factor=0)
    initial = p.initialize(1e6, 0.2, 0.1)
    result = p.advance(initial, 0.02, left=boundary(), right=boundary())
    assert result["properties"]["temperature_k"] == pytest.approx([300] * 4, abs=1e-9)
    assert result["properties"]["mixture_velocity_m_s"] == pytest.approx([0.1] * 4, abs=1e-9)
    u = p.primitive(1e6, 0.2, 0.1)
    expected = (u[3] + 1e6) * 0.1 * p.area_m2 * 0.01
    assert result["steps"][0]["left_boundary_transfer"][3] == pytest.approx(expected, rel=1e-10)


def test_open_hot_feed_mass_momentum_and_energy_balance():
    p = pipe(heat_transfer_w_m_k=10, ambient_temperature_k=340)
    initial = p.initialize(1e6, 0.2, 0.1)
    result = p.advance(initial, 0.02, left=boundary(330, 1.001e6), right=boundary())
    end = ThermalPipeInventory(**result["state"])
    flux = sum(
        np.array(s["left_boundary_transfer"]) - s["right_boundary_transfer"]
        for s in result["steps"]
    )
    before = np.array(
        [
            sum(initial.liquid_mass_kg),
            sum(initial.gas_mass_kg),
            sum(initial.momentum_kg_m_s),
            sum(initial.total_energy_j),
        ]
    )
    after = np.array(
        [
            sum(end.liquid_mass_kg),
            sum(end.gas_mass_kg),
            sum(end.momentum_kg_m_s),
            sum(end.total_energy_j),
        ]
    )
    source = [
        0,
        0,
        sum(s["body_impulse_kg_m_s"] for s in result["steps"]),
        sum(s["heat_j"] for s in result["steps"]),
    ]
    assert after - before == pytest.approx(flux + source, abs=2e-5)
    assert max(result["properties"]["temperature_k"]) > 300


def test_thermal_shutdown_and_reversal_restart_and_energy_tamper():
    p = pipe()
    initial = p.initialize(1e6, 0.2)
    periods = [
        {"duration_s": 0.01, "left": boundary(310, 1.001e6, 0), "right": boundary(v=0)},
        {"duration_s": 0.01},
        {"duration_s": 0.01, "left": boundary(v=0), "right": boundary(310, 1.001e6, 0)},
    ]
    full = run_pipe_schedule(p, initial, periods)
    partial = run_pipe_schedule(p, initial, periods, stop_after=1)
    saved = copy.deepcopy(partial)
    resumed = run_pipe_schedule(p, initial, periods, checkpoint=json.loads(json.dumps(partial)))
    assert full == resumed and partial == saved
    bad = copy.deepcopy(partial)
    bad["state"]["total_energy_j"][0] *= 1.001
    bad.pop("checksum")
    bad["checksum"] = _digest(bad)
    with pytest.raises(ConservationError):
        run_pipe_schedule(p, initial, periods, checkpoint=bad)


def test_domain_and_wrong_inventory_are_not_silently_accepted():
    p = pipe()
    initial = p.initialize(1e6, 0.2)
    with pytest.raises(PropertyModelError, match="homogeneous"):
        pipe(distribution_parameter=1.1)
    with pytest.raises(PropertyModelError, match="horizontal"):
        pipe(elevation_change_m=1)
    with pytest.raises(ValueError, match="temperature"):
        p.advance(
            initial,
            0.01,
            left={"pressure_pa": 1e6, "gas_void_fraction": 0.2, "mixture_velocity_m_s": 0},
        )
    bad = replace(initial, total_energy_j=tuple(v * 0.01 for v in initial.total_energy_j))
    with pytest.raises(PropertyModelError):
        p.inspect(bad)
    with pytest.raises(ValueError):
        pipe(gas_cv_j_kg_k=0)


def test_heating_step_accuracy_guard_subdivides_without_losing_time_or_heat():
    p = pipe(heat_transfer_w_m_k=1e6, ambient_temperature_k=350, max_temperature_change_k=1)
    initial = p.initialize(1e6, 0.2)
    result = p.advance(initial, 0.01, initial_dt_s=0.01)
    assert result["failed_steps"] > 0
    assert result["state"]["time_s"] == 0.01 and initial.time_s == 0
    heat = sum(s["heat_j"] for s in result["steps"])
    assert sum(result["state"]["total_energy_j"]) - sum(initial.total_energy_j) == pytest.approx(
        heat, abs=2e-5
    )
