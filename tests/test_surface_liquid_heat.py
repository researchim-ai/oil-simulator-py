"""Independent caloric transport, finite reservoirs and full-energy ledgers."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import LiquidTransientNetwork
from surface.liquid_heat import LiquidHeatNetwork, LiquidHeatState


def passive_physics():
    return LiquidTransientNetwork.from_dict(
        {
            "network": {
                "model": "incompressible_liquid",
                "fluid": {"density_kg_m3": 1000, "viscosity_pa_s": 0.001},
                "nodes": [{"name": "a"}, {"name": "out", "pressure_pa": 500000}],
                "pipes": [],
                "pumps": [],
                "valves": [
                    {"name": "v", "start": "a", "end": "out", "diameter_m": 0.01, "opening": 0}
                ],
            },
            "storage_volumes_m3": {"a": 0.001},
            "bulk_modulus_pa": 1e8,
            "reference_pressure_pa": 500000,
        }
    )


def caloric(**updates):
    data = {
        "physics": passive_physics(),
        "specific_heat_j_kg_k": 1000,
        "boundary_temperatures_k": {"out": 300},
        "reference_temperature_k": 300,
        "minimum_dt_s": 1e-12,
        "maximum_temperature_change_k": 100,
    }
    data.update(updates)
    return LiquidHeatNetwork(**data)


def record(physics, time=0.0, dt=1.0):
    return {
        "time_s": time,
        "dt_s": dt,
        "final_pressures_pa": {n: physics.reference_pressure_pa for n in physics.free},
        "midpoint_branch_volume_rates_m3_s": {b.name: 0.0 for b in physics._branches},
        "boundary_mass_transfers_in_kg": {
            n.name: 0.0 for n in physics.network.nodes if n.pressure_pa is not None
        },
        "dissipation_heat_j": {
            "friction": {b.name: 0.0 for b in physics.network.pipes + physics.network.valves},
            **{k: {n: 0.0 for n in physics.names} for k in ("pump", "shaft_drag", "motor")},
        },
        **{
            k: 0.0
            for k in (
                "electrical_energy_j",
                "boundary_work_in_j",
                "shaft_kinetic_energy_change_j",
                "pipe_kinetic_energy_change_j",
                "acoustic_energy_change_j",
                "friction_heat_j",
                "pump_heat_j",
                "shaft_drag_heat_j",
                "motor_heat_j",
            )
        },
    }


def test_constant_heat_is_exact_and_full_energy_audit_closes():
    m = caloric()
    s = m.initialize(m.physics.initialize(), {"a": 300})
    r = m.advance_record(s, record(m.physics, dt=10), heat_duties_w={"a": 100})
    assert r["state"]["temperatures_k"]["a"] == pytest.approx(301, abs=1e-11)
    assert sum(x["external_heat_duty_j"] for x in r["steps"]) == pytest.approx(1000)
    assert abs(r["combined_energy_residual_j"]) < 1e-7
    assert LiquidHeatState(**r["state"]).to_dict() == r["state"]


def test_two_heat_capacities_match_exact_relaxation_with_second_order():
    errors = []
    for dt in (1.0, 0.5, 0.25):
        m = caloric(
            walls={"a": {"heat_capacity_j_k": 2000, "fluid_ua_w_k": 100}},
            maximum_dt_s=dt,
            temporal_error_control=False,
        )
        s = m.initialize(m.physics.initialize(), {"a": 320}, {"a": 300})
        r = m.advance_record(s, record(m.physics, dt=10))
        eq = (1000 * 320 + 2000 * 300) / 3000
        difference = 20 * math.exp(-0.15 * 10)
        expected = (eq + 2 / 3 * difference, eq - 1 / 3 * difference)
        actual = (r["state"]["temperatures_k"]["a"], r["state"]["wall_temperatures_k"]["a"])
        errors.append(max(abs(a - b) for a, b in zip(actual, expected)))
        assert 300 <= min(actual) <= max(actual) <= 320
        assert abs(r["combined_energy_residual_j"]) < 1e-7
        assert m.inspect(LiquidHeatState(**r["state"]))["fluid_energy_j"] + m.inspect(
            LiquidHeatState(**r["state"])
        )["wall_energy_j"] == pytest.approx(
            m.inspect(s)["fluid_energy_j"] + m.inspect(s)["wall_energy_j"], abs=1e-7
        )
    assert errors[0] / errors[1] > 3.99 and errors[1] / errors[2] > 3.99, errors


def test_wall_to_ambient_matches_independent_matrix_exponential():
    m = caloric(
        walls={
            "a": {
                "heat_capacity_j_k": 2000,
                "fluid_ua_w_k": 100,
                "ambient_ua_w_k": 50,
                "ambient_temperature_k": 290,
            }
        },
        maximum_dt_s=0.05,
        temporal_error_control=False,
    )
    s = m.initialize(m.physics.initialize(), {"a": 320}, {"a": 300})
    r = m.advance_record(s, record(m.physics, dt=10))
    expected = 290 + expm(np.array([[-0.1, 0.1], [0.05, -0.075]]) * 10) @ np.array([30, 10])
    assert r["state"]["temperatures_k"]["a"] == pytest.approx(expected[0], abs=3e-5)
    assert r["state"]["wall_temperatures_k"]["a"] == pytest.approx(expected[1], abs=3e-5)
    assert sum(x["ambient_heat_in_j"] for x in r["steps"]) < 0
    assert abs(r["combined_energy_residual_j"]) < 1e-7


def test_stiff_passive_exchange_respects_maximum_principle():
    m = caloric(
        walls={"a": {"heat_capacity_j_k": 0.01, "fluid_ua_w_k": 1000}}, temporal_error_control=False
    )
    r = m.advance_record(
        m.initialize(m.physics.initialize(), {"a": 320}, {"a": 300}), record(m.physics, dt=0.001)
    )
    assert len(r["steps"]) > 100
    assert all(
        300 - 1e-10 <= x["final_wall_temperatures_k"]["a"] <= 320 + 1e-10 for x in r["steps"]
    )
    assert r["state"]["temperatures_k"]["a"] < 320


def test_inlet_reservoir_is_not_an_infinite_temperature_sink():
    p = passive_physics()
    m = LiquidHeatNetwork(
        p, 1000, {}, boundary_reservoirs_kg={"out": 2}, reference_temperature_k=300
    )
    s = m.initialize(p.initialize(), {"a": 300, "out": 300})
    rec = record(p)
    rec["dissipation_heat_j"]["friction"]["v"] = 100
    rec["friction_heat_j"] = 100
    # Power entering through the prescribed-pressure drive supplies this heat.
    rec["boundary_work_in_j"] = 100
    rec["midpoint_branch_volume_rates_m3_s"]["v"] = 0
    r = m.advance_record(s, rec)
    assert r["state"]["temperatures_k"]["out"] == pytest.approx(300.05)
    assert r["state"]["temperatures_k"]["a"] == 300
    assert r["steps"][0]["equipment_heat_out_j"] == 0
    assert abs(r["combined_energy_residual_j"]) < 1e-7


def test_heat_dumped_to_prescribed_boundary_is_reported_as_external_loss():
    m = caloric()
    rec = record(m.physics)
    rec["dissipation_heat_j"]["friction"]["v"] = 100
    rec["friction_heat_j"] = rec["boundary_work_in_j"] = 100
    r = m.advance_record(m.initialize(m.physics.initialize()), rec)
    assert r["state"]["temperatures_k"]["a"] == 300
    assert sum(x["equipment_heat_out_j"] for x in r["steps"]) == 100
    assert abs(r["combined_energy_residual_j"]) < 1e-8


def test_rejects_inconsistent_equipment_dissipation():
    m = caloric()
    rec = record(m.physics)
    rec["friction_heat_j"] = 100
    with pytest.raises(RuntimeError, match="ledger"):
        m.advance_record(m.initialize(m.physics.initialize()), rec)


@pytest.mark.parametrize(
    "change",
    [
        {"boundary_temperatures_k": {}},
        {"boundary_reservoirs_kg": {"out": 1}},
        {"specific_heat_j_kg_k": -1},
        {"pipe_cells": {"missing": 1}},
        {"motor_heat_to_fluid_fraction": 1.01},
        {"walls": {"missing": {"heat_capacity_j_k": 1, "fluid_ua_w_k": 1}}},
    ],
)
def test_bad_caloric_configuration(change):
    with pytest.raises((ValueError, TypeError)):
        caloric(**change)


def test_unknown_temperature_boundary_and_work_budget_rejected():
    m = caloric()
    s = m.initialize(m.physics.initialize())
    with pytest.raises(ValueError):
        m.advance_record(s, record(m.physics), boundary_temperatures_k={"missing": 300})
    with pytest.raises(RuntimeError, match="budget"):
        m.advance_record(s, record(m.physics, dt=10), max_substeps=1)


def test_finite_reservoir_depletion_does_not_create_makeup_liquid():
    p = passive_physics()
    m = LiquidHeatNetwork(
        p, 1000, {}, boundary_reservoirs_kg={"out": 1}, reference_temperature_k=300
    )
    rec = record(p)
    rec["boundary_mass_transfers_in_kg"]["out"] = 1.001
    with pytest.raises(ValueError, match="depleted"):
        m.advance_record(m.initialize(p.initialize()), rec)


def throughflow(cells=4, reverse=False, dt=0.02):
    # Manufactured conservative flux, independent of momentum solution; this
    # isolates the energy transport operator and pipe residence discretization.
    p = LiquidTransientNetwork.from_dict(
        {
            "network": {
                "model": "incompressible_liquid",
                "fluid": {"density_kg_m3": 1000, "viscosity_pa_s": 0.001},
                "nodes": [
                    {"name": "in", "pressure_pa": 500000},
                    {"name": "a"},
                    {"name": "out", "pressure_pa": 500000},
                ],
                "pipes": [
                    {
                        "name": "line",
                        "start": "in",
                        "end": "a",
                        "length_m": 1,
                        "diameter_m": math.sqrt(4 / (1000 * math.pi)),
                    }
                ],
                "pumps": [],
                "valves": [{"name": "v", "start": "a", "end": "out", "diameter_m": 0.01}],
            },
            "storage_volumes_m3": {"a": 0.001},
            "bulk_modulus_pa": 1e8,
            "reference_pressure_pa": 500000,
        }
    )
    m = LiquidHeatNetwork(
        p,
        1000,
        {"in": 310, "out": 300},
        reference_temperature_k=300,
        pipe_cells={"line": cells},
        maximum_dt_s=dt,
        temporal_error_control=False,
        maximum_temperature_change_k=100,
    )
    initial = 310 if reverse else 300
    s = m.initialize(p.initialize(), {n: initial for n in m.names})
    rec = record(p, dt=2)
    rec["midpoint_branch_volume_rates_m3_s"] = {
        "line": -0.001 if reverse else 0.001,
        "v": -0.001 if reverse else 0.001,
    }
    rec["boundary_mass_transfers_in_kg"] = {"in": -2 if reverse else 2, "out": 2 if reverse else -2}
    return m, s, rec


@pytest.mark.parametrize("reverse", [False, True])
def test_signed_pipe_transport_matches_independent_compartment_matrix(reverse):
    errors = []
    for dt in (0.02, 0.01, 0.005):
        m, s, rec = throughflow(reverse=reverse, dt=dt)
        r = m.advance_record(s, rec)
        cells = 4
        order = (
            ["a"] + [f"line::{i}" for i in range(3, -1, -1)]
            if reverse
            else [f"line::{i}" for i in range(4)] + ["a"]
        )
        rates = [1] + [cells] * cells if reverse else [cells] * cells + [1]
        matrix = np.diag(-np.asarray(rates, dtype=float))
        for i in range(1, 5):
            matrix[i, i - 1] = rates[i]
        feed = 300 if reverse else 310
        initial = 310 if reverse else 300
        expected = feed + expm(matrix * 2) @ np.full(5, initial - feed)
        actual = np.array([r["state"]["temperatures_k"][n] for n in order])
        errors.append(float(max(abs(actual - expected))))
        assert min(actual) >= 300 and max(actual) <= 310
        assert abs(r["combined_energy_residual_j"]) < 1e-7
    assert errors[0] / errors[1] > 3.9 and errors[1] / errors[2] > 3.9, errors
    assert errors[-1] < 1e-4


def test_finer_pipe_storage_reduces_premature_thermal_breakthrough():
    values = []
    for cells in (4, 16, 64):
        m, s, rec = throughflow(cells=cells, dt=0.002)
        rec["dt_s"] = 0.5
        rec["boundary_mass_transfers_in_kg"] = {"in": 0.5, "out": -0.5}
        r = m.advance_record(s, rec)
        values.append(r["state"]["temperatures_k"][f"line::{cells - 1}"] - 300)
    assert values[0] > values[1] > values[2] >= 0, values
    assert values[-1] < 1e-4


def test_draining_inventory_preserves_temperature_and_finite_receiving_mass():
    p = passive_physics()
    m = LiquidHeatNetwork(
        p, 1000, {}, boundary_reservoirs_kg={"out": 2}, reference_temperature_k=300
    )
    s = m.initialize(p.initialize(), {"a": 320, "out": 300})
    rec = record(p)
    rec["final_pressures_pa"]["a"] = 490000
    rec["midpoint_branch_volume_rates_m3_s"]["v"] = 1e-7
    rec["boundary_mass_transfers_in_kg"]["out"] = -0.0001
    r = m.advance_record(s, rec)
    assert r["state"]["temperatures_k"]["a"] == pytest.approx(320, abs=1e-10)
    assert r["state"]["temperatures_k"]["out"] == pytest.approx(
        (2 * 300 + 0.0001 * 320) / 2.0001, abs=1e-10
    )
    assert sum(r["state"]["masses_kg"].values()) == pytest.approx(3, abs=1e-12)
    assert abs(r["combined_energy_residual_j"]) < 1e-7


def test_compact_logging_preserves_final_inventory_and_energy_ledger():
    m = caloric(walls={"a": {"heat_capacity_j_k": 2000, "fluid_ua_w_k": 100}})
    data = m.to_dict()
    data["store_step_temperatures"] = False
    compact = LiquidHeatNetwork.from_dict(data)
    a = m.advance_record(
        m.initialize(m.physics.initialize(), {"a": 320}, {"a": 300}), record(m.physics, dt=2)
    )
    b = compact.advance_record(
        compact.initialize(compact.physics.initialize(), {"a": 320}, {"a": 300}),
        record(compact.physics, dt=2),
    )
    for key in ("temperatures_k", "wall_temperatures_k", "masses_kg"):
        assert a["state"][key] == b["state"][key]
    assert a["combined_energy_residual_j"] == b["combined_energy_residual_j"]
    assert all("final_temperatures_k" not in r for r in b["steps"])
    assert all(r["passive_exchange_entropy_j_k"] >= 0 for r in a["steps"])
    # A constant-coefficient record with fixed time spacing reuses its LU;
    # adaptive rejected trials legitimately require additional factorizations.
    data.update(temporal_error_control=False, maximum_dt_s=0.1)
    fixed = LiquidHeatNetwork.from_dict(data)
    cached = fixed.advance_record(
        fixed.initialize(fixed.physics.initialize(), {"a": 320}, {"a": 300}),
        record(fixed.physics, dt=2),
    )
    assert cached["linear_factorizations"] <= 2
    assert len(cached["steps"]) >= 20


def test_positive_inventory_does_not_permit_unresolvable_stiff_step():
    m = caloric(walls={"a": {"heat_capacity_j_k": 1e-12, "fluid_ua_w_k": 1e10}})
    with pytest.raises(RuntimeError, match="positivity limit"):
        m.advance_record(
            m.initialize(m.physics.initialize(), {"a": 320}, {"a": 300}), record(m.physics)
        )


def test_generated_thermal_cell_cannot_alias_prescribed_pressure_boundary():
    m, _, _ = throughflow()
    data = m.to_dict()
    for node in data["physics"]["network"]["nodes"]:
        if node["name"] == "out":
            node["name"] = "line::0"
    for valve in data["physics"]["network"]["valves"]:
        if valve["end"] == "out":
            valve["end"] = "line::0"
    data["boundary_temperatures_k"]["line::0"] = data["boundary_temperatures_k"].pop("out")
    with pytest.raises(ValueError, match="collision"):
        LiquidHeatNetwork.from_dict(data)
