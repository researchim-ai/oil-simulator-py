"""Shared hydraulic/thermal stages, finite inventories and energy closure."""

import json

import numpy as np
import pytest
from check_liquid_thermal_collocation import ConservativeReference, case
from check_liquid_transient import oscillator
from scipy.integrate import solve_ivp

from surface import (
    HydraulicConvergenceError,
    LiquidHeatNetwork,
    LiquidHeatState,
    LiquidTransientState,
    ThermalLiquidCollocation,
    VesselWall,
)
from surface.liquid_thermal_collocation import _HeatCoupling


def observations(report, reference):
    out = []
    h = reference.heat
    from surface.torch_liquid import TorchLiquidEquations

    e = TorchLiquidEquations(h.physics)
    for physical, thermal in zip(report["hydraulics"]["steps"], report["thermal_steps"]):
        z = np.array(physical["scaled_state"])
        # The standalone hydraulic scaling is part of the fixture, not a heat
        # operator or the production energy audit.
        out.append(
            np.r_[
                z[[0, 1, 2, 6, 7]] * np.r_[[e.ps] * 2, e.qs, [100, 110]],
                [thermal["temperatures_k"][n] for n in h.names],
                [thermal["wall_temperatures_k"][n] for n in h.wall_names],
                [thermal["masses_kg"][n] for n in reference.reservoirs],
            ]
        )
    return np.array(out)


@pytest.mark.parametrize("cells,finite", [(1, False), (4, False), (4, True)])
def test_coupled_solution_against_independent_conservative_ode(cells, finite):
    _, heat, p, t = case(cells, finite)
    ref = ConservativeReference(heat)
    initial = ref.encode(p, t)
    solution = solve_ivp(
        ref, (0.0, 0.5), initial, method="DOP853", rtol=1e-12, atol=1e-9, dense_output=True
    )
    assert solution.success
    result = ThermalLiquidCollocation(heat).advance(
        p, t, 0.5, motor_torques_nm={"a": 5.0, "b": 7.0}
    )
    times = [s["end_time_s"] for s in result["thermal_steps"]]
    expected = np.array([ref.observe(solution.sol(v)) for v in times])
    actual = observations(result, ref)
    errors = abs(actual - expected).max(axis=0)
    budget = np.r_[
        [1.0, 1.0, 1e-8, 1e-4, 1e-4],
        [1e-5] * (len(heat.names) + len(heat.wall_names)),
        [1e-7] * len(ref.reservoirs),
    ]
    assert np.all(errors < budget), errors
    assert len(result["hydraulics"]["steps"]) == len(result["thermal_steps"])
    heat._check(
        LiquidHeatState(**result["state"]["heat"]),
        LiquidTransientState(**result["state"]["hydraulics"]),
    )
    if finite:
        assert result["state"]["heat"]["masses_kg"]["in"] < 1000.0
        assert result["state"]["heat"]["temperatures_k"]["in"] == pytest.approx(330.0, abs=1e-10)


def test_combined_energy_is_reconstructed_from_stocks_and_external_work():
    physics, heat, p, t = case()
    r = ThermalLiquidCollocation(heat).advance(p, t, 0.3, motor_torques_nm={"a": 5.0, "b": 7.0})
    before = heat.inspect(t)
    after = heat.inspect(LiquidHeatState(**r["state"]["heat"]))
    pm0 = physics.inspect(p)
    pm1 = physics.inspect(LiquidTransientState(**r["state"]["hydraulics"]))
    mechanical_change = sum(
        pm1[n] - pm0[n]
        for n in ("acoustic_energy_j", "pipe_kinetic_energy_j", "shaft_kinetic_energy_j")
    )
    thermal_change = (
        after["fluid_energy_j"]
        + after["wall_energy_j"]
        - before["fluid_energy_j"]
        - before["wall_energy_j"]
    )
    external = 0.0
    for hy, th in zip(r["hydraulics"]["steps"], r["thermal_steps"]):
        dissipated = sum(
            hy[n] for n in ("motor_heat_j", "shaft_drag_heat_j", "pump_heat_j", "friction_heat_j")
        )
        escaped = dissipated - th["absorbed_dissipation_j"]
        assert escaped >= -1e-7
        assert th["escaped_dissipation_j"] == pytest.approx(escaped, abs=1e-9)
        external += (
            hy["electrical_energy_j"]
            + hy["boundary_work_in_j"]
            + th["advective_energy_in_j"]
            + th["ambient_heat_in_j"]
            + th["external_heat_in_j"]
            - escaped
        )
    assert abs(mechanical_change + thermal_change - external) < 2e-6
    assert r["state"]["heat"]["time_s"] == r["state"]["hydraulics"]["time_s"] == 0.3


def test_thermal_refusal_rejects_hydraulics_too(monkeypatch):
    _, heat, p, t = case()
    before = p.to_dict(), t.to_dict()
    original = _HeatCoupling.trial
    calls = []

    def reject_once(self, *a, **kw):
        calls.append(a[3])
        candidate, error, valid = original(self, *a, **kw)
        return (candidate, max(error, 2.0), False) if len(calls) == 1 else (candidate, error, valid)

    monkeypatch.setattr(_HeatCoupling, "trial", reject_once)
    r = ThermalLiquidCollocation(heat).advance(p, t, 0.03, motor_torques_nm={"a": 5.0, "b": 7.0})
    assert r["hydraulics"]["rejected_steps"] >= 1
    assert r["thermal_steps"][0]["dt_s"] < calls[0]
    assert len(r["thermal_steps"]) == len(r["hydraulics"]["steps"])
    assert (p.to_dict(), t.to_dict()) == before


def test_sparse_thermal_linear_solve_matches_dense_without_host_fallback():
    _, heat, p, t = case(64)
    a = ThermalLiquidCollocation(heat, thermal_dense_limit=0).advance(
        p, t, 0.02, motor_torques_nm={"a": 5.0, "b": 7.0}
    )
    b = ThermalLiquidCollocation(heat, thermal_dense_limit=1000).advance(
        p, t, 0.02, motor_torques_nm={"a": 5.0, "b": 7.0}
    )
    assert a["state"]["heat"]["temperatures_k"] == pytest.approx(
        b["state"]["heat"]["temperatures_k"], abs=1e-9, rel=0
    )
    assert a["state"]["heat"]["wall_temperatures_k"] == pytest.approx(
        b["state"]["heat"]["wall_temperatures_k"], abs=1e-9, rel=0
    )
    assert max(s["linear_residual_k"] for s in a["thermal_steps"]) < 1e-10


def test_reverse_flow_closed_network_converts_dissipation_once():
    pmodel = oscillator(0.0005)
    p = pmodel.initialize({"a": 601000.0, "b": 599000.0}, {"line": -0.001})
    heat = LiquidHeatNetwork(
        pmodel, 2000.0, {}, pipe_cells={"line": 3}, reference_temperature_k=300.0
    )
    t = heat.initialize(p, {"a": 290.0, "b": 330.0})
    r = ThermalLiquidCollocation(heat).advance(p, t, 0.002)
    assert r["state"]["hydraulics"]["pipe_volume_rates_m3_s"]["line"] < 0
    p1 = LiquidTransientState(**r["state"]["hydraulics"])
    t1 = LiquidHeatState(**r["state"]["heat"])

    def total(p, t):
        mp = pmodel.inspect(p)
        mh = heat.inspect(t)
        return (
            sum(
                mp[n]
                for n in ("acoustic_energy_j", "pipe_kinetic_energy_j", "shaft_kinetic_energy_j")
            )
            + mh["fluid_energy_j"]
            + mh["wall_energy_j"]
        )

    assert abs(total(p1, t1) - total(p, t)) < 1e-6
    assert t1.to_dict()["temperatures_k"]["a"] > 290.0


def test_restart_replays_and_depleted_reservoir_cannot_create_makeup_liquid():
    _, heat, p, t = case()
    solver = ThermalLiquidCollocation(heat)
    first = solver.advance(p, t, 0.1, motor_torques_nm={"a": 5.0, "b": 7.0})
    saved = json.loads(json.dumps(first["state"]))
    p1, t1 = LiquidTransientState(**saved["hydraulics"]), LiquidHeatState(**saved["heat"])
    kwargs = {
        "motor_torques_nm": {"a": 5.0, "b": 7.0},
        "initial_step_s": first["hydraulics"]["next_step_s"],
    }
    assert solver.advance(p1, t1, 0.1, **kwargs) == ThermalLiquidCollocation(heat).advance(
        p1, t1, 0.1, **kwargs
    )
    data = heat.to_dict()
    data["boundary_reservoirs_kg"]["in"] = 0.0001
    small = LiquidHeatNetwork.from_dict(data)
    t0 = small.initialize(p, {"in": 330.0})
    before = p.to_dict(), t0.to_dict()
    with pytest.raises(HydraulicConvergenceError):
        ThermalLiquidCollocation(small).advance(
            p, t0, 0.1, motor_torques_nm={"a": 5.0, "b": 7.0}, max_attempts=40
        )
    assert (p.to_dict(), t0.to_dict()) == before


def passive(dt=0.2, tolerance=1.0, relative=1e-3):
    m = oscillator(dt)
    p = m.initialize({"a": 600000.0, "b": 600000.0})
    h = LiquidHeatNetwork(
        m,
        2000.0,
        {},
        reference_temperature_k=300.0,
        maximum_dt_s=dt,
        walls={"a": VesselWall(10000.0, 5000.0, max_temperature_change_k=100.0)},
        maximum_temperature_change_k=100.0,
        temperature_tolerance_k=tolerance,
        relative_tolerance=relative,
    )
    return h, p, h.initialize(p, {"a": 350.0}, {"a": 300.0})


def test_coupled_thermal_fourth_order_against_exact_two_capacity_relaxation():
    import math

    errors = []
    for dt in (0.2, 0.1, 0.05):
        heat, p, t = passive(dt)
        r = ThermalLiquidCollocation(heat).advance(p, t, 1.0, initial_step_s=dt)
        expected = 1000 / 3 + 50 / 3 * math.exp(-0.75)
        errors.append(abs(r["state"]["heat"]["temperatures_k"]["a"] - expected))
    assert min(errors[i] / errors[i + 1] for i in (0, 1)) > 14.0


def test_thermal_accuracy_controls_the_shared_grid_when_hydraulics_are_static():
    weak, p, t = passive()
    strong, p1, t1 = passive(tolerance=1e-9, relative=1e-12)
    a = ThermalLiquidCollocation(weak).advance(p, t, 0.1)
    b = ThermalLiquidCollocation(strong).advance(p1, t1, 0.1)
    assert len(b["thermal_steps"]) > len(a["thermal_steps"])
    assert len(b["thermal_steps"]) == len(b["hydraulics"]["steps"])
    assert b["hydraulics"]["rejected_steps"] > 0
    assert all(s["temperature_error_ratio"] <= 1 for s in b["thermal_steps"])
    assert b["state"]["hydraulics"]["pressures_pa"] == dict(p1.pressures_pa)


def test_inventory_error_can_control_grid_independently_of_temperature():
    from surface import LiquidTransientNetwork

    physics, heat, _, _ = case()
    cfg = physics.to_dict()
    cfg.update(
        relative_tolerance=1e-3,
        pressure_tolerance_pa=1000.0,
        flow_tolerance_m3_s=1e-5,
        speed_tolerance_rad_s=1.0,
    )
    physics = LiquidTransientNetwork.from_dict(cfg)
    cfg = heat.to_dict()
    cfg.update(physics=physics.to_dict(), relative_tolerance=1e-12, temperature_tolerance_k=1.0)
    heat = LiquidHeatNetwork.from_dict(cfg)
    p = physics.initialize(
        {"h": 550000.0, "t": 500000.0}, {"line": 0.002}, {"a": 100.0, "b": 110.0}
    )
    t = heat.initialize(p, {"in": 330.0})
    a = ThermalLiquidCollocation(heat, mass_tolerance_kg=0.1).advance(
        p, t, 0.1, motor_torques_nm={"a": 5.0, "b": 7.0}
    )
    b = ThermalLiquidCollocation(heat, mass_tolerance_kg=1e-11).advance(
        p, t, 0.1, motor_torques_nm={"a": 5.0, "b": 7.0}
    )
    assert len(b["thermal_steps"]) > len(a["thermal_steps"])
    assert all(s["mass_error_ratio"] <= 1 for s in b["thermal_steps"])
