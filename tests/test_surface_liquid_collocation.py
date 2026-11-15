"""Physical work/precision, fourth order and atomic failure of liquid Gauss."""

import json
import math

import numpy as np
import pytest
import torch
from check_liquid_transient import (
    coupled_model,
    coupled_rhs,
    oscillator,
    oscillator_reference,
    wave_model,
    wave_reference,
)
from scipy.integrate import solve_ivp

from surface import (
    EquipmentOperatingError,
    HydraulicConvergenceError,
    LiquidCollocation,
    LiquidTransientState,
)


def coupled():
    m = coupled_model(0.02)
    s = m.initialize({"h": 550000.0, "t": 500000.0}, {"line": 0.002}, {"a": 100.0, "b": 110.0})
    return m, s


def integrator(m):
    return LiquidCollocation(
        m,
        relative_tolerance=1e-8,
        pressure_tolerance_pa=0.1,
        flow_tolerance_m3_s=1e-11,
        speed_tolerance_rad_s=1e-6,
    )


def tensors(i):
    e = i.e
    fixed = e.tensor(
        [
            {"in": 400000.0, "out": 480000.0}.get(n, i.physics.reference_pressure_pa)
            for n in i.physics._nodes
        ]
    )
    return e.tensor([5.0, 7.0]), fixed, e.tensor([1.0])


def test_true_continuous_dae_against_independent_eliminated_equations():
    m, _ = coupled()
    i = integrator(m)
    e = i.e
    commands, fixed, openings = tensors(i)
    for ph, pt, q, wa, wb in [
        (550000.0, 500000.0, 0.002, 100.0, 110.0),
        (560000.0, 520000.0, -0.001, 98.0, 108.0),
        (540000.0, 510000.0, 0.003, 101.0, 111.0),
    ]:
        y = e.tensor([ph / e.ps, pt / e.ps, q / e.qs, 0.0, 0.0, 0.0, wa / 100, wb / 110])
        y = i._consistent(y, fixed, openings)
        rates, g, _ = e.instantaneous(y, commands, fixed, openings)
        actual = rates[[0, 1, 2, 6, 7]].numpy() * np.array([e.ps, e.ps, e.qs, 100, 110])
        np.testing.assert_allclose(
            actual, coupled_rhs(0.0, np.array([ph, pt, q, wa, wb])), rtol=1e-12, atol=1e-9
        )
        assert float(g.abs().max()) < 2e-11
        assert float(e.continuous_residual(y, rates, commands, fixed, openings).abs().max()) < 2e-11


def test_continuous_residual_exact_ad_against_finite_differences():
    m, _ = coupled()
    i = integrator(m)
    e = i.e
    cmd, fixed, op = tensors(i)
    y = i._consistent(e.tensor([1.1, 1.0, 0.002 / e.qs, 0, 0, 0, 1, 1]), fixed, op)
    yp = e.instantaneous(y, cmd, fixed, op)[0]
    direction = torch.linspace(-0.03, 0.03, len(y), dtype=y.dtype)
    fn = lambda z: e.continuous_residual(z, yp, cmd, fixed, op)
    jac = torch.func.jacfwd(fn)(y)
    fd = (fn(y + 1e-5 * direction) - fn(y - 1e-5 * direction)) / (2e-5)
    assert float((jac @ direction - fd).abs().max()) < 1e-8


def test_fourth_order_against_analytic_damped_acoustic_mode():
    errors = []
    for dt in (0.004, 0.002, 0.001):
        m = oscillator()
        i = LiquidCollocation(m)
        e = i.e
        cmd, fix, op = e.tensor([]), e.tensor([600000.0, 600000.0]), e.tensor([])
        y = e.tensor([601000.0 / e.ps, 599000.0 / e.ps, 0.0])
        for _ in range(round(0.02 / dt)):
            y, _, valid, _, _ = i._trial(y, dt, cmd, fix, op)
            assert valid
        expected, _ = oscillator_reference(0.02)
        errors.append(float(max(abs(y[:2].numpy() * e.ps - expected))))
    assert min(errors[k] / errors[k + 1] for k in (0, 1)) > 14


def test_coupled_accuracy_and_independent_mass_energy_journals():
    m, s = coupled()
    i = integrator(m)
    before = s.to_dict()
    r = i.advance(s, 1.0, motor_torques_nm={"a": 5.0, "b": 7.0})
    reference = solve_ivp(
        coupled_rhs,
        (0.0, 1.0),
        [550000.0, 500000.0, 0.002, 100.0, 110.0],
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
        dense_output=True,
    )
    times = np.array([step["end_time_s"] for step in r["steps"]])
    ys = np.array([step["scaled_state"] for step in r["steps"]])
    e = i.e
    physical = ys[:, [0, 1, 2, 6, 7]] * np.r_[[e.ps] * 2, e.qs, [100, 110]]
    errors = abs(physical - reference.sol(times).T).max(axis=0)
    assert np.all(errors < [1.0, 1.0, 1e-8, 1e-4, 1e-4])
    assert r["state"]["time_s"] == 1.0
    assert math.fsum(step["dt_s"] for step in r["steps"]) == pytest.approx(1.0, abs=1e-14)
    assert all(step["error_ratio"] <= 1 for step in r["steps"])
    # Independently reconstruct stage losses/energy, without _powers or AD.
    previous = np.array([550000.0, 500000.0, 0.002, 100.0, 110.0])
    for step, end in zip(r["steps"], physical):
        energy_in = heat = 0.0
        outward_h = outward_t = 0.0
        for z in step["stage_scaled_states"]:
            ph, pt = np.array(z[:2]) * e.ps
            q, qa, qb, qv = np.array(z[2:6]) * e.qs
            wa, wb = np.array(z[6:]) * [100, 110]
            resistance = 128 * 0.1 * 10 / (math.pi * 0.05**4)
            shaft_power = (ph - 400000.0) * np.array([qa, qb]) / [0.8, 0.75] + np.array([5, 8]) * (
                np.array([wa, wb]) / [100, 110]
            ) ** 3
            electric = 5 * wa / 0.9 + 7 * wb / 0.9
            motor_heat = electric - 5 * wa - 7 * wb
            heat += (
                0.5
                * step["dt_s"]
                * (
                    motor_heat
                    + 0.01 * wa**2
                    + 0.015 * wb**2
                    + sum(shaft_power - (ph - 400000.0) * np.array([qa, qb]))
                    + resistance * q * q
                    + (pt - 480000.0) * qv
                )
            )
            energy_in += (
                0.5
                * step["dt_s"]
                * (electric + (400000.0 - 500000.0) * (qa + qb) - (480000.0 - 500000.0) * qv)
            )
            outward_h += 0.5 * step["dt_s"] * (q - qa - qb)
            outward_t += 0.5 * step["dt_s"] * (qv - q)
        compliance, inertance = 0.1 / 1e7, 1000 * 10 / (math.pi * 0.05**2 / 4)

        def energy(y, compliance=compliance, inertance=inertance):
            return (
                0.5 * compliance * sum((y[:2] - 500000.0) ** 2)
                + 0.5 * inertance * y[2] ** 2
                + 0.25 * y[3] ** 2
                + 0.35 * y[4] ** 2
            )

        assert abs(energy_in - heat - (energy(end) - energy(previous))) < 1e-7
        assert abs(1000 * (compliance * (end[0] - previous[0]) + outward_h)) < 1e-8
        assert abs(1000 * (compliance * (end[1] - previous[1]) + outward_t)) < 1e-8
        assert abs(step["energy_residual_j"]) < 1e-7
        assert step["max_unit_shaft_energy_residual_j"] < 1e-7
        previous = end
    assert s.to_dict() == before


def test_restart_and_failure_are_atomic(monkeypatch):
    m, s = coupled()
    i = integrator(m)
    first = i.advance(s, 0.1, motor_torques_nm={"a": 5.0, "b": 7.0})
    restart = LiquidTransientState(**json.loads(json.dumps(first["state"])))
    kwargs = {"motor_torques_nm": {"a": 5.0, "b": 7.0}, "initial_step_s": first["next_step_s"]}
    second = i.advance(restart, 0.1, **kwargs)
    replay = integrator(m).advance(restart, 0.1, **kwargs)
    assert second == replay
    before = restart.to_dict()

    def fail(*a, **kw):
        raise HydraulicConvergenceError("Injected failed stage solve")

    monkeypatch.setattr(i, "_trial", fail)
    with pytest.raises(HydraulicConvergenceError, match="attempt budget"):
        i.advance(restart, 0.1, max_attempts=2, **kwargs)
    assert restart.to_dict() == before


def test_rejected_trial_recovers_without_accepting_failed_history(monkeypatch):
    m, s = coupled()
    i = integrator(m)
    original = i._trial
    attempts = []

    def once(*a, **kw):
        attempts.append(a[1])
        if len(attempts) == 1:
            raise HydraulicConvergenceError("Injected first attempt failure")
        return original(*a, **kw)

    monkeypatch.setattr(i, "_trial", once)
    r = i.advance(s, 0.02, motor_torques_nm={"a": 5.0, "b": 7.0})
    assert r["rejected_steps"] >= 1
    assert r["state"]["time_s"] == 0.02
    assert r["steps"][0]["time_s"] == 0.0
    assert r["steps"][0]["dt_s"] < attempts[0]


@pytest.mark.parametrize("change", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_tolerances_rejected(change):
    m, _ = coupled()
    with pytest.raises((ValueError, TypeError)):
        LiquidCollocation(m, relative_tolerance=change)


def test_station_power_allocation_and_unilateral_shaft_constraints():
    m, _ = coupled()
    data = m.to_dict()
    data["max_station_electrical_draw_w"] = 100.0
    limited = type(m).from_dict(data)
    i = integrator(limited)
    e = i.e
    cmd, fixed, op = tensors(i)
    y = i._consistent(e.tensor([1.1, 1.0, 0.002 / e.qs, 0, 0, 0, 1, 1]), fixed, op)
    _, _, physical = e.instantaneous(y, cmd, fixed, op)
    electrical = (physical[6] * physical[2] / e.motor_eta).sum()
    assert float(electrical) == pytest.approx(100.0, abs=1e-10)
    stopped = y.clone()
    stopped[6:] = 0.0
    rates, _, _ = e.instantaneous(stopped, e.tensor([-5.0, -5.0]), fixed, op)
    assert torch.equal(rates[6:], torch.zeros_like(rates[6:]))
    # Speed limits constrain acceleration, rather than a fictitious dt=1 cap.
    at_limit = y.clone()
    at_limit[6:] = e.maxw / e.nominal
    unlimited = integrator(m).e
    rates, _, _ = unlimited.instantaneous(at_limit, e.tensor([20.0, 20.0]), fixed, op)
    assert bool((rates[6:] <= 1e-14).all())


def test_stationary_closed_pumps_keep_exact_equilibrium():
    m, _ = coupled()
    s = m.initialize({"h": 480000.0, "t": 480000.0}, {"line": 0.0}, {"a": 0.0, "b": 0.0})
    r = integrator(m).advance(s, 0.1)
    assert r["state"]["pressures_pa"] == dict(s.pressures_pa)
    assert r["state"]["speeds_rad_s"] == {"a": 0.0, "b": 0.0}
    assert r["state"]["pipe_volume_rates_m3_s"] == {"line": 0.0}
    assert all(step["energy_residual_j"] == 0.0 for step in r["steps"])


def test_coupled_stage_pattern_matches_dense_tangent():
    m, _ = coupled()
    i = integrator(m)
    e = i.e
    cmd, fixed, op = tensors(i)
    y = i._consistent(e.tensor([1.1, 1.0, 0.002 / e.qs, 0, 0, 0, 1, 1]), fixed, op)
    stage = torch.stack((y, y)).flatten()
    jac = torch.func.jacfwd(lambda z: i._stage_residual(z, y, 0.01, cmd, fixed, op))(stage)
    support = set(zip(*i.pattern))
    assert all((r, c) in support for r, c in torch.nonzero(jac != 0).tolist())


def test_sparse_distributed_acoustic_graph_agrees_with_dense_solution():
    m = wave_model(64)
    s = m.initialize(dict(zip(m.free, wave_reference(64, 0))))
    sparse = LiquidCollocation(m)
    dense = LiquidCollocation(m)
    assert sparse.e.size * 2 > sparse.newton.dense_limit
    dense.newton.dense_limit = 1024
    a = sparse.advance(s, 1e-4)
    b = dense.advance(s, 1e-4)
    assert a["state"]["pressures_pa"] == pytest.approx(b["state"]["pressures_pa"], abs=1e-5, rel=0)
    assert a["state"]["pipe_volume_rates_m3_s"] == pytest.approx(
        b["state"]["pipe_volume_rates_m3_s"], abs=1e-9, rel=0
    )
    assert max(abs(step["energy_residual_j"]) for step in a["steps"]) < 1e-7


def test_model_changes_and_unsupported_schedule_states_are_rejected():
    m, s = coupled()
    i = integrator(m)
    tripped = s.to_dict()
    tripped["tripped"] = True
    with pytest.raises(ValueError, match="event-aware"):
        i.advance(LiquidTransientState(**tripped), 0.01)
    m.maximum_dt_s *= 2
    replacement = LiquidTransientState(**{**s.to_dict(), "model_id": m.model_id})
    with pytest.raises(ValueError, match="model changed"):
        i.advance(replacement, 0.01)


def test_total_pump_shaft_power_is_not_silently_accepted():
    m, _ = coupled()
    data = m.to_dict()
    data["network"]["pumps"][0]["max_shaft_power_w"] = 1.0
    restricted = type(m).from_dict(data)
    s = restricted.initialize(
        {"h": 550000.0, "t": 500000.0}, {"line": 0.002}, {"a": 100.0, "b": 110.0}
    )
    with pytest.raises(EquipmentOperatingError, match="physical envelope"):
        integrator(restricted).advance(s, 0.01, motor_torques_nm={"a": 5.0, "b": 7.0})


def test_linear_acoustic_error_controller_is_not_blind_to_tighter_tolerances():
    m = oscillator(0.005)
    s = m.initialize({"a": 601000.0, "b": 599000.0})
    weak = LiquidCollocation(
        m, relative_tolerance=1e-3, pressure_tolerance_pa=1.0, flow_tolerance_m3_s=1e-7
    )
    strict = LiquidCollocation(
        m, relative_tolerance=1e-7, pressure_tolerance_pa=1e-4, flow_tolerance_m3_s=1e-10
    )
    a = weak.advance(s, 0.02)
    b = strict.advance(s, 0.02)
    assert len(b["steps"]) > len(a["steps"])
    expected, _ = oscillator_reference(0.02)

    def error(report):
        return max(
            abs(np.array([report["state"]["pressures_pa"][n] for n in ("a", "b")]) - expected)
        )

    assert error(b) < error(a)
    assert error(b) < 0.01
    assert all(step["error_ratio"] <= 1 for step in b["steps"])


@pytest.mark.parametrize("duration", [0.02 + 1e-12, 1e-12])
def test_clipped_reporting_tail_does_not_collapse_next_proposal(duration):
    from surface import LiquidTransientState

    m = oscillator(0.02)
    initial = m.initialize({"a": 600000.0, "b": 600000.0})
    solver = LiquidCollocation(m)
    first = solver.advance(initial, duration, initial_step_s=0.02)
    assert first["state"]["time_s"] == duration
    assert len(first["steps"]) == (2 if duration > 0.02 else 1)
    assert first["steps"][-1]["dt_s"] < m.minimum_dt_s
    assert first["next_step_s"] == 0.02
    restart = LiquidTransientState(**first["state"])
    following = solver.advance(restart, 0.02, initial_step_s=first["next_step_s"])
    assert len(following["steps"]) <= 2


def test_clipped_step_with_measured_large_error_can_reduce_next_proposal(monkeypatch):
    m = oscillator(0.02)
    initial = m.initialize({"a": 600000.0, "b": 600000.0})
    solver = LiquidCollocation(m)
    original = solver._trial

    def near_budget(*a, **kw):
        state, _, valid, record, evaluations = original(*a, **kw)
        return state, 0.9, valid, record, evaluations

    monkeypatch.setattr(solver, "_trial", near_budget)
    result = solver.advance(initial, 0.001, initial_step_s=0.02)
    assert result["next_step_s"] < 0.001
