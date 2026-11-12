"""Short reporting intervals and large epochs must execute, never epsilon-skip."""

import copy
import math

import pytest
import torch
from test_fim_wells import BASE, _objects

from simulator.simulation import Simulator


def make(epoch=0.0, *, maximum=1.0):
    cfg = copy.deepcopy(BASE)
    cfg["wells"] = []
    cfg["reservoir"]["dimensions"] = [1, 1, 1]
    cfg["simulation"].update(
        initial_time_step_days=maximum / 86400.0,
        max_time_step_days=maximum / 86400.0,
        min_time_step_days=1e-12 / 86400.0,
    )
    r, f, w = _objects(cfg)
    sim = Simulator(r, f, w, cfg["simulation"], device=torch.device("cpu"))
    sim.current_time = epoch
    return sim


@pytest.mark.parametrize("duration", [1e-9, 1e-6, 1.000001e-6])
def test_short_interval_really_calls_physical_solver_and_hook(duration):
    sim = make()
    actual = []
    hook = []
    step = sim.fim_solver.step

    def record(dt, current_time=None):
        actual.append((current_time, dt))
        return step(dt, current_time=current_time)

    sim.fim_solver.step = record
    sim.substep_hook = hook.append
    assert sim.run_step(duration)
    assert actual and math.fsum(dt for _, dt in actual) == duration
    assert sim.current_time == duration and len(hook) == len(actual)


def test_large_epoch_odd_ulp_tail_is_integrated_with_no_clock_only_snap():
    epoch = float(2**20)
    ulp = math.ulp(epoch)
    sim = make(epoch, maximum=2 * ulp)
    actual = []
    step = sim.fim_solver.step

    def record(dt, current_time=None):
        actual.append((current_time, dt))
        return step(dt, current_time=current_time)

    sim.fim_solver.step = record
    assert sim.run_step(5 * ulp)
    assert sim.current_time == epoch + 5 * ulp
    assert math.fsum(dt for _, dt in actual) == 5 * ulp
    assert max(dt for _, dt in actual) <= 2 * ulp


@pytest.mark.parametrize("duration", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_interval_is_rejected_before_state_change(duration):
    sim = make()
    before = sim.fluid.pressure.clone()
    with pytest.raises(ValueError):
        sim.run_step(duration)
    assert sim.current_time == 0.0 and torch.equal(sim.fluid.pressure, before)


def test_days_conversion_roundoff_does_not_create_unresolvable_final_tail():
    duration = 1e-4
    sim = make(maximum=math.nextafter(duration, 0.0))
    actual = []
    step = sim.fim_solver.step

    def record(dt, current_time=None):
        actual.append(dt)
        return step(dt, current_time=current_time)

    sim.fim_solver.step = record
    assert sim.run_step(duration)
    assert actual == [duration] and sim.current_time == duration


def test_unrepresentable_interval_is_rejected_before_any_solver_call():
    epoch = float(2**20)
    sim = make(epoch)

    def forbidden(*a, **kw):
        raise AssertionError("Physical solve on an unrepresentable interval")

    sim.fim_solver.step = forbidden
    with pytest.raises(ValueError, match="clock resolution"):
        sim.run_step(math.ulp(epoch) / 4)
    assert sim.current_time == epoch


@pytest.mark.parametrize("policy", ["iteration", "opm"])
def test_uniform_proposal_does_not_drift_into_a_spurious_ninth_step(policy):
    width = 2.5e-7
    sim = make(maximum=width)
    sim.sim_params.update(dt_increase_factor=1.0, newton_target_iter=30)
    if policy == "opm":
        sim.sim_params["dt_control"] = "opm"
        if not hasattr(sim.fluid, "s_g"):
            sim.fluid.s_g = torch.zeros_like(sim.fluid.s_w)
        sim._opm_dt = lambda dt, *args: dt
    actual = []
    step = sim.fim_solver.step

    def record(dt, current_time=None):
        actual.append(dt)
        return step(dt, current_time=current_time)

    sim.fim_solver.step = record
    assert sim.run_step(8 * width)
    assert len(actual) == 8 and math.fsum(actual) == 8 * width
    assert all(abs(dt - width) <= 8 * math.ulp(width) for dt in actual)
    assert sim.current_time == 8 * width
