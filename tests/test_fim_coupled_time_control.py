"""Surface time error must bound the actual driver, including report tails."""

import math
from types import SimpleNamespace

import pytest
import torch
from test_fim_reporting_clock import make

from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM


class ErrorControlledStepper:
    """Real reservoir steps plus a manufactured, independently known dt**2 defect."""

    limit_next_time_step_s = DynamicReceiverCoupledFIM.limit_next_time_step_s

    def __init__(self, solver):
        self.solver = solver
        self.report = None
        self.last_attempt = None
        self.accepted = []
        self.rejected = []

    def step(self, dt, current_time=None):
        if dt > 1.0:
            self.rejected.append(dt)
            self.last_attempt = {"reason": "manufactured surface temporal error"}
            return False
        result = self.solver.step(dt, current_time=current_time)
        if result:
            self.report = {"dt_s": dt, "temporal_error_ratio": dt * dt}
            self.accepted.append((current_time, dt))
        return result


@pytest.mark.parametrize("policy", ["iteration", "opm"])
def test_accepted_coupled_error_prevents_growth_rejection_and_tail_bridging(policy):
    sim = make(maximum=2.0)
    sim.sim_params.update(initial_time_step_days=1.0 / 86400, newton_target_iter=100)
    if policy == "opm":
        sim.sim_params["dt_control"] = "opm"
        if not hasattr(sim.fluid, "s_g"):
            sim.fluid.s_g = torch.zeros_like(sim.fluid.s_w)
        sim._opm_dt = lambda dt, *args: dt * 2.0
    coupled = ErrorControlledStepper(sim.fim_solver)
    sim.facility_coupling = coupled
    assert sim.run_step(2.1)
    assert coupled.rejected == []
    assert [dt for _, dt in coupled.accepted] == pytest.approx([1.0, 0.9, 0.2])
    assert sim.current_time == 2.1
    stats = sim.last_fim_interval_stats
    assert stats["accepted_steps"] == 3 and stats["rejected_steps"] == 0
    assert stats["minimum_accepted_step_s"] == pytest.approx(0.2)
    assert stats["integrated_duration_s"] == 2.1
    assert stats["time_error_limited_proposals"] >= 1
    assert sim._fim_dt == pytest.approx(0.9)
    # Low-error short tail preserves the useful proposal for the next report.
    assert sim.run_step(0.9)
    assert coupled.accepted[-1] == pytest.approx((2.1, 0.9))
    assert coupled.rejected == [] and sim.current_time == 3.0


@pytest.mark.parametrize("error", [0.0, 0.04, 0.25])
def test_low_error_clipped_tail_keeps_a_longer_valid_proposal(error):
    state = SimpleNamespace(report={"dt_s": 1e-12, "temporal_error_ratio": error})
    assert DynamicReceiverCoupledFIM.limit_next_time_step_s(state, 1.0, clipped=True) == 1.0


def test_large_error_tail_still_limits_next_report_and_zero_error_does_not_cap_growth():
    state = SimpleNamespace(report={"dt_s": 0.5, "temporal_error_ratio": 1.0})
    assert DynamicReceiverCoupledFIM.limit_next_time_step_s(state, 2.0, clipped=True) == 0.45
    state.report["temporal_error_ratio"] = 0.0
    assert DynamicReceiverCoupledFIM.limit_next_time_step_s(state, 2.0) == 2.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, 1.01])
def test_invalid_accepted_estimate_cannot_disable_temporal_control(bad):
    state = SimpleNamespace(report={"dt_s": 1.0, "temporal_error_ratio": bad})
    with pytest.raises(ValueError, match="temporal estimate"):
        DynamicReceiverCoupledFIM.limit_next_time_step_s(state, 2.0)


def test_disabled_prediction_keeps_accuracy_checks_and_reports_wasted_attempts():
    sim = make(maximum=2.0)
    sim.sim_params.update(
        initial_time_step_days=1.0 / 86400, newton_target_iter=100, coupled_time_control=False
    )
    coupled = ErrorControlledStepper(sim.fim_solver)
    sim.facility_coupling = coupled
    assert sim.run_step(2.1)
    assert coupled.rejected and all(dt > 1 for dt in coupled.rejected)
    assert all(dt <= 1 for _, dt in coupled.accepted)
    assert sim.last_fim_interval_stats["rejected_steps"] == len(coupled.rejected)
    assert sim.last_fim_interval_stats["time_error_limited_proposals"] == 0
    assert sim.current_time == 2.1
