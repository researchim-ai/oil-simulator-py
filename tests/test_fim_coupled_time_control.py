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


def test_rejection_feedback_skips_known_oversized_proposals():
    sim = make(maximum=10.)
    sim.sim_params.update(initial_time_step_days=10. / 86400, newton_target_iter=100)
    coupled = ErrorControlledStepper(sim.fim_solver)
    coupled.limit_rejected_time_step_s = lambda proposed_s, *, failed_dt_s: min(proposed_s, .9)
    sim.facility_coupling = coupled
    assert sim.run_step(10.)
    assert coupled.rejected == [10.]
    assert sim.current_time == 10.


@pytest.mark.parametrize('bad', [0., -1., math.nan, 6., True])
def test_invalid_rejection_feedback_is_rejected(bad):
    sim = make(maximum=10.)
    sim.sim_params.update(initial_time_step_days=10. / 86400)
    coupled = ErrorControlledStepper(sim.fim_solver)
    coupled.limit_rejected_time_step_s = lambda *args, **kwargs: bad
    sim.facility_coupling = coupled
    with pytest.raises(ValueError, match='rejection limiter'):
        sim.run_step(10.)


@pytest.mark.parametrize('error, expected', [(4., 4.5), (100., 1.), (1e8, 1.)])
def test_dynamic_receiver_rejected_defect_predicts_bounded_retry(error, expected):
    state = SimpleNamespace(last_attempt={
        'success': False, 'dt_s': 10., 'temporal_error_ratio': error,
    })
    limit = DynamicReceiverCoupledFIM.limit_rejected_time_step_s
    assert limit(state, 5., failed_dt_s=10.) == pytest.approx(expected)
    assert limit(state, .5, failed_dt_s=10.) == .5
    assert limit(state, 5., failed_dt_s=9.) == 5.  # stale attempt


def test_dynamic_receiver_without_time_defect_keeps_driver_cut():
    limit = DynamicReceiverCoupledFIM.limit_rejected_time_step_s
    for attempt in (None, {'success': False, 'dt_s': 10., 'reason': 'nonlinear failure'},
                    {'success': True, 'dt_s': 10., 'temporal_error_ratio': .1}):
        assert limit(SimpleNamespace(last_attempt=attempt), 5., failed_dt_s=10.) == 5.


@pytest.mark.parametrize('error', [True, 1., -1., math.nan, math.inf])
def test_dynamic_receiver_rejects_invalid_retry_estimate(error):
    state = SimpleNamespace(last_attempt={
        'success': False, 'dt_s': 10., 'temporal_error_ratio': error,
    })
    with pytest.raises(ValueError, match='rejected coupled temporal'):
        DynamicReceiverCoupledFIM.limit_rejected_time_step_s(state, 5., failed_dt_s=10.)


def test_real_driver_uses_dynamic_receiver_rejection_policy():
    sim = make(maximum=10.)
    sim.sim_params.update(initial_time_step_days=10. / 86400, newton_target_iter=100)

    class MeasuredStepper(ErrorControlledStepper):
        limit_rejected_time_step_s = DynamicReceiverCoupledFIM.limit_rejected_time_step_s

        def step(self, dt, current_time=None):
            accepted = super().step(dt, current_time)
            if not accepted:
                self.last_attempt = {
                    'success': False, 'dt_s': dt, 'temporal_error_ratio': dt * dt,
                }
            return accepted

    coupled = MeasuredStepper(sim.fim_solver)
    sim.facility_coupling = coupled
    assert sim.run_step(10.)
    assert coupled.rejected == [10.]
    assert sim.current_time == 10.
    assert all(dt <= 1. for _, dt in coupled.accepted)
