"""A completed infeasible pressure probe must never become an accepted step."""
import math
from types import SimpleNamespace

import pytest

from simulator.receiver_subcycling import advance_receiver_locally
from surface.stratified_pr_flow import ReceiverFeedPressureError


@pytest.mark.parametrize('later_failure', [False, True])
def test_completed_probe_rejects_even_discarded_pressure_violation(monkeypatch, later_failure):
    from simulator import receiver_subcycling
    from surface.liquid import HydraulicConvergenceError

    end = SimpleNamespace(column=object())
    monkeypatch.setattr(receiver_subcycling.StratifiedThermalState, 'from_dict', lambda raw: end)
    violations = [ReceiverFeedPressureError(1e6, 1e6 + 50.),
                  ReceiverFeedPressureError(1e6, 1e6 + 10.)]
    class Flow:
        column = SimpleNamespace(sample=lambda *args, **kw: {'pressure_pa': 1e6 + 5.})
        def _step(self, state, dt, *, _feed_pressure_observer):
            _feed_pressure_observer(violations.pop(0))
        def advance(self, initial, dt, *, pressure_tolerance_pa, _stepper):
            _stepper(initial, dt)  # Coarse trial, potentially discarded.
            _stepper(initial, dt / 2)
            if later_failure:
                raise HydraulicConvergenceError('No valid endpoint')
            return {'state': {}}
    with pytest.raises(ReceiverFeedPressureError) as caught:
        advance_receiver_locally(Flow(), object(), .01, pressure_height_m=1.)
    assert caught.value.receiver_pressure_pa == 1e6 + 50.
    assert getattr(caught.value, 'endpoint_pressure_pa', None) == (
        None if later_failure else 1e6 + 5.)


def test_real_low_pressure_feed_is_rejected_with_or_without_complete_probe():
    from test_surface_stratified_pr_flow import setup

    from surface.component_energy import ComponentEnergyStream
    from surface.stratified_pr_flow import StratifiedFeed, StratifiedPRFlow
    from surface.vessel_wall import VesselWall

    column, stocks = setup()
    stream = ComponentEnergyStream.from_pt(column.fluid, [1e-9, 2e-9], 990000., 320.,
                                           elevation_m=column.elevation_m + 1.)
    flow = StratifiedPRFlow(column, VesselWall(1000., 0., 0.),
                            feeds=[StratifiedFeed(1., stream)])
    initial = flow.initialize(stocks, stocks.temperature_k)
    for height in (None, 1.):
        with pytest.raises(ReceiverFeedPressureError) as caught:
            advance_receiver_locally(flow, initial, .001, pressure_height_m=height)
        if height is not None:
            assert math.isfinite(caught.value.endpoint_pressure_pa)
        assert initial.time_s == 0.


def test_invalid_probe_does_not_retry_a_failed_thermodynamic_step():
    from surface.liquid import ConservationError

    class Flow:
        attempts = 0
        def _step(self, state, dt, *, _feed_pressure_observer):
            self.attempts += 1
            _feed_pressure_observer(ReceiverFeedPressureError(1e6, 1e6 + 50.))
            raise ConservationError('Infeasible feed also fails entropy')
        def advance(self, initial, dt, *, pressure_tolerance_pa, _stepper):
            for _ in range(5):
                try:
                    _stepper(initial, dt)
                except ConservationError:
                    dt *= .5
    flow = Flow()
    with pytest.raises(ReceiverFeedPressureError):
        advance_receiver_locally(flow, object(), .01, pressure_height_m=1.)
    assert flow.attempts == 1
