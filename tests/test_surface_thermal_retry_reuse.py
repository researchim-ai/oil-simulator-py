"""Reuse a rejected attempt's valid half without changing the accepted trajectory."""
from test_surface_stratified_pr_thermal import model


def test_retry_reuses_exact_half_and_preserves_all_ledgers():
    results, calls = [], []
    for enabled in (False, True):
        flow, state = model('cpu', temperature_tolerance_k=1e-5)
        flow._reuse_rejected_half_step = enabled
        count = []
        def measured(old, dt, count=count, flow=flow):
            count.append(dt)
            return flow._step(old, dt)
        result = flow.advance(state, .1, _stepper=measured)
        results.append(result)
        calls.append(len(count))
    assert results[0] == results[1]
    rejected = results[0]['attempts'] - len(results[0]['steps']) // 2
    assert rejected > 0
    assert calls[0] - calls[1] == rejected


def test_physical_failure_does_not_cache_an_unqualified_attempt():
    from surface.liquid import ConservationError

    flow, state = model('cpu')
    calls = []
    def measured(old, dt):
        calls.append((old.time_s, dt))
        if len(calls) == 3:
            raise ConservationError('Injected fine-stage physical failure')
        return flow._step(old, dt)
    result = flow.advance(state, .1, _stepper=measured)
    assert calls[3] == (0., .05)  # Fresh coarse solve after physical rejection.
    assert result['state']['time_s'] == .1
    assert abs(result['energy_residual_j']) < 1e-6
    assert state.time_s == 0.
