"""Exact initial-state reuse; nonlinear iterates and later steps stay independent."""
import copy
import pickle
from dataclasses import replace

import pytest
import torch
from test_facility_state_commit import assert_physical_equal
from test_fim_facility_coupling import build, masses

from simulator.fim_initial_cell_cache import InitialCellStateCache, audit_initial_cell_cache


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('commit', ['replay', 'state_restore'])
def test_physical_states_events_and_restart_are_exact(device, commit):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    a, b = build(device=device), build(device=device)
    for c in (a, b):
        c.control = replace(c.control, commit_method=commit, reuse_inactive_caps=True)
    b.control = replace(b.control, reuse_initial_cell_state=True)
    for time, dt, controls in ((0., 60., {}), (60., 90., {'gas_valve_max_opening': .8}),
                               (150., 120., {})):
        for c in (a, b):
            c.controls = copy.deepcopy(controls)
            assert c.step(dt, current_time=time)
            assert not hasattr(c.solver, '_capacity_initial_cell_cache')
        assert_physical_equal(a, b)
        assert a.report['trace'] == b.report['trace']
        stats = b.report['initial_cell_state_cache']
        assert stats['evaluations'] == 1
        assert stats['hits'] == b.report['reservoir_evaluations'] - 1
        assert 0 < stats['stored_bytes'] <= stats['budget_bytes']
        a, b = [pickle.loads(pickle.dumps(c)) for c in (a, b)]


def test_cache_budget_bypasses_without_changing_the_solution():
    a, b = build(), build()
    b.control = replace(b.control, reuse_initial_cell_state=True, initial_cell_state_cache_bytes=1)
    assert a.step(60.) and b.step(60.)
    assert_physical_equal(a, b)
    stats = b.report['initial_cell_state_cache']
    assert stats['bypassed'] and stats['stored_bytes'] == stats['hits'] == 0
    assert stats['evaluations'] == b.report['reservoir_evaluations']


def test_failure_releases_cache_and_restores_physical_state(monkeypatch):
    c = build()
    c.control = replace(c.control, reuse_initial_cell_state=True)
    before = masses(c.solver).clone()
    original = c._material_account
    def fail(*args):
        raise ValueError('injected commit failure')
    monkeypatch.setattr(c, '_material_account', fail)
    with pytest.raises(ValueError, match='injected'):
        c.step(60.)
    assert not hasattr(c.solver, '_capacity_initial_cell_cache')
    torch.testing.assert_close(masses(c.solver), before, rtol=0, atol=0)
    monkeypatch.setattr(c, '_material_account', original)
    assert c.step(60.)
    assert c.report['initial_cell_state_cache']['evaluations'] == 1


def test_cached_values_are_isolated_from_graph_buffer_and_caller_mutation():
    class Solver:
        pass
    s = Solver()
    cache = InitialCellStateCache(s, 60., 0.)
    values = (torch.tensor([1.]), torch.tensor([2.]))
    cache.evaluate(s, 60., 0., lambda: values)
    values[0].zero_()  # Simulate later graph replay overwriting its output.
    reused = cache.evaluate(s, 60., 0., lambda: pytest.fail('unexpected recompute'))
    assert reused[0].item() == 1.
    reused[0].zero_()
    assert cache.evaluate(s, 60., 0., None)[0].item() == 1.
    for owner, dt, time in ((Solver(), 60., 0.), (s, 61., 0.), (s, 60., 1.)):
        with pytest.raises(RuntimeError, match='scope'):
            cache.evaluate(owner, dt, time, None)
    with pytest.raises(ValueError), cache.installed():
        raise ValueError('interrupt')
    assert not hasattr(s, '_capacity_initial_cell_cache')


@pytest.mark.parametrize('flag', [1, 'true', None])
def test_flag_is_boolean(flag):
    with pytest.raises(ValueError, match='boolean'):
        replace(build().control, reuse_initial_cell_state=flag)


@pytest.mark.parametrize('budget', [0, -1, True, 2.5])
def test_budget_requires_positive_integer(budget):
    with pytest.raises(ValueError, match='positive integer'):
        replace(build().control, initial_cell_state_cache_bytes=budget)


def test_drsdt_state_and_newton_results_remain_exact():
    a, b = build(), build()
    for c in (a, b):
        c.solver.drsdt = .25 / 86400
    b.control = replace(b.control, reuse_initial_cell_state=True)
    for time in (0., 60.):
        assert a.step(60., current_time=time) and b.step(60., current_time=time)
        assert_physical_equal(a, b)


@pytest.mark.parametrize('change', ['physical', 'numerical', 'flag'])
def test_exact_checker_rejects_changes_to_other_policies(monkeypatch, change):
    import check_facility_initial_cell_reuse as check
    a = {'config': {'simulation': {'coupled_facility': {'control': {'min_fraction': .001}}},
                    'physical_value': 1.}}
    b = copy.deepcopy(a)
    control = b['config']['simulation']['coupled_facility']['control']
    control['reuse_initial_cell_state'] = True
    if change == 'physical':
        b['config']['physical_value'] = 2.
    elif change == 'numerical':
        control['min_fraction'] = .01
    else:
        control['reuse_initial_cell_state'] = 1
    monkeypatch.setattr(check, 'completed', lambda directory, days: ({}, a if directory == 'a' else b, None))
    with pytest.raises(ValueError, match='Only initial cell state reuse'):
        check.compare('a', 'b', 30.)


@pytest.mark.parametrize('mutation', ['hits', 'evaluations', 'stored_bytes', 'budget_bytes', 'bypassed', 'missing'])
def test_audit_rejects_corrupted_work_and_memory_claims(mutation):
    c = {'evaluations': 4, 'reservoir_evaluations': 4,
         'initial_cell_state_cache': {'evaluations': 1, 'hits': 3, 'stored_bytes': 80,
                                     'budget_bytes': 100, 'bypassed': False}}
    control = {'reuse_initial_cell_state': True, 'initial_cell_state_cache_bytes': 100}
    assert audit_initial_cell_cache(c, control) == (1, 3)
    if mutation == 'missing':
        del c['initial_cell_state_cache']
    elif mutation == 'bypassed':
        c['initial_cell_state_cache']['bypassed'] = True
    else:
        c['initial_cell_state_cache'][mutation] += 1
        if mutation == 'stored_bytes':
            c['initial_cell_state_cache'][mutation] = 101
    with pytest.raises(ValueError):
        audit_initial_cell_cache(c, control)
