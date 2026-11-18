"""Local continuation keeps physical endpoints and requests broad fallback."""
from dataclasses import replace

import pytest

from simulator.facility_coupling import CoupledFacilityControl
from surface.capacity_bracket import local_capacity_bracket


def test_local_bracket_contains_root_and_both_endpoints_were_evaluated():
    points=[]
    def evaluate(x):
        points.append(x);v=x-.003
        return {'fraction':x,'feasible':v<=0,'constraint_margins':{'gas':v},'normalized_margin':v}
    pair=local_capacity_bracket(evaluate,.00301,{'gas':1.},.0001,1e-7)
    assert pair[0]['fraction']<=.003<pair[1]['fraction']
    assert pair[0]['fraction'] in points and pair[1]['fraction'] in points
    assert len(points)<=4


@pytest.mark.parametrize('margins,slopes', [({},{}),({'new':1.},{'gas':1.}),({'gas':None},{'gas':1.}),({'gas':1.},{'gas':0.})])
def test_unusable_history_requests_full_scan(margins,slopes):
    def evaluate(x):
        return {'fraction':x,'feasible':False,'constraint_margins':margins}
    assert local_capacity_bracket(evaluate,.003,slopes,.0001,1e-7) is None


def test_remote_root_does_not_escape_local_window():
    points=[]
    def evaluate(x):
        points.append(x)
        return {'fraction':x,'feasible':False,'constraint_margins':{'gas':x-.001}}
    assert local_capacity_bracket(evaluate,.003,{'gas':1.},.0001,1e-7) is None
    assert points==[.003]


@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_scan_period_requires_positive_integer(value):
    with pytest.raises(ValueError,match='continuation_scan_interval'):
        replace(CoupledFacilityControl(),continuation_scan_interval=value)


def test_real_periodic_scans_events_restart_and_rollback():
    import copy
    import pickle

    from test_fim_facility_coupling import build

    c=build();c.control=replace(c.control,search_method='continuation_safeguarded',
        continuation_scan_interval=3,fraction_tolerance=1e-7,commit_method='state_restore')
    used=[]
    for i in range(5):
        assert c.step(3600.,current_time=i*3600.)
        evidence=c.report['capacity_continuation'];used.append(evidence['used'])
        assert 0<=evidence['age']<3
        assert c.report['capacity_bracket']['width']<=1e-7
        c=pickle.loads(pickle.dumps(c))
    assert any(used) and not used[0] and not used[3]
    c.controls={'gas_valve_max_opening':.8}
    assert c.step(3600.,current_time=18000.)
    assert not c.report['capacity_continuation']['used']
    saved=copy.deepcopy(c.report)
    def fail(*args):raise ValueError('ledger')
    c._material_account=fail
    with pytest.raises(ValueError,match='ledger'):c.step(3600.,current_time=21600.)
    assert c.report==saved


def test_audit_rejects_missing_full_scan_or_stale_continuation():
    import copy

    from simulator.capacity_continuation_audit import audit_capacity_continuation

    control={'search_method':'continuation_safeguarded','continuation_scan_interval':3,
             'min_fraction':.01,'scan_points':3}
    full={'search_method':'continuation_safeguarded','fraction':.2,
          'operator_controls':{},'requested_rate_caps_m3_day':[100],
          'trace':[{'fraction':x} for x in (1.,.01,.505)],
          'capacity_continuation':{'used':False,'age':0,'scan_interval':3,'reason':'full_scan'}}
    audit_capacity_continuation(full,control)
    bad=copy.deepcopy(full);bad['trace'].pop()
    with pytest.raises(ValueError,match='domain samples'):audit_capacity_continuation(bad,control)
    local=copy.deepcopy(full);local['capacity_continuation'].update(used=True,age=1,reason='local_bracket')
    audit_capacity_continuation(local,control,full)
    local['operator_controls']={'gas_valve_max_opening':.8}
    with pytest.raises(ValueError,match='stale controls'):audit_capacity_continuation(local,control,full)
    local['operator_controls']={};local['capacity_continuation']['age']=3
    with pytest.raises(ValueError,match='invalid continuation'):audit_capacity_continuation(local,control,full)


def narrow_report():
    low, high = .003, .0030005
    return {'fraction': low,
            'capacity_bracket': {'feasible_fraction': low, 'infeasible_fraction': high},
            'trace': [
                {'fraction': low, 'feasible': True, 'constraint_margins': {'gas': -.0003}},
                {'fraction': high, 'feasible': False, 'constraint_margins': {'gas': .00035}}]}


def test_resolved_narrow_bracket_tracks_next_operating_point():
    from surface.capacity_bracket import capacity_history_slopes, continuation_slopes

    report = narrow_report()
    assert not capacity_history_slopes(report['trace'], report['fraction'], 1e-6)
    slopes = continuation_slopes(report, 1e-6)
    evaluated = []
    root = .003002
    def evaluate(x):
        evaluated.append(x)
        margin = 1300*(x-root)
        return {'fraction': x, 'feasible': margin <= 0,
                'constraint_margins': {'gas': margin}}
    pair = local_capacity_bracket(evaluate, report['fraction'], slopes, .0001, 1e-6)
    assert pair is not None
    assert pair[0]['fraction'] <= root < pair[1]['fraction']
    assert all(row['fraction'] in evaluated for row in pair)


@pytest.mark.parametrize('case', ['missing', 'certificate', 'wrong_feasibility',
                                  'roundoff_width', 'roundoff_signal', 'negative', 'nan'])
def test_narrow_continuation_rejects_unresolved_evidence(case):
    import math

    from surface.capacity_bracket import continuation_slopes

    report = narrow_report()
    high = report['trace'][1]
    if case == 'missing':
        report['trace'].pop()
    elif case == 'certificate':
        high['certified_from_trace_index'] = 0
    elif case == 'wrong_feasibility':
        high['feasible'] = True
    elif case == 'roundoff_width':
        high['fraction'] = math.nextafter(report['fraction'], 1.)
        report['capacity_bracket']['infeasible_fraction'] = high['fraction']
    elif case == 'roundoff_signal':
        report['trace'][0]['constraint_margins']['gas'] = -1e-16
        high['constraint_margins']['gas'] = 1e-16
    elif case == 'negative':
        high['constraint_margins']['gas'] = -.001
    else:
        high['constraint_margins']['gas'] = float('nan')
    assert continuation_slopes(report, 1e-6) == {}


def test_continuation_preserves_existing_wide_history_proposal():
    from surface.capacity_bracket import capacity_history_slopes, continuation_slopes

    report = narrow_report()
    report['trace'].append({'fraction': .00302, 'feasible': False,
                            'constraint_margins': {'gas': .018}})
    expected = capacity_history_slopes(report['trace'], report['fraction'], 1e-6)
    assert expected
    assert continuation_slopes(report, 1e-6) == expected
