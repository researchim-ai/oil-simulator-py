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
