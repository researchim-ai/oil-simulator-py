"""Eliminate identical inactive-bound solves, never replace physical solutions."""
import copy
import pickle
from dataclasses import replace

import pytest
import torch
from test_facility_state_commit import assert_physical_equal
from test_fim_facility_coupling import build, masses

from simulator.rate_cap_invariance import (
    DAY,
    audit_inactive_cap_trace,
    certificate_covers,
    make_inactive_cap_certificate,
)


def proof():
    report={'wells':{'P':{'mode':'surface','bhp_pa':2e6,'network_reference_pressure_pa':2e6,
                         'standard_control_rate_m3_s':.00001}}}
    return make_inactive_cap_certificate(report,{'P':('rate',100/DAY)},
                                         {'P':100.},{'P':1e5},1.)


def test_only_deep_inactive_domain_is_certified():
    p=proof()
    assert p and certificate_covers(p,.5)
    assert not certificate_covers(p,p['minimum_fraction']/2)
    assert not certificate_covers(p,1.1)
    bad=copy.deepcopy(p);bad['wells']['P']['mode']='rate'
    with pytest.raises(ValueError):certificate_covers(bad,.5)


def case(reuse, device='cpu'):
    c=build(device=device)
    for w in c.solver.wells.wells:w.control_value*=100
    c.control=replace(c.control,reuse_inactive_caps=reuse,commit_method='state_restore',
                      min_fraction=1e-5,fraction_tolerance=1e-9,max_evaluations=60,
                      search_method='facet_safeguarded')
    return c


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_equations_and_all_physical_state_match_with_fewer_solves(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a,b=case(False,device),case(True,device)
    for i,controls in enumerate(({}, {'gas_valve_max_opening':.8},{})):
        for c in [a,b]:
            c.controls=copy.deepcopy(controls)
            assert c.step(60.,current_time=i*60.)
        assert_physical_equal(a,b)
        assert a.report['evaluations']==b.report['evaluations']
        assert b.report['reservoir_evaluations']<a.report['reservoir_evaluations']
        assert b.report['certified_evaluations']>0
        assert len(b.report['trace'])==b.report['evaluations']
        config_wells = [{'name': w.name, 'control_value': w.control_value, 'bhp_min': w.bhp_min}
                        for w in b.solver.wells.wells]
        assert audit_inactive_cap_trace(b.report, config_wells, True)==b.report['reservoir_evaluations']
        # Cache belongs to one search only; no states/history survive a step or restart.
        a,b=[pickle.loads(pickle.dumps(c)) for c in [a,b]]


def test_failure_after_selection_restores_physical_state_and_requests(monkeypatch):
    c=case(True);before=masses(c.solver).clone();caps=[w.control_value for w in c.solver.wells.wells]
    def fail(*args):raise ValueError('injected ledger failure')
    monkeypatch.setattr(c,'_material_account',fail)
    with pytest.raises(ValueError,match='injected'):c.step(60.)
    torch.testing.assert_close(before,masses(c.solver),rtol=0,atol=0)
    assert c.report is None and [w.control_value for w in c.solver.wells.wells]==caps


@pytest.mark.parametrize('mutation',['feasible','forward','chain','cap','floor','registry','bound','count',
                                     'outside','margin','reason','constraints'])
def test_audit_rejects_invalid_reuse_proofs(mutation):
    p=proof();source={'fraction':1.,'feasible':False,'reason':[], 'normalized_margin':2.,
                     'inactive_cap_certificate':p}
    reused={'fraction':.5,'feasible':False,'reason':[],'normalized_margin':2.,
            'certified_from_trace_index':0}
    c={'trace':[source,reused],'reservoir_evaluations':1,'certified_evaluations':1}
    wells=[{'name':'P','type':'producer','control_type':'rate','control_value':100.,'bhp_min':.1}]
    assert audit_inactive_cap_trace(c,wells,True)==1
    if mutation=='feasible':reused['feasible']=True
    elif mutation=='forward':reused['certified_from_trace_index']=1
    elif mutation=='chain':source['certified_from_trace_index']=0
    elif mutation=='cap':p['wells']['P']['cap_m3_s']*=2
    elif mutation=='floor':p['wells']['P']['floor_pa']*=2
    elif mutation=='registry':p['wells']['other']=p['wells'].pop('P')
    elif mutation=='bound':p['minimum_fraction']*=.5
    elif mutation=='count':c['reservoir_evaluations']=2
    elif mutation=='outside':reused['fraction']=p['minimum_fraction']/2
    elif mutation=='margin':reused['normalized_margin']=3.
    elif mutation=='reason':reused['reason']=['different physical result']
    elif mutation=='constraints':reused['constraint_margins']={'different':3.}
    with pytest.raises(ValueError):audit_inactive_cap_trace(c,wells,True)


def test_all_feasible_trial_never_uses_an_infeasible_shortcut():
    c=build(False)
    c.control=replace(c.control,reuse_inactive_caps=True)
    assert c.step(60.)
    assert c.report['fraction']==1.
    assert c.report['certified_evaluations']==0
    assert c.report['reservoir_evaluations']==c.report['evaluations']


@pytest.mark.parametrize('flag',[1,'true',None])
def test_reuse_flag_requires_boolean(flag):
    with pytest.raises(ValueError,match='boolean'):
        replace(build().control,reuse_inactive_caps=flag)


@pytest.mark.parametrize('change',['physical','numerical','flag'])
def test_optimization_comparison_cannot_hide_other_config_changes(monkeypatch, change):
    import check_facility_cap_reuse as check
    a={'config':{'simulation':{'coupled_facility':{'control':{'fraction_tolerance':1e-9}},
                             'initial_time_step_days':.001},'physical_value':1.}}
    b=copy.deepcopy(a)
    b['config']['simulation']['coupled_facility']['control']['reuse_inactive_caps']=True
    if change=='physical':b['config']['physical_value']=2.
    elif change=='numerical':b['config']['simulation']['initial_time_step_days']=.1
    else:b['config']['simulation']['coupled_facility']['control']['reuse_inactive_caps']=1
    monkeypatch.setattr(check,'completed',lambda directory, days: ({},a if directory=='a' else b,None))
    with pytest.raises(ValueError,match='Only inactive-cap reuse'):
        check.compare('a','b',30.)
