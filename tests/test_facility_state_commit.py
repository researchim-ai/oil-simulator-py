"""Commit an independently solved capacity endpoint without replaying Newton."""
import copy
import pickle
from dataclasses import replace

import pytest
import torch
from test_fim_facility_coupling import build, masses
from test_fim_facility_forecast import config
from run_fim_blackoil_surface import run_case
from check_fim_facility_forecast import audit


def restored(limited=True):
    c = build(limited)
    c.control = replace(c.control, commit_method='state_restore')
    return c


def assert_physical_equal(a, b):
    for name in ('pressure', 's_w', 's_g', 's_o', 'rs', 'rv'):
        torch.testing.assert_close(getattr(a.solver.fluid, name), getattr(b.solver.fluid, name), rtol=0, atol=0)
    for name in ('x', 'sat', 'ug', 'so_max'):
        torch.testing.assert_close(getattr(a.solver, name), getattr(b.solver, name), rtol=0, atol=0)
    assert a.solver.wells.surface_report == b.solver.wells.surface_report
    assert a.report['facility'] == b.report['facility']
    assert a.report['material_account'] == b.report['material_account']
    assert a.report['fraction'] == b.report['fraction']
    for wa, wb in zip(a.solver.wells.wells,b.solver.wells.wells):
        for name in ('cum_surface', 'cum_prod', 'cum_inj', 'last_surface', 'control_value'):
            assert getattr(wa,name,None) == getattr(wb,name,None)
    for name in ('p_bh', '_Q'):
        torch.testing.assert_close(getattr(a.solver.wells,name),getattr(b.solver.wells,name),rtol=0,atol=0)
    for name in ('last_newton_iters','last_max_dp','last_max_dsw'):
        assert getattr(a.solver,name) == getattr(b.solver,name)


def test_three_steps_events_and_restart_match_fresh_replay_exactly():
    reference, candidate = build(), restored()
    for i, controls in enumerate(({}, {'gas_valve_max_opening': .8}, {})):
        for c in (reference,candidate):
            c.controls = copy.deepcopy(controls)
            assert c.step(3600., current_time=i*3600.)
        assert_physical_equal(reference,candidate)
        assert candidate.report['evaluations'] == reference.report['evaluations']-1
        commit=candidate.report['capacity_commit']
        row=candidate.report['trace'][commit['solved_trace_index']]
        assert row['feasible'] and row['fraction']==candidate.report['fraction']
        reference,candidate=[pickle.loads(pickle.dumps(c)) for c in (reference,candidate)]


def test_full_capacity_has_no_saved_state_and_no_extra_trial():
    reference, candidate = build(False), restored(False)
    assert reference.step(3600.) and candidate.step(3600.)
    assert_physical_equal(reference,candidate)
    assert candidate.report['evaluations']==reference.report['evaluations']==1
    assert 'capacity_commit' not in candidate.report


def test_post_restore_failure_rolls_back_all_material_and_can_retry(monkeypatch):
    c=restored()
    assert c.step(3600.)
    before=masses(c.solver).clone();report=copy.deepcopy(c.report)
    original=c._material_account
    def fail(*args):
        raise ValueError('invalid material ledger')
    monkeypatch.setattr(c,'_material_account',fail)
    with pytest.raises(ValueError,match='invalid material ledger'):
        c.step(3600.,current_time=3600.)
    torch.testing.assert_close(masses(c.solver),before,rtol=0,atol=0)
    assert c.report==report
    assert [w.control_value for w in c.solver.wells.wells]==[100,100]
    monkeypatch.setattr(c,'_material_account',original)
    assert c.step(3600.,current_time=3600.)


@pytest.fixture(scope='module')
def trajectory():
    c=config();c['simulation']['coupled_facility']['control']['commit_method']='state_restore'
    c['simulation']['coupled_facility']['temporal_control']={}
    events=[{'time_days':.02,'controls':{'gas_valve_max_opening':.8}}]
    r=run_case(c,1e6,.04,.04,facility_schedule=events)
    return c,events,r


def test_independent_audit_accepts_real_half_stages_and_rejects_wrong_saved_trial(trajectory):
    c,events,r=trajectory
    assert audit(c,r,expected_schedule=events,expected_days=.04)['accepted']
    bad=copy.deepcopy(r)
    commit=bad['accepted_substeps'][0]['facility_coupling']['capacity_commit']
    commit['solved_trace_index']=0
    with pytest.raises(ValueError,match='matching solved feasible trial'):
        audit(c,bad,expected_schedule=events,expected_days=.04)


def test_invalid_commit_method_is_rejected():
    c=build()
    with pytest.raises(ValueError,match='commit_method'):
        replace(c.control,commit_method='scale_flows')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_actual_cuda_state_commit_matches_replay_and_counts_one_less_solve():
    reference, candidate = build(device='cuda'), build(device='cuda')
    candidate.control=replace(candidate.control,commit_method='state_restore')
    assert reference.step(3600.) and candidate.step(3600.)
    for name in ('pressure','s_w','s_g','rs','rv'):
        torch.testing.assert_close(getattr(reference.solver.fluid,name),
                                   getattr(candidate.solver.fluid,name),rtol=1e-10,atol=1e-5 if name=='pressure' else 1e-10)
    for name in ('p_bh','_Q'):
        torch.testing.assert_close(getattr(reference.solver.wells,name),
                                   getattr(candidate.solver.wells,name),rtol=1e-10,atol=1e-5 if name=='p_bh' else 1e-10)
    assert reference.report['evaluations']==candidate.report['evaluations']+1
    assert reference.report['fraction']==pytest.approx(candidate.report['fraction'],abs=1e-12)
