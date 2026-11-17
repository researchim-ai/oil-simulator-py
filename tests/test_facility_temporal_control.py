"""Whole-system temporal control: actual fine stages, ledgers and rollback."""
import copy
import json

import pytest
import torch
from check_fim_facility_forecast import audit
from run_fim_blackoil_surface import run_case
from test_fim_blackoil_surface import ROOT
from test_fim_facility_coupling import build, masses

from simulator.facility_temporal_control import FacilityTemporalControl


def test_two_actual_half_steps_equal_explicit_reference_and_conserve_mass():
    controlled, reference = build(False), build(False)
    controlled.temporal_control = FacilityTemporalControl()
    old = masses(controlled.solver).clone()
    assert controlled.step(3600., current_time=0.)
    assert reference.step(1800., current_time=0.)
    assert reference.step(1800., current_time=1800.)
    torch.testing.assert_close(controlled.solver.fluid.pressure, reference.solver.fluid.pressure, rtol=0, atol=0)
    stages = controlled.accepted_stages
    assert [x['dt_s'] for x in stages] == [1800., 1800.]
    assert stages[0]['time_days'] < stages[1]['time_days'] == 3600 / 86400
    withdrawn = torch.tensor([x['withdrawn_component_mass_kg'] for x in stages], dtype=torch.float64).sum(0)
    torch.testing.assert_close(old - masses(controlled.solver), withdrawn, atol=.01, rtol=1e-7)
    assert controlled.report['temporal_control']['indicator'] <= 1
    assert controlled.last_attempt['evaluations'] == 3
    assert controlled.solver.last_max_dp > 0


def test_error_rejection_restores_reservoir_wells_report_and_requests():
    c = build(False)
    assert c.step(3600., current_time=0.)
    s = c.solver
    pressure, sw, pb = [t.clone() for t in (s.fluid.pressure, s.fluid.s_w, s.wells.p_bh)]
    previous = copy.deepcopy(c.report)
    caps = [w.control_value for w in s.wells.wells]
    metrics = (s.last_newton_iters, s.last_max_dp, s.last_max_dsw)
    c.temporal_control = FacilityTemporalControl(pressure_tolerance_pa=1e-10)
    assert not c.step(3600., current_time=3600.)
    assert 'temporal defect' in c.last_attempt['reason']
    for actual, before in ((s.fluid.pressure, pressure), (s.fluid.s_w, sw), (s.wells.p_bh, pb)):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
    assert c.report == previous and not c.accepted_stages
    assert [w.control_value for w in s.wells.wells] == caps
    assert metrics == (s.last_newton_iters, s.last_max_dp, s.last_max_dsw)
    c.temporal_control = FacilityTemporalControl()
    assert c.step(3600., current_time=3600.)


def test_half_step_exception_rolls_back_without_publishing_stages(monkeypatch):
    c = build(False)
    c.temporal_control = FacilityTemporalControl()
    p = c.solver.fluid.pressure.clone()
    original, calls = c._step_once, []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise ValueError('invalid property table')
        return original(*args, **kwargs)
    monkeypatch.setattr(c, '_step_once', fail_second)
    with pytest.raises(ValueError, match='invalid property table'):
        c.step(3600.)
    torch.testing.assert_close(c.solver.fluid.pressure, p, rtol=0, atol=0)
    assert c.report is None and not c.accepted_stages
    assert c.last_attempt['success'] is False
    assert c.last_attempt['reason'] == 'invalid property table'
    assert c.temporal_work['capacity_evaluations'] == 2


@pytest.mark.parametrize('scale', ['increment', 'cumulative'])
def test_forecast_emits_actual_half_stage_ledger_and_hits_operating_event(scale):
    c = json.loads((ROOT / 'configs/fim_blackoil_facility_coupled.json').read_text())
    c['fluid']['pvt_path'] = str(ROOT / c['fluid']['pvt_path'])
    spec = c['simulation']['coupled_facility']
    spec.pop('time_control', None)
    spec['temporal_control'] = {'startup_step_days': .05, 'withdrawal_error_scale': scale,
                                'well_bhp_tolerance_pa': 100.}
    events = [{'time_days': .05, 'controls': {'gas_valve_max_opening': .8}}]
    r = run_case(c, 1e6, .1, .1, facility_schedule=events)
    assert audit(c, r, expected_schedule=events, expected_days=.1)['accepted']
    stages = r['accepted_substeps']
    assert len(stages) >= 4
    assert any(x['time_days'] == .05 for x in stages)
    for step in stages:
        assert step['dt_s'] == pytest.approx(step['facility_coupling']['dt_s'])
    assert sum(x['dt_s'] for x in stages) == pytest.approx(.1 * 86400)


@pytest.mark.parametrize('kwargs', [{'safety_factor': 1.}, {'pressure_tolerance_pa': float('nan')},
                                    {'saturation_tolerance': False}, {'maximum_growth': .5}])
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FacilityTemporalControl(**kwargs)


def test_step_proposal_uses_quadratic_local_defect_and_event_restart():
    c = build(False)
    c.temporal_control = FacilityTemporalControl(startup_step_days=.1)
    assert c.limit_initial_time_step_s(86400.) == 8640.
    c.report = {'operator_controls': {}, 'dt_s': 50.,
                'temporal_control': {'macro_dt_s': 100., 'indicator': .25}}
    assert c.limit_next_time_step_s(1000.) == pytest.approx(180.)
    c.controls = {'gas_valve_max_opening': .8}
    assert c.limit_next_time_step_s(86400.) == 8640.


def test_independent_audit_rejects_temporal_metadata_and_missing_half_stage():
    c = json.loads((ROOT / 'configs/fim_blackoil_facility_coupled.json').read_text())
    c['fluid']['pvt_path'] = str(ROOT / c['fluid']['pvt_path'])
    c['simulation']['coupled_facility']['temporal_control'] = {}
    r = run_case(c, 1e6, .02, .02)
    corrupted = copy.deepcopy(r)
    corrupted['accepted_substeps'][1]['facility_coupling']['temporal_control']['errors']['pressure'] = 2.
    with pytest.raises(ValueError, match='temporal defect'):
        audit(c, corrupted, expected_days=.02)
    corrupted = copy.deepcopy(r)
    corrupted['accepted_substeps'][1]['facility_coupling']['temporal_control']['total_capacity_evaluations'] = 0
    with pytest.raises(ValueError, match='coarse-solve work'):
        audit(c, corrupted, expected_days=.02)
    corrupted = copy.deepcopy(r)
    corrupted['accepted_substeps'].pop(0)
    with pytest.raises(ValueError):
        audit(c, corrupted, expected_days=.02)


def test_duplicate_temporal_policies_are_rejected_before_solving():
    c = build(False)
    from simulator.facility_coupling import FacilityCoupledFIM
    with pytest.raises(ValueError, match='Choose temporal_control'):
        FacilityCoupledFIM(c.solver, c.facility, time_control={}, temporal_control={})


def test_rejected_proposal_uses_measured_defect_without_weakening_tolerance():
    c = build(False)
    c.temporal_control = FacilityTemporalControl()
    c.last_attempt = {'success': False, 'temporal_errors': {'pressure': 100., 's_w': 4.}}
    assert c.limit_rejected_time_step_s(500., failed_dt_s=1000.) == 100.
    c.last_attempt = {'success': False, 'reason': 'Newton failed'}
    assert c.limit_rejected_time_step_s(500., failed_dt_s=1000.) == 500.


@pytest.mark.parametrize('bhp_tolerance,scale', [(None, 'increment'), (100., 'increment'),
                                               (100., 'cumulative')])
def test_restart_preserves_temporal_history_and_exact_next_state(bhp_tolerance, scale):
    import pickle
    c = build(False)
    c.temporal_control = FacilityTemporalControl(well_bhp_tolerance_pa=bhp_tolerance,
                                                withdrawal_error_scale=scale)
    assert c.step(1800., current_time=0.)
    resumed = pickle.loads(pickle.dumps(c))
    assert resumed.limit_next_time_step_s(3600.) == c.limit_next_time_step_s(3600.)
    assert c.step(1800., current_time=1800.)
    assert resumed.step(1800., current_time=1800.)
    torch.testing.assert_close(c.solver.fluid.pressure, resumed.solver.fluid.pressure, rtol=0, atol=0)
    assert c.report == resumed.report
    assert c.temporal_work == resumed.temporal_work


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_defect_reduction_matches_known_values_and_preserves_nonfinite_failure(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from surface import HydraulicConvergenceError
    policy = FacilityTemporalControl(pressure_tolerance_pa=2., saturation_tolerance=.01,
                                     solution_ratio_tolerance=.1,
                                     withdrawal_relative_tolerance=.1,
                                     withdrawal_absolute_tolerance_kg=1.)
    old = {k: torch.zeros(3, dtype=torch.float64, device=device) for k in ('pressure','s_w','s_g','rs','rv')}
    new = {k: v.clone() for k, v in old.items()}
    for k, value in [('pressure',1.),('s_w',.002),('s_g',.003),('rs',.04),('rv',.05)]:
        new[k][1] = value
    errors = policy.defect(old, new, torch.tensor([1.,2.,3.]), torch.tensor([1.,2.,4.]))
    for k, value in [('pressure',.5),('s_w',.2),('s_g',.3),('rs',.4),('rv',.5)]:
        assert errors[k] == pytest.approx(value)
    assert errors['withdrawal'] == pytest.approx(1 / 1.4)
    new['s_g'][0] = float('nan')
    with pytest.raises(HydraulicConvergenceError, match='Nonfinite temporal defect'):
        policy.defect(old, new, torch.ones(3), torch.ones(3))


@pytest.mark.parametrize('value', [0., -1., True, float('nan'), float('inf')])
def test_invalid_bhp_error_budget_is_rejected(value):
    with pytest.raises(ValueError):
        FacilityTemporalControl(well_bhp_tolerance_pa=value)


def test_bhp_defect_is_opt_in_and_fails_closed_on_missing_or_nonfinite_wells():
    from surface import HydraulicConvergenceError
    assert FacilityTemporalControl().well_defect({}, {}) == {}
    p = FacilityTemporalControl(well_bhp_tolerance_pa=2.)
    assert p.well_defect({'P': 100.}, {'P': 101.}) == {'well_bhp': .5}
    for a, b in [({'P': 1.}, {}), ({'P': float('nan')}, {'P': 1.}),
                 ({'P': 1.}, {'P': float('inf')})]:
        with pytest.raises(HydraulicConvergenceError):
            p.well_defect(a, b)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_actual_bhp_defect_alone_can_reject_and_restore_the_whole_step(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    accepted = build(False, device=device)
    accepted.temporal_control = FacilityTemporalControl(well_bhp_tolerance_pa=1e8)
    assert accepted.step(3600.)
    meta = accepted.report['temporal_control']
    delta = meta['errors']['well_bhp'] * 1e8
    assert delta > 0 and 'coarse_well_bhp_pa' in meta
    rejected = build(False, device=device)
    rejected.temporal_control = FacilityTemporalControl(well_bhp_tolerance_pa=delta / 2)
    before = rejected.solver.fluid.pressure.clone()
    assert not rejected.step(3600.)
    assert rejected.last_attempt['temporal_errors']['well_bhp'] == pytest.approx(2.)
    assert all(v <= 1 for k, v in rejected.last_attempt['temporal_errors'].items() if k != 'well_bhp')
    torch.testing.assert_close(before, rejected.solver.fluid.pressure, rtol=0, atol=0)
    assert not rejected.accepted_stages and rejected.report is None


@pytest.mark.parametrize('mutation', ['missing_error', 'missing_evidence', 'changed_evidence'])
def test_independent_audit_requires_measured_bhp_defect(mutation):
    c = json.loads((ROOT / 'configs/fim_blackoil_facility_coupled.json').read_text())
    c['fluid']['pvt_path'] = str(ROOT / c['fluid']['pvt_path'])
    c['simulation']['coupled_facility']['temporal_control'] = {'well_bhp_tolerance_pa': 100.}
    r = run_case(c, 1e6, .02, .02)
    assert audit(c, r, expected_days=.02)['accepted']
    r = copy.deepcopy(r)
    meta = r['accepted_substeps'][1]['facility_coupling']['temporal_control']
    if mutation == 'missing_error':
        meta['errors'].pop('well_bhp')
    elif mutation == 'missing_evidence':
        meta.pop('coarse_well_bhp_pa')
    else:
        fine = r['accepted_substeps'][1]['surface']['wells']
        key = max(meta['coarse_well_bhp_pa'],
                  key=lambda k: abs(meta['coarse_well_bhp_pa'][k] - fine[k]['bhp_pa']))
        direction = 1 if meta['coarse_well_bhp_pa'][key] >= fine[key]['bhp_pa'] else -1
        meta['coarse_well_bhp_pa'][key] += direction * 10.
    with pytest.raises(ValueError, match='BHP|mismatch'):
        audit(c, r, expected_days=.02)


def test_cumulative_withdrawal_scale_uses_state_totals_not_a_tiny_substep():
    p = FacilityTemporalControl(withdrawal_error_scale='cumulative',
                               withdrawal_absolute_tolerance_kg=1., withdrawal_relative_tolerance=.1)
    fields = {'pressure': torch.zeros(1, dtype=torch.float64)}
    a,b = torch.tensor([1.,2.,3.]),torch.tensor([1.,2.,4.])
    x,y = torch.tensor([101.,102.,103.]),torch.tensor([101.,102.,104.])
    e = p.defect(fields,fields,a,b,coarse_total_mass=x,fine_total_mass=y)
    assert e['withdrawal'] == pytest.approx(1/11.4)
    from surface import HydraulicConvergenceError
    with pytest.raises(HydraulicConvergenceError,match='cumulative'):
        p.defect(fields,fields,a,b)
    with pytest.raises(ValueError,match='withdrawal_error_scale'):
        FacilityTemporalControl(withdrawal_error_scale='arbitrary')


@pytest.mark.parametrize('tamper',[False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_cumulative_control_audit_and_corrupted_coarse_evidence(tamper, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    c = json.loads((ROOT / 'configs/fim_blackoil_facility_coupled.json').read_text())
    c['fluid']['pvt_path'] = str(ROOT / c['fluid']['pvt_path'])
    c['simulation']['coupled_facility'].pop('time_control', None)
    c['simulation']['coupled_facility']['temporal_control'] = {
        'withdrawal_error_scale':'cumulative','well_bhp_tolerance_pa':100.,'startup_step_days':.01}
    r = run_case(c,1e6,.03,.01,device)
    assert audit(c,r,expected_days=.03)['accepted']
    if tamper:
        m=r['accepted_substeps'][-1]['facility_coupling']['temporal_control']
        m['coarse_withdrawn_total_kg'][0] += 1.
        with pytest.raises(ValueError,match='mismatch'):
            audit(c,r,expected_days=.03)


@pytest.mark.parametrize('mode', [None, True, 'invalid'])
def test_invalid_withdrawal_scaling_is_rejected(mode):
    with pytest.raises(ValueError, match='withdrawal_error_scale'):
        FacilityTemporalControl(withdrawal_error_scale=mode)


def test_cumulative_withdrawal_scale_rejects_nonfinite_or_missing_totals():
    from surface import HydraulicConvergenceError
    p = FacilityTemporalControl(withdrawal_error_scale='cumulative')
    fields = {'pressure':torch.zeros(1)}
    for totals in [(None,None),(torch.ones(2),torch.ones(3)),
                   (torch.tensor([1.,float('nan'),1.]),torch.ones(3))]:
        with pytest.raises(HydraulicConvergenceError,match='cumulative'):
            p.defect(fields,fields,torch.ones(3),torch.ones(3),
                     coarse_total_mass=totals[0],fine_total_mass=totals[1])
