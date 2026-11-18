"""Coupled cap response against whole implicit finite differences, not frozen cells."""
import copy
import pickle
from dataclasses import replace

import pytest
import torch
from test_facility_state_commit import assert_physical_equal
from test_fim_facility_coupling import build, masses

from simulator.facility_tangent import FacilityTangentCapture, plant_margin_tangent
from surface.capacity_bracket import CapacityBracket


@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('mixed', [False, True])
def test_whole_reservoir_well_response_matches_finite_differences(structured, mixed):
    s = build().solver
    s.tol_cnv, s.tol_mb, s.tol_well_rate, s.tol_well_bhp = 1e-10, 1e-12, 1e-9, .001
    if structured:
        s.wells.well_block_backend, s.wells.sensitivity_backend = 'tree_structured', 'tree_operator'
    if mixed:
        s.wells.wells[0].control_value = 1.
    caps = [w.control_value for w in s.wells.wells]
    old = pickle.dumps(s)
    fraction, h = (1., 1e-5) if mixed else (.015, 1e-6)
    for w, cap in zip(s.wells.wells, caps):
        w.control_value = cap * fraction
    capture = FacilityTangentCapture(s, caps)
    with capture.installed():
        assert s.step(60.)
    physical = pickle.dumps(s.fluid)
    stats = s._last_linear_stats
    response = capture.response()
    assert response is not None and s._last_linear_stats is stats
    assert pickle.dumps(s.fluid) == physical
    values = []
    for f in (fraction+h, fraction-h):
        c = pickle.loads(old)
        for w, cap in zip(c.wells.wells, caps):
            w.control_value = cap*f
        assert c.step(60.)
        assert c.wells.step_mode == s.wells.step_mode
        values.append(torch.tensor([row['component_mass_rates_kg_s']
                                   for row in c.wells.surface_report['wells'].values()], dtype=torch.float64))
    fd = (values[0]-values[1])/(2*h)
    torch.testing.assert_close(torch.tensor(response['direction'], dtype=torch.float64), fd,
                               rtol=3e-4, atol=1e-7)
    assert not hasattr(s, '_facility_tangent_capture')


def accelerated(method='sensitivity_safeguarded'):
    c = build()
    c.control = replace(c.control, search_method=method, commit_method='state_restore',
                        fraction_tolerance=1e-10, min_fraction=.0001, max_evaluations=60)
    return c


def test_real_capacity_controls_restart_accuracy_and_fewer_fim_steps():
    a, b = accelerated('facet_safeguarded'), accelerated()
    saved = [0, 0]
    for i, control in enumerate(({}, {'gas_valve_max_opening': .8}, {})):
        for c in (a, b):
            c.controls = copy.deepcopy(control)
            assert c.step(60., current_time=i*60.)
            assert not hasattr(c.solver, '_facility_tangent_capture')
        for name, atol in (('pressure', .1), ('s_w', 1e-8), ('s_g', 1e-8)):
            torch.testing.assert_close(getattr(a.solver.fluid, name), getattr(b.solver.fluid, name),
                                       rtol=0, atol=atol)
        assert abs(a.report['fraction']-b.report['fraction']) < 1e-10
        saved[0] += a.report['reservoir_evaluations']
        saved[1] += b.report['reservoir_evaluations']
        assert b.report['sensitivity_work']['proposals'] > 0
        a, b = [pickle.loads(pickle.dumps(c)) for c in (a, b)]
    assert saved[1] < saved[0]


def test_sparse_budget_fallback_reproduces_facet_search_exactly():
    a, b = accelerated('facet_safeguarded'), accelerated()
    b.control = replace(b.control, sensitivity_sparse_budget_bytes=1)
    assert a.step(60.) and b.step(60.)
    assert_physical_equal(a, b)
    assert a.report['evaluations'] == b.report['evaluations']
    assert b.report['sensitivity_work'] == {'linear_solves': 0, 'plant_evaluations': 0, 'proposals': 0}


def test_failed_auxiliary_linear_solve_does_not_modify_physical_state(monkeypatch):
    c = build(); s = c.solver
    caps = [w.control_value for w in s.wells.wells]
    for w in s.wells.wells:
        w.control_value *= .015
    capture = FacilityTangentCapture(s, caps)
    with capture.installed():
        assert s.step(60.)
    before = masses(s).clone()
    stats = s._last_linear_stats
    monkeypatch.setattr(s, 'solve_linear', lambda *args, **kwargs: None)
    assert capture.response() is None
    assert capture.reason == 'auxiliary_linear_failure' and s._last_linear_stats is stats
    torch.testing.assert_close(before, masses(s), rtol=0, atol=0)


def test_post_selection_failure_rolls_back_state_and_caps(monkeypatch):
    c = accelerated(); before = masses(c.solver).clone()
    caps = [w.control_value for w in c.solver.wells.wells]
    def fail(*args):
        raise ValueError('ledger failure')
    monkeypatch.setattr(c, '_material_account', fail)
    with pytest.raises(ValueError, match='ledger'):
        c.step(60.)
    torch.testing.assert_close(before, masses(c.solver), rtol=0, atol=0)
    assert [w.control_value for w in c.solver.wells.wells] == caps and c.report is None
    assert not hasattr(c.solver, '_facility_tangent_capture')


def test_newton_proposal_requires_same_increasing_violated_constraint():
    b = CapacityBracket(.01, 1., 1e-9, -.9, 2.,
                        low_margins={'gas': -.9, 'power': -1.}, high_margins={'gas': 2., 'power': -1.})
    assert b.tangent_proposal({'gas': 10., 'power': 1e6}) == pytest.approx(.1)
    for slopes in ({}, {'gas': -1.}, {'gas': float('nan')}, {'unrelated': 10.}):
        assert b.tangent_proposal(slopes) is None
    b.slow_updates = 2
    assert b.tangent_proposal({'gas': 10.}) is None


def test_small_sparse_budget_falls_back_to_identical_physical_search():
    a, b = accelerated('facet_safeguarded'), accelerated()
    b.control = replace(b.control, sensitivity_sparse_budget_bytes=1)
    assert a.step(60.) and b.step(60.)
    assert_physical_equal(a, b)
    assert a.report['evaluations'] == b.report['evaluations']
    assert b.report['sensitivity_work'] == {'linear_solves': 0, 'plant_evaluations': 0, 'proposals': 0}
    assert any(t.get('capacity_sensitivity', {}).get('reason') == 'sparse_value_budget' for t in b.report['trace'])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('structured', [False, True])
def test_repeated_steps_controls_and_restart_conserve_material(device, structured):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    a, b = build(device=device), build(device=device)
    if structured:
        for c in (a, b):
            c.solver.wells.well_block_backend = 'tree_structured'
            c.solver.wells.sensitivity_backend = 'tree_operator'
    for c, method in ((a, 'facet_safeguarded'), (b, 'sensitivity_safeguarded')):
        c.control = replace(c.control, search_method=method, commit_method='state_restore',
                            fraction_tolerance=1e-10, min_fraction=.0001, max_evaluations=60,
                            reuse_inactive_caps=True, reuse_initial_cell_state=True)
    for i, controls in enumerate(({}, {'gas_valve_max_opening': .8}, {})):
        for c in (a, b):
            c.controls = copy.deepcopy(controls)
            assert c.step(60., current_time=i*60.)
            assert not hasattr(c.solver, '_facility_tangent_capture')
            assert max(map(abs, c.report['material_account']['end_to_end_residual_kg'].values())) < .01
        for field in ('pressure', 's_w', 's_g', 'rs'):
            torch.testing.assert_close(getattr(a.solver.fluid, field), getattr(b.solver.fluid, field),
                                       rtol=1e-8, atol=1e-5 if field == 'pressure' else 1e-8)
        assert b.report['fraction'] == pytest.approx(a.report['fraction'], abs=1e-10)
        a, b = [pickle.loads(pickle.dumps(c)) for c in (a, b)]


def test_capacity_work_audit_rejects_corrupted_auxiliary_counters():
    from simulator.capacity_sensitivity_audit import audit_capacity_sensitivity
    c = accelerated()
    assert c.step(60.)
    assert audit_capacity_sensitivity(c.report) == (
        c.report['sensitivity_work']['linear_solves'], c.report['sensitivity_work']['plant_evaluations'])
    bad = copy.deepcopy(c.report)
    bad['sensitivity_work']['linear_solves'] += 1
    with pytest.raises(ValueError, match='counters'):
        audit_capacity_sensitivity(bad)
    bad = copy.deepcopy(c.report)
    proposal = next(row for row in bad['trace'] if 'capacity_proposal' in row)
    proposal['capacity_proposal']['low_fraction'] = proposal['fraction']
    with pytest.raises(ValueError, match='bracket'):
        audit_capacity_sensitivity(bad)


def test_plant_response_agrees_with_complete_fim_perturbations():
    from copy import copy as shallow_copy

    from surface import ComponentStream
    from surface.capacity_bracket import facility_constraint_margins
    c = build()
    caps = [w.control_value for w in c.solver.wells.wells]
    old = pickle.dumps(c.solver)
    fraction, h = .015, 1e-5
    for w, cap in zip(c.solver.wells.wells, caps):
        w.control_value = cap*fraction
    capture = FacilityTangentCapture(c.solver, caps)
    with capture.installed():
        assert c.solver.step(60.)
    response = capture.response()
    slopes, evaluations, reason = plant_margin_tangent(c.facility, response, fraction, {}, ())
    assert reason is None and evaluations == 4
    margins = []
    for f in (fraction+h, fraction-h):
        s = pickle.loads(old)
        for w, cap in zip(s.wells.wells, caps):
            w.control_value = cap*f
        assert s.step(60.)
        plant = shallow_copy(c.facility)
        plant.feeds = tuple(ComponentStream.from_mass_rates(*row['component_mass_rates_kg_s'], plant.densities)
                            for row in s.wells.surface_report['wells'].values())
        margins.append(facility_constraint_margins(plant.run()))
    fd = (margins[0]['gas_export.valve']-margins[1]['gas_export.valve'])/(2*h)
    assert slopes['gas_export.valve'] == pytest.approx(fd, rel=3e-4)


def test_changing_oil_and_gas_constraints_keeps_all_actual_limits_feasible():
    a, b = accelerated('facet_safeguarded'), accelerated()
    originals = [c.facility.limits for c in (a, b)]
    for i in range(3):
        for c, original in zip((a, b), originals):
            if i == 1:
                bound = c.report['facility']['limits'][0]['value']*.5
                c.facility.limits = (replace(original[0], maximum=bound), *original[1:])
            elif i == 2:
                c.facility.limits = original
            assert c.step(60., current_time=i*60.)
            assert all(row['passed'] for row in c.report['facility']['limits'])
            assert c.report['facility']['gas_export']['valve']['utilization'] <= 1
        assert abs(a.report['fraction']-b.report['fraction']) <= 1e-10


def test_wider_inactive_certificate_matches_an_independently_solved_state():
    from simulator.rate_cap_invariance import certificate_covers, make_inactive_cap_certificate
    s = build().solver
    for w in s.wells.wells:
        w.control_value = 35.6
    original = pickle.dumps(s)
    caps = {w.name: w.control_value for w in s.wells.wells}
    assert s.step(60.)
    targets = {w.name: s.wells.step_targets[i] for i, w in enumerate(s.wells.wells)}
    floors = {w.name: s.wells.bhp_limit(w) for w in s.wells.wells}
    old = make_inactive_cap_certificate(s.wells.surface_report, targets, caps, floors, 1.)
    new = make_inactive_cap_certificate(s.wells.surface_report, targets, caps, floors, 1., version=2)
    assert old is not None and new is not None
    assert not certificate_covers(old, .5) and certificate_covers(new, .5)
    other = pickle.loads(original)
    for w in other.wells.wells:
        w.control_value *= .5
    assert other.step(60.)
    for key in ('pressure', 's_w', 's_g', 'rs'):
        torch.testing.assert_close(getattr(s.fluid, key), getattr(other.fluid, key), rtol=0, atol=0)
    assert s.wells.surface_report == other.wells.surface_report


@pytest.mark.parametrize('budget', [0, -1, True, 1.5])
def test_budget_is_validated(budget):
    with pytest.raises(ValueError, match='positive integer'):
        replace(build().control, sensitivity_sparse_budget_bytes=budget)
