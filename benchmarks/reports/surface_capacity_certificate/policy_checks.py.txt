"""Expanded exact inactive-cap reuse without auxiliary sensitivity solves."""
import copy
import pickle
from dataclasses import replace

import pytest
import torch
from test_rate_cap_invariance import case

from simulator.facility_coupling import CoupledFacilityControl


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_expanded_certificate_matches_direct_solves_through_events_and_restart(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    a, b = case(True, device), case(True, device)
    for c in (a, b):
        for w in c.solver.wells.wells:
            w.control_value *= .005
    b.control = replace(b.control, inactive_cap_certificate_version=2)
    saved = 0
    for i, controls in enumerate(({}, {'gas_valve_max_opening': .8}, {})):
        for c in (a, b):
            c.controls = copy.deepcopy(controls)
            assert c.step(60., current_time=60.*i)
        saved += a.report['reservoir_evaluations'] - b.report['reservoir_evaluations']
        assert a.report['fraction'] == pytest.approx(b.report['fraction'], abs=1e-9)
        for field, tolerance in [('pressure', .1), ('s_w', 1e-8), ('s_g', 1e-8)]:
            torch.testing.assert_close(getattr(a.solver.fluid, field),
                                       getattr(b.solver.fluid, field), rtol=0, atol=tolerance)
        assert 'sensitivity_work' not in b.report
        assert all(row['inactive_cap_certificate']['version'] == 2
                   for row in b.report['trace'] if 'inactive_cap_certificate' in row)
        assert b.report['capacity_bracket']['width'] <= b.control.fraction_tolerance
        a, b = [pickle.loads(pickle.dumps(c)) for c in (a, b)]
    assert saved > 0


@pytest.mark.parametrize('value', [True, 0, 3, 2., '2', None])
def test_invalid_certificate_version(value):
    with pytest.raises(ValueError, match='certificate_version'):
        CoupledFacilityControl(inactive_cap_certificate_version=value)


@pytest.mark.parametrize('mutation', ['physics', 'tolerance', 'search', 'boolean'])
def test_strict_comparison_rejects_unrelated_changes(mutation):
    from check_capacity_certificate_policy import check_policy_pair
    a = {'simulation': {'coupled_facility': {'control': {
        'search_method': 'facet_safeguarded', 'fraction_tolerance': 1e-9,
        'reuse_inactive_caps': True}}}, 'porosity': .2}
    b = copy.deepcopy(a)
    policy = b['simulation']['coupled_facility']['control']
    policy['inactive_cap_certificate_version'] = 2
    check_policy_pair(a, b)
    if mutation == 'physics':
        b['porosity'] = .3
    elif mutation == 'tolerance':
        policy['fraction_tolerance'] = 1e-3
    elif mutation == 'search':
        policy['search_method'] = 'sensitivity_safeguarded'
    else:
        policy['inactive_cap_certificate_version'] = True
    with pytest.raises(ValueError, match='Only inactive_cap_certificate_version'):
        check_policy_pair(a, b)


def test_forecast_audit_rejects_certificate_not_requested_by_config():
    from check_fim_facility_forecast import audit
    from run_fim_blackoil_surface import run_case
    from test_fim_facility_forecast import config
    c = config()
    policy = c['simulation']['coupled_facility']['control']
    policy.update(reuse_inactive_caps=True, inactive_cap_certificate_version=2,
                  search_method='facet_safeguarded', min_fraction=1e-5)
    for w in c['wells']:
        w['control_value'] *= 100
    r = run_case(c, 1e6, .00005, .00005)
    assert audit(c, r, expected_days=.00005)['accepted']
    proofs = [row['inactive_cap_certificate'] for s in r['accepted_substeps']
              for row in s['facility_coupling']['trace'] if 'inactive_cap_certificate' in row]
    assert proofs
    bad = copy.deepcopy(c)
    bad['simulation']['coupled_facility']['control']['inactive_cap_certificate_version'] = 1
    with pytest.raises(ValueError, match='version differs'):
        audit(bad, r, expected_days=.00005)
