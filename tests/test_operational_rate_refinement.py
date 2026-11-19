"""Adaptive refinement must tighten the active rate controller, not only its cap."""
import copy

import pytest
from check_facility_refinement_pair import same_other_settings, stricter
from validate_operational_forecast import refined_config


def case():
    return {'simulation': {'max_time_step_days': 1., 'initial_time_step_days': .001,
            'coupled_facility': {'time_control': {'rate_relative_tolerance': .001}}}}


def test_rate_refinement_preserves_physics_and_tightens_actual_controller():
    original = case()
    saved = copy.deepcopy(original)
    fine = refined_config(original)
    a, b = {'config': original}, {'config': fine}
    assert original == saved
    assert stricter(a, b) == 'endpoint_rate_and_maximum_step'
    same_other_settings(a, b)
    policy = fine['simulation']['coupled_facility']['time_control']
    assert policy['rate_relative_tolerance'] == .0005
    assert policy['rate_absolute_tolerance_kg_s'] == 5e-9
    assert fine['simulation']['max_time_step_days'] == .5


@pytest.mark.parametrize('mutation', ['unchanged', 'looser', 'growth', 'family', 'physics'])
def test_unqualified_adaptive_comparisons_are_rejected(mutation):
    a = case()
    b = refined_config(a)
    policy = b['simulation']['coupled_facility']['time_control']
    if mutation == 'unchanged':
        b['simulation']['coupled_facility']['time_control'] = copy.deepcopy(a['simulation']['coupled_facility']['time_control'])
    elif mutation == 'looser':
        policy['rate_absolute_tolerance_kg_s'] = 1e-5
    elif mutation == 'growth':
        policy['maximum_growth'] = 4.
    elif mutation == 'family':
        del b['simulation']['coupled_facility']['time_control']
    else:
        b['fluid'] = {'density': 999.}
    with pytest.raises(ValueError):
        stricter({'config': a}, {'config': b})
        same_other_settings({'config': a}, {'config': b})
