"""Engineering budgets must not bypass source, refinement or conservation gates."""
import copy
import json
import shutil

import pytest
from check_facility_engineering_profile import check, validate_budgets
from test_facility_refinement_pair import CANDIDATE, REFERENCE

BUDGETS = {
    'fields': {'final_pressure_pa': 1000., 'final_s_w': .0002, 'final_s_g': .0002,
               'final_rs': .05, 'withdrawal_relative': .002},
    'wells': {'bhp_abs_pa': 1000., 'component_rate_scaled': .002,
              'cumulative_mass_relative': .002},
}


def test_real_archived_refinement_passes_declared_budgets():
    result = check(CANDIDATE, REFERENCE, BUDGETS, days=.02)
    assert result['accepted']
    assert result['well_trajectories']['budgets'] == BUDGETS['wells']
    assert result['same_python_and_pvt']


def test_tighter_rate_budget_rejects_real_pair():
    budgets = copy.deepcopy(BUDGETS)
    budgets['wells']['component_rate_scaled'] = 1e-12
    result = check(CANDIDATE, REFERENCE, budgets, days=.02)
    assert not result['accepted'] and not result['well_trajectories']['accepted']


@pytest.mark.parametrize('value', [True, 0, -1, float('inf'), float('nan'), '0.1'])
def test_invalid_budgets_rejected(value):
    budgets = copy.deepcopy(BUDGETS)
    budgets['wells']['bhp_abs_pa'] = value
    with pytest.raises(ValueError, match='Budgets'):
        validate_budgets(budgets)


def test_incomplete_budgets_rejected():
    budgets = copy.deepcopy(BUDGETS)
    del budgets['fields']['final_rs']
    with pytest.raises(ValueError, match='complete fields'):
        validate_budgets(budgets)


@pytest.mark.parametrize('mutation', ['device', 'threads', 'gpu', 'failed'])
def test_changed_execution_or_failure_cannot_qualify(tmp_path, mutation):
    target = tmp_path / 'reference'
    shutil.copytree(REFERENCE, target)
    summary_path = target / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if mutation == 'device':
        summary['device'] = 'different'
    elif mutation == 'threads':
        summary['execution']['torch_threads'] += 1
    elif mutation == 'gpu':
        path = target / 'protocol.json'
        protocol = json.loads(path.read_text())
        protocol['cuda_visible_devices'] = 'different GPU'
        path.write_text(json.dumps(protocol))
    else:
        (target / 'failure.json').write_text('{}')
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        check(CANDIDATE, target, BUDGETS, days=.02)
