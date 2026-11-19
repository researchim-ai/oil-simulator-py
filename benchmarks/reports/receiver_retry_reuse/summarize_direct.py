"""Numerical refinement deltas, not exact-solution error estimates."""
import json
from pathlib import Path

root = Path(__file__).parent
coarse, fine = json.loads((root / 'direct.json').read_text())
local = json.loads((root / 'comparison.json').read_text())[0]


def delta(a, b):
    if isinstance(a, list):
        return max((delta(x, y) for x, y in zip(a, b, strict=True)), default=0.)
    return abs(a - b)


def compare(a, b):
    return {
        'inlet_pressure_pa': abs(a['inlet_pressure_pa'] - b['inlet_pressure_pa']),
        'temperature_k': delta(a['state']['column']['temperature_k'], b['state']['column']['temperature_k']),
        'component_moles': delta(a['state']['column']['component_moles'], b['state']['column']['component_moles']),
        'total_energy_j': delta(a['state']['column']['total_energy_j'], b['state']['column']['total_energy_j']),
        'reservoir': {key: delta(a['reservoir'][key], b['reservoir'][key]) for key in a['reservoir']},
    }


local['inlet_pressure_pa'] = local['report']['stages'][-1]['pressure_trials'][-1]['receiver_pressure_pa']
result = {
    'local_integration_s': local['wall_s'],
    'direct_005_s': coarse['integration_s'],
    'direct_0025_s': fine['integration_s'],
    'direct_refinement_differences': compare(coarse, fine),
    'local_vs_direct_0025_differences': compare(local, fine),
    'all_direct_audits_accepted': all(a['accepted'] for row in (coarse, fine) for a in row['audits']),
    'direct_max_temporal_error_ratio': max(r['temporal_error_ratio'] for row in (coarse, fine) for r in row['reports']),
}
(root / 'direct_summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
