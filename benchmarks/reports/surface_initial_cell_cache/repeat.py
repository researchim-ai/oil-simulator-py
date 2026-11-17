import copy
import hashlib
import json
from pathlib import Path
import statistics

from run_fim_blackoil_surface import run_case

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / 'initial_cache.json').read_text())
sources = sorted((ROOT / 'src').rglob('*.py'))
before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
rows = []
for repeat, order in enumerate(((False, True), (True, False), (False, True))):
    pair = {}
    for enabled in order:
        c = copy.deepcopy(config)
        c['simulation']['coupled_facility']['control']['reuse_initial_cell_state'] = enabled
        result = run_case(c, 1e6, .003, .003, 'cpu')
        pair[enabled] = result
        rows.append({'repeat': repeat, 'enabled': enabled, 'solve_wall_s': result['solve_wall_s'],
                     'accepted_stages': len(result['accepted_substeps'])})
        print(json.dumps(rows[-1]), flush=True)
    a, b = pair[False], pair[True]
    for key in ('final_pressure_pa', 'final_s_w', 'final_s_g', 'final_rs', 'withdrawn_component_mass_kg'):
        assert a[key] == b[key], key
    assert len(a['accepted_substeps']) == len(b['accepted_substeps'])
    for sa, sb in zip(a['accepted_substeps'], b['accepted_substeps'], strict=True):
        for key in ('time_days', 'dt_s', 'withdrawn_component_mass_kg', 'surface', 'newton_iterations'):
            assert sa[key] == sb[key], key
        for key in ('trace', 'material_account', 'facility', 'fraction'):
            assert sa['facility_coupling'][key] == sb['facility_coupling'][key], key
    print('pair physical states: exact', repeat, flush=True)
after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
assert before == after, 'Profile sources changed'
medians = {str(enabled): statistics.median(r['solve_wall_s'] for r in rows if r['enabled'] is enabled)
           for enabled in (False, True)}
report = {'scope': 'Three alternating CPU pairs on a shared host; 0.003 days, 1024 cells, 16 wells. '
                   'All accepted fine-stage well states, material ledgers and final fields exactly equal.',
          'rows': rows, 'median_s': medians, 'observed_ratio': medians['False'] / medians['True'],
          'sources_unchanged': True, 'source_sha256': before}
(ROOT / 'initial_cache_repeats.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2),flush=True)
