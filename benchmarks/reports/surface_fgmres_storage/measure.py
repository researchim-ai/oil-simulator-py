import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path('/home/user/workspace/oil-simulator-py-impes')
WORK = ROOT / '.scratch/fgmres_lazy'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from solver.fgmres import fgmres
from check_facility_refinement_pair import completed
from compare_fim_facility_forecast import compare as compare_fields
from check_long_facility_trajectory import compare as compare_wells
from check_facility_capacity_search import FIELD_BUDGETS, WELL_BUDGETS

spec = importlib.util.spec_from_file_location('solver.reference', WORK/'reference.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
torch.set_num_threads(1)
torch.cuda.set_per_process_memory_fraction(.12)
device = 'cuda'
rows = []
for n in [3*262144, 3*1048576]:
    d = (1. + torch.arange(n, device=device, dtype=torch.float64) % 4)
    b = torch.ones_like(d)
    answers = {}
    for run in range(4):
        for label, solve in ([('reference', ref.fgmres), ('candidate', fgmres)]
                             if run % 2 == 0 else [('candidate', fgmres), ('reference', ref.fgmres)]):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            stats = {}
            start = time.perf_counter()
            x, info = solve(lambda v: d*v, b, restart=50, tol=1e-11, stats=stats)
            torch.cuda.synchronize()
            wall = time.perf_counter()-start
            assert info == 0
            residual = float(torch.linalg.vector_norm(d*x-b)/torch.linalg.vector_norm(b))
            assert residual < 1e-11
            answers[label] = x.cpu()
            rows.append(dict(n=n, label=label, run=run, warmup=run==0,
                             wall_s=wall, extra_peak_bytes=torch.cuda.max_memory_allocated()-before,
                             stats=stats, true_residual=residual))
            del x
        assert torch.equal(answers['reference'], answers['candidate'])
    del d,b,answers
    (WORK/'micro.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('MICRO', n, [(label, statistics.median(r['wall_s'] for r in rows
          if r['n']==n and r['label']==label and not r['warmup']))
          for label in ('reference','candidate')], flush=True)

# Exactly one source difference; independent completion, checksum and material audit.
sources = {}
for label in ('reference', 'candidate'):
    source = WORK/label
    for name in ('src','scripts'):
        shutil.copytree(ROOT/name, source/name, ignore=shutil.ignore_patterns('__pycache__'))
    if label == 'reference':
        shutil.copyfile(WORK/'reference.py', source/'src/solver/fgmres.py')
    sources[label] = source
manifests = {label: {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for folder in ('src','scripts') for p in (source/folder).rglob('*.py')}
             for label,source in sources.items()}
assert manifests['reference'].keys() == manifests['candidate'].keys()
assert [k for k in manifests['reference'] if manifests['reference'][k] != manifests['candidate'][k]] == ['src/solver/fgmres.py']
(WORK/'source_manifests.json').write_text(json.dumps(manifests,indent=2)+'\n')
config = json.loads((ROOT/'.scratch/capacity_amg/final/large_True.json').read_text())
(WORK/'config.json').write_text(json.dumps(config, indent=2)+'\n')
rows = []
for repeat in range(2):
    outputs = {}
    bundles = {}
    for label in (('reference','candidate') if repeat==0 else ('candidate','reference')):
        source=sources[label]
        out=WORK/f'large_{repeat}_{label}'
        outputs[label]=out
        env=dict(os.environ,PYTHONPATH=f'{source/"src"}:{source/"scripts"}')
        cmd=[sys.executable,str(source/'scripts/run_fim_facility_forecast.py'),
             '--config',str(WORK/'config.json'),'--days','.00005','--report-days','.00005',
             '--device','cuda','--cuda-memory-fraction','.12','--output',str(out)]
        with (WORK/f'large_{repeat}_{label}.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
        summary,bundle,path=completed(out,.00005)
        bundles[label]=bundle
        rows.append(dict(label=label,repeat=repeat,wall_s=summary['solve_wall_s'],
                         execution=summary['execution'],work=bundle['result']['temporal_work']))
        print('FORECAST',json.dumps(rows[-1]),flush=True)
    assert bundles['reference']['config']==bundles['candidate']['config']
    paths=[outputs[x]/'trajectory.json.gz' for x in ('reference','candidate')]
    fields=compare_fields(*paths,FIELD_BUDGETS)
    wells=compare_wells(*paths)
    assert fields['accepted'] and wells['mode_mismatches']==0
    assert all(wells['errors'][k]<=v for k,v in WELL_BUDGETS.items())
    (WORK/f'accuracy_{repeat}.json').write_text(json.dumps(dict(fields=fields,wells=wells),indent=2)+'\n')
    (WORK/'forecast.json').write_text(json.dumps(rows,indent=2)+'\n')
print('COMPLETE',flush=True)
