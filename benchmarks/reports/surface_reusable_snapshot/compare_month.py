"""Exact source-pair regression, including every accepted capacity trace."""
import argparse
import json
from pathlib import Path
from benchmark_initial_cell_cache import require_equal
from check_facility_refinement_pair import completed
from check_fim_facility_forecast import audit

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference',type=Path)
parser.add_argument('candidate',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ref,new=args.reference,args.candidate
sa,a,_=completed(ref,30.)
sb,b,_=completed(new,30.)
assert a['config']==b['config'] and a['schedule']==b['schedule']
for key in ('torch_version','torch_threads','device_name','gathering_backend','hydraulic_backend'):
    assert sa['execution'][key]==sb['execution'][key], key
for bundle in (a,b):
    assert audit(bundle['config'],bundle['result'],expected_schedule=bundle['schedule'],expected_days=30.)['accepted']
require_equal(a['result'],b['result'])

def sources(directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    return {marker+name.split(marker,1)[1]:digest
            for name,digest in protocol['source_sha256'].items()
            for marker in ('/src/','/scripts/') if marker in name}
x,y=sources(ref),sources(new)
assert x.keys()==y.keys()
changed=sorted(k for k in x if x[k]!=y[k])
assert changed==['/src/simulator/facility_coupling.py','/src/simulator/fim_transaction.py'],changed
result={'accepted':True,'exact_all_accepted_stages':True,'accepted_stages':len(b['result']['accepted_substeps']),
        'changed_sources':changed,'reference':str(ref),'candidate':str(new),
        'reference_wall_s':sa['solve_wall_s'],'candidate_wall_s':sb['solve_wall_s'],
        'scope':'Exact execution regression; shared host at different times, not an isolated speedup measurement.'}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
