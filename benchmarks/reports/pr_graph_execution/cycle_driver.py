"""Complete original 16-node 1 s startup/isolation/restart verification, CPU."""
import argparse
import gzip
import hashlib
import json
import tarfile
import time
from pathlib import Path
from audit_pr_controlled_mixed import audit
from run_pr_controlled_surface import run
from surface.pr_collocation import PRCollocation

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference',action='store_true')
parser.add_argument('output',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[3]
config=json.loads((root/'configs/surface_pr_integrated_facility_subcycled.json').read_text())
p=args.output;p.mkdir(parents=True,exist_ok=False)
(p/'request.json').write_text(json.dumps(config,indent=2)+'\n')
paths=sorted((root/'src/surface').glob('*.py'))+sorted((root/'scripts').glob('*.py'))+[Path(__file__).resolve()]
hashes={str(v.relative_to(root)):hashlib.sha256(v.read_bytes()).hexdigest() for v in paths}
with tarfile.open(p/'sources.tar.gz','w:gz') as arc:
 for v in paths:arc.add(v,arcname=str(v.relative_to(root)))
if args.reference:
 PRCollocation._dense_cpu_jacobian_mode='forward';PRCollocation._reuse_sdirk_points=False
start=time.perf_counter()
with (p/'progress.jsonl').open('w',buffering=1) as stream:
 def progress(event):
  row={**event,'wall_s':time.perf_counter()-start}
  stream.write(json.dumps(row)+'\n')
  if event['event']=='accepted': print(json.dumps(row),flush=True)
 report=run(config,integrator='sdirk2-common',progress=progress)
wall=time.perf_counter()-start
with gzip.open(p/'trajectory.json.gz','wt') as stream:json.dump(report,stream)
print('integration complete',wall,flush=True)
certificate=audit(config,json.loads(json.dumps(report)))
for name,digest in hashes.items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed: '+name)
summary={'accepted':certificate['accepted'],'reference':args.reference,'integration_wall_s':wall,
         'model_duration_s':sum(c['duration_s'] for c in config['schedule']),
         'audit':certificate,'source_sha256':hashes,
         'nonlinear_solves':sum(i.get('collocation_nonlinear_solves',0) for i in report['intervals']),
         'fallback_intervals':sum(i.get('fallback_intervals',0) for i in report['intervals']),
         'rejected_attempts':sum(i.get('rejected_attempts',0) for i in report['intervals']),
         'scope':'one complete original CPU eager cycle including initialization and durable progress; audits/serialization excluded; not a median warm throughput benchmark'}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary['audit']),flush=True)
