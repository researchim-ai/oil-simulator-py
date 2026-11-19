import json,time,signal
from pathlib import Path
from run_pr_controlled_surface import run
from check_pr_controlled_surface import audit
root=Path('/tmp/pr-local-subcycles');previous=json.loads((root/'result.json').read_text())
c=previous['config'];c['initial']={'state':previous['intervals'][-1]['state']};c['schedule']=[{'duration_s':9e-4}]
t=time.perf_counter()
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('240s continuation budget')));signal.alarm(240)
try:
 r=run(c,progress=lambda p:print(json.dumps(p),flush=True));seconds=time.perf_counter()-t
 signal.alarm(0);r=json.loads(json.dumps(r));(root/'continuation-raw.json').write_text(json.dumps({'seconds':seconds,'report':r}));checked=audit(c,r)
 (root/'continuation.json').write_text(json.dumps({'seconds':seconds,'report':r,'audit':checked}))
 print(json.dumps({'seconds':seconds,'audit':checked}))
except Exception as e:
 (root/'continuation-failure.json').write_text(json.dumps({'seconds':time.perf_counter()-t,'error':repr(e)}));raise
