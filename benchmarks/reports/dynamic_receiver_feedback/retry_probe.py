import json
import time
from dataclasses import asdict
from pathlib import Path

from check_dynamic_receiver import audit_step
from test_fim_dynamic_receiver import build
from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM

base = build()
control = asdict(base.control)
# Explicit tighter accuracy case, leaving EOS, ports and pressure iteration intact.
for key in control:
    if key.startswith('temporal_'):
        control[key] /= 100
m = DynamicReceiverCoupledFIM(base.solver, base.receiver, base.receiver_state,
                            base.assay, base.thermal_conditions, base.inlet_height_m, control)
before = m.receiver_state
start = time.perf_counter()
accepted = m.step(.02)
result = {'control': control, 'initial_dt_s': .02, 'initial_accepted': accepted,
          'initial_attempt': m.last_attempt, 'initial_wall_s': time.perf_counter() - start}
assert not accepted and m.receiver_state == before
retry_dt = m.limit_rejected_time_step_s(.01, failed_dt_s=.02)
start = time.perf_counter()
success = m.step(retry_dt)
result.update(retry_dt_s=retry_dt, retry_accepted=success,
              retry_wall_s=time.perf_counter()-start, retry_attempt=m.last_attempt)
if success:
    result['retry_report'] = m.report
    result['independent_audit'] = audit_step(
        {**m.to_dict(), 'initial_state': before.to_dict()}, m.report)
Path('/tmp/dynamic-receiver-feedback/retry.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
assert success, m.last_attempt
print({k: v for k, v in result.items() if k not in ('retry_report', 'independent_audit')})
