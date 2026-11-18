"""Count successful CPR warmups/captures/replays without changing proposals."""
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
args = sys.argv[3:]
sys.path[:0] = [str(root/'src'), str(root/'scripts')]
from simulator.cpr_gpu import CPRPreconditionerGPU

original = CPRPreconditionerGPU._matvec_graph
counts = {'warmup_calls':0, 'captures':0, 'replays':0, 'failed_calls':0,
          'applications_per_instance':[]}

def counted(self, rhs):
    if not hasattr(self, '_measurement_index'):
        self._measurement_index = len(counts['applications_per_instance'])
        counts['applications_per_instance'].append(0)
    counts['applications_per_instance'][self._measurement_index] += 1
    before = self._graph
    try:
        result = original(self, rhs)
    except RuntimeError:
        counts['failed_calls'] += 1
        raise
    if isinstance(self._graph, tuple):
        counts['replays'] += 1
        counts['captures'] += int(not isinstance(before, tuple))
    else:
        counts['warmup_calls'] += 1
    return result

CPRPreconditionerGPU._matvec_graph = counted
sys.argv = [str(root/'scripts/run_fim_facility_forecast.py'), *args]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    counts['graphs_enabled_at_end'] = CPRPreconditionerGPU.use_graphs
    output.write_text(json.dumps(counts, indent=2)+'\n')
