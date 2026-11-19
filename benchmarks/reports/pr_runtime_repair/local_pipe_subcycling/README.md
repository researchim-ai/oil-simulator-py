# Conservative local pipe subcycling

This is a short numerical qualification, not a completed transient facility
cycle or an experimental validation of multiphase closures.

The original integrated facility retains every physical node, pipe inventory,
momentum, component and energy state. Only `gas_recycle_0` is selected for a
fast operator, together with its two finite vessel boundaries. The other
operator receives their updated stocks. Global step doubling controls the
operator-splitting error; local transport uses 0.1 times the original relative,
temperature and velocity tolerances. No physical tolerance is relaxed.

The interval 0–1e-4 s completed in 134.2177 CPU seconds in one outer attempt,
with temporal error 0.11548. The accepted trajectory contains two split
halfsteps, 106 accepted local transport records and two slow plant records.
The independent checker reconstructed all exchanges and obtained an energy
residual of 2.4161e-8 J. Source files and physical-source SHA256 are included;
formatting was normalized after the run without a numerical change.

The original monolithic first trial at the same interval had error 22.27425,
rejected again at 1.90696e-5 s, and first accepted 3.39537e-6 s. The original
single-interval reference was interrupted by SIGTERM. A reference then
completed the same end time through six checkpointed reporting intervals;
every interval passed its independent audit. Its 468.79 s includes extra
reporting boundaries and model reconstruction: do not advertise their ratio
as a clean timing speedup.

Compared with that reference, the maximum node pressure difference is
14.88 Pa, temperature difference 1.47e-6 K, and velocity difference
0.00202438 m/s in the fast recycle pipe. The original error norm is 2.02438,
so this comparison alone does not establish agreement within one original
local tolerance. Temporal refinement is recorded separately; neither
trajectory is an exact analytical reference.

`source/` captures the first probe implementation. `post_probe_fixes.patch`
records subsequent handling of slow Darcy isolation and changing outer
step-size hints in the independent audit, plus regression additions. The
first fixture has no Darcy links. Physical model tolerances were not relaxed.

Files:

- `result.json.gz`: complete request, accepted trajectory and numerical policy.
- `audit.json.gz`: independent stage and global conservation checks.
- `summary.json`: measured scope and limitations.
- `source/`, `source_sha256.json`: implementation snapshot and physical sources.

Recheck without running the expensive integration:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
python - <<'PY'
import gzip, json
from check_pr_controlled_surface import audit
with gzip.open('benchmarks/reports/pr_runtime_repair/local_pipe_subcycling/result.json.gz', 'rt') as f:
    result = json.load(f)
print(audit(result['config'], result))
PY
```

The opt-in preparation CLI is documented in `docs/PERFORMANCE.md`. Supported
partitions end at plain stirred finite vessels. Fast actuators, protection
trips, equipment attached to selected pipe bodies and complex boundary-vessel
states are rejected rather than silently omitted.

Continuation from1e-4 to1e-3 s completed in235.3767 CPU seconds with four
attempts, all accepted. Largest macrostep3.375e-4 s; temporal errors were
0.05462,0.03429,0.07647,0.02390. The accepted records contain132 local and8
slow records. The independent audit passed with global energy residual
-9.9288e-8 J. See `continuation.json.gz`. Including the first interval,
physical time0.001 s cost369.59 CPU seconds: this is not production-speed
qualification. The numerical global-step limitation has been decoupled;
local physical solves remain expensive.

Temporal refinement completed independently: maximum outer step halved to
5e-5 s and local error fraction reduced from0.1 to0.025. Both macro attempts
were accepted. Its endpoint differs from the original subcycled result by
0.380518 in the ORIGINAL full-network error norm (below1); independent stage
and global audits passed. Runtime270.61 s,258 local records,4 slow records;
energy residual-2.3896e-7 J. See `refinement.json.gz`. This supports the short
subcycled trajectory's temporal resolution, not exactness or full-cycle
qualification. Final targeted regressions:34 passed in103.14 s.
