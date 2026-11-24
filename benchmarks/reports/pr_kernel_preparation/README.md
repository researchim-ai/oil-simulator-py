# Connected native execution: preparation and private replay storage

This work preserves the existing PR equations, physical stocks, geometry,
controls and acceptance tolerances. It uses the same 16-node operating
checkpoint and three reporting intervals as `pr_connected_native`:
**0.04864483384 simulated seconds**, starting at **0.70623809959 s**.
It is not a complete startup/isolation/restart or reservoir benchmark.

## Preparation diagnosis and kernel verification

`steady/summary.json` compares the existing backward-Euler policy with
embedded fast Radau plus native execution. Three warm CPU runs give medians
**16.5743 versus 5.77967 s (2.8677×)**. All trajectories pass the independent
physical audit; reporting-state differences remain within the original
requested error norm. This comparison includes the integrator/execution
combination, not just the small verification change described below.

The first native run takes **152.416 s with a previously populated Inductor
disk cache**. Of that, recorded lowering takes 60.2993 s, code generation/cache
loading 5.45333 s, trace/dispatch remainder 79.7822 s and eager numerical
verification 1.05932 s. The trace/dispatch remainder includes the first replay,
not only the Dynamo tracer. These counters describe successful admitted
compilations and do not account for interrupted preparation.

This identifies graph construction as the remaining preparation bottleneck
on this cached repeat. It does **not** establish that changing verification
reduced the previous 399 s preparation to 152 s: disk-cache state and host
conditions differ. Nor should the 5.78 s warm observation be attributed to
verification alone; previous warm runs already performed no new verification.

Previously, compiling any new operator invalidated the checks of every
previous operator. Verification now belongs to a concrete guarded kernel;
new specializations are checked, and returning to a verified specialization
retains its check. This segment needs **12 checks for 12 kernels**, versus
25 checks in the preceding measured implementation. A deliberately corrupted
new specialization is rejected before use, even after another specialization
has already passed. Final Newton acceptance still uses the original residual.

## Private replay inputs and layout correctness

Each admitted CPU kernel now owns bounded private input buffers. Every replay
copies current tensor values into those buffers; outputs retain independent
storage. Mutable coefficients and noncontiguous input views remain live.
Constants and retained input buffers share `maximum_constant_bytes`; admission
falls back to original AD rather than exceeding that resident budget.
Explicit close clears the buffers even if a compiler wrapper remains alive.

A new view test found a real bug in the preceding cloning path: a sliced
coefficient could be compacted by `clone`, while the compiled kernel expected
its original strides. Compilation now uses examples matching the owned layout.
Original Dynamo examples are retained when their layout already agrees,
avoiding a separate installed functorch/FakeTensor metadata failure. AD tracing
uses separate copies. Tests cover view updates, old-output ownership, memory
admission, bounded storage and eager final acceptance.

`replay_validated/summary.json` is the final source-bound alternating comparison
of per-call cloning and private buffers. Both policies reuse the **same compiled
kernels** and restart from the same physical state. Initialization is timed;
audits and serialization are not. One preparation run and one settling run are
reported separately. Every physical reporting state must match exactly across
policies; every trajectory must pass the independent audit. The command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
TORCHINDUCTOR_CACHE_DIR=/tmp/oil_pr_native_execution \
/home/user/miniconda3/bin/python scripts/benchmark_pr_native_replay.py \
  benchmarks/reports/pr_connected_native/steady_embedded.json \
  --repeats 3 --output-dir /tmp/pr-replay-repeat
```

The final three-pair comparison passes all eight trajectory audits, including
preparation and settling. Warm medians are **8.40587 s cloning versus 5.80818 s
private buffers** (observed ratio 1.4472). Individual cloning runs are
5.89837 / 8.40587 / 15.09334 s, versus 5.80818 / 5.74485 / 7.57252 s with
buffers. These broad shared-host ranges and the earlier 4.4% pilot do not
establish a reproducible 45% throughput gain. All physical reporting states
match exactly; there are twelve verified kernels and no native fallback.

Every supplied three-interval schedule previously makes **46,090 native-input
clone allocations**; private replay replaces them by updates of retained
storage. Twelve kernels allocate 1,426 input buffers in total, retaining
39,624 input bytes plus 424 constant bytes. Those are native replay counters,
not all allocations in the complete model. Final-context preparation takes
158.976 s with the populated disk cache (64.969 s lowering, 5.631 s code
generation/loading and 77.413 s trace/dispatch remainder).

The old view-layout failure is independently reproduced from the retained
pre-change implementation in `old_view_layout_probe.json`. Current regression
checks pass: eighteen native execution tests and two journal/recovery tests;
the final updated view-policy test is rerun separately. Logs and hashes are in
`regression_checks.json`.

`replay_buffers/` is an earlier five-pair pilot before the view-layout fix. Its
6.63671 versus 6.35835 s medians are only a **4.4% shared-host observation**,
with overlapping run ranges. They are not evidence of a large speedup.
`replay_release/` was interrupted after a compiler metadata failure and is not
qualified evidence. Each completed series retains exact source hashes; do not
silently substitute one version's qualification for another.

Input-buffer counters count native replay ownership copies, not all tensor
allocations in the model: output copies, AD tracing and Newton input
normalization still exist. No whole-installation real-time performance is
claimed. The previous complete operating-cycle qualification remains pinned
to its own source hashes in `pr_connected_native/cold_cycle_restart.json`.

## Rejected alternatives

`excluded_candidates.json` and the accompanying reproductions retain two
isolated diagnostics. Sparse assembly of the captured 268-unknown problem is
exact but uses 168 colors in three chunks and takes 0.876 s versus 0.319 s for
dense forward AD. It was not substituted into this small coupled problem;
the existing sparse path remains available for large graphs.

Fake-mode lowering fails in this installed PyTorch version with a functorch
layer assertion and nested-forward-AD failure. It was tested in a separate
process and excluded. Production retains real ATen lowering and numerical
verification. None of these numerical checks constitutes experimental
calibration of multiphase closures or equipment.
