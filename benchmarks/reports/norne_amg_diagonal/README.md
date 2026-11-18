# Norne: direct CSR diagonal extraction for AMG

A CPU/CUDA profile of the third linear solve after resuming Norne at day 2431
showed substantial index-operation, kernel-launch and synchronization cost.
The change replaces AMG's row expansion, boolean indexing and scatter for
CSR diagonal extraction with a direct GPU kernel. No floating-point reduction
is introduced: diagonal values are loaded directly. Missing entries are zero;
unsorted and long rows are supported. Every call reads current matrix values.
No cross-matrix numerical cache, tolerance or timestep change is used.
CPU/unsupported dtypes and gradient-bearing matrices keep the original path.

## Validation protocol

29 focused CPU/CUDA tests passed, including empty/long/unsorted rows, missing
and duplicate diagonal entries, both index widths, float32/float64, in-place
matrix value updates, graph replay and existing AMG/FIM regressions. Final
run used CUBLAS_WORKSPACE_CONFIG=:4096:8. Tests finished before benchmarks.

RTX 3090 explicitly selected by UUID. Frozen reference and candidate sources
differ only in `solver/classical_amg.py` and `solver/csr_spmv.py`. Reference
already contains the earlier FGMRES vector-update kernel. Two sequential
pairs in AB/BA order for each window:

- 0–115 days, freshly initialized, strict validator with component-balance
  and source/input-integrity audits.
- 2431–2521 days (90 days), identical historical checkpoint, strict convergence
  checks and exact paired regression. This continuation runner does not perform
  an independent cumulative component-balance audit.

Times are sums of solver step calls; imports, grid/deck setup, audit and output
are excluded. No profiler or simultaneous GPU test was used in timed runs.
Compare saved fields/SUMMARY exactly, and all attempted-step clocks, lengths,
acceptance flags and Newton iteration counts. The late final fields are float64;
the initial validator stores field snapshots in its standard output precision.

`results.json` contains every observation, with original numerical outputs,
statuses and manifests in run subdirectories. The patch is relative to the
pre-experiment workspace, not necessarily Git HEAD. Archived scripts require
the existing repository-local Norne deck/checkpoint and fresh output paths.
Profile tables locate costs only and must not be interpreted as benchmark times.
These measurements concern reservoir+wells, not the full surface network.
No full nine-year runtime or external physical validation is inferred.

## Retained result

Direct diagonal extraction is retained. Median step times: initial window
**11.6825 → 11.6192 s** (0.54%, within observed scatter), late window
**17.6417 → 17.2828 s** (2.03%, both pairs improved). All numerical and work
comparisons are exact; every window accepted 17 steps with no failures.

On the same profiled third linear solve, CUDA stream synchronization calls
fell **187 → 143**, kernel launches **3845 → 3315**, and index-put calls
35 → 24. Profiled CPU self-time was 44.172 → 36.586 ms; these instrumented
times are diagnostic, not a whole-run speed estimate. The profiled continuations
also produced exactly matching saved outputs and work.

An additional experiment passed known CSR output sizes to eight existing
`repeat_interleave` call sites. It did not improve window medians: early
12.2018 → 12.5752 s, late 18.3058 → 18.5009 s. That addition was reverted.
Its separately measured outputs and patch are archived in `final/` (the
name reflects the attempted combined candidate, **not** retained production).
Production matches the original `change.patch` and root `results.json`.
Late frozen source hashes were verified after execution; existing workspace
changes were preserved and no Git staging/commit/push was performed.
