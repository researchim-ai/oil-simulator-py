# Norne: MGS vector update kernel

The candidate combines only `w - coefficient * v` in FGMRES modified
Gram–Schmidt into one Triton kernel on CUDA. Dot products, their ordering,
normalization, Givens rotations, convergence tolerances and timestep policy
are unchanged. FP fusion is disabled to retain eager multiply-then-subtract
rounding. CPU, unsupported types and gradient-bearing inputs use eager PyTorch.
The kernel allocates a new output; it does not mutate Krylov basis storage.

## Protocol

RTX 3090 selected by UUID. Two sequential pairs, AB then BA, on each window:
0–115 days and 2431–2521 days (90 days) from the identical historical checkpoint.
The source copies differ only in `solver/fgmres.py` and `solver/krylov_cuda.py`.
The recorded time is the sum of step calls, excluding deck/grid initialization,
imports, audits and output. No profiler or concurrent GPU tests were used.
First kernel compilation, if needed, remains included in the candidate time.

Initial runs use `validate_norne.py`, including cumulative component balance,
strict CNV/MB and source/input integrity audits. Continuations use
`probe_norne_timestep.py` with strict convergence checks, but no independent
cumulative component-balance audit. Their checkpoint is day 2431 of an earlier
run, not a fresh full-history calculation. All saved summary/field arrays and
attempted-step clocks, outcomes and Newton counts are compared exactly.

39 focused CPU/CUDA tests passed before the runs. These cover exact vector
updates, full restarted FGMRES trajectory and true residual, Givens rotations,
and basis storage. This is numerical regression testing, not external physics
validation. These are reservoir/well runs, not a coupled surface network.

See `results.json` for all paired observations and run subdirectories for
original manifests, numerical results and statuses. `change.patch` is relative
to the pre-experiment workspace, not necessarily Git HEAD. The archived launcher
expects repository-local deck/checkpoint files and fresh scratch output paths.

## Observed result and scope

Early-window median step-call time: **12.9295 → 12.3480 s**, a 4.50%
reduction. Both individual pairs improved (3.91% and 5.05%). Late-window
median: **18.5431 → 18.7775 s**, 1.26% slower, with mixed pair signs
(3.19% slower and 0.62% faster). Thus no general/full-Norne speedup is
established. The kernel is retained as a local, numerically exact change;
late-window performance remains an explicit limitation. No full-history
runtime is extrapolated from these observations.

All four comparisons have identical saved fields/SUMMARY and work records,
17 accepted steps and no failures per window. Initial runs passed balance
and integrity audits. Late source hashes were checked again after all runs.
