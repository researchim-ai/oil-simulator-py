# Adaptive AMG reuse across Norne timesteps (experimental)

`amg_rebuild: adaptive` is an opt-in black-oil policy. Default remains `step`.
The hierarchy may span four consecutive timestep attempts. CPR still updates
all numerical matrices every linear solve. Rebuild on structure/hierarchy change,
nonconsecutive time (including retry), dt ratio outside [0.5, 2], failed linear
acceptance or growth beyond max(20, twice the fresh-solve iteration count).
A failed reused solve reassembles the same Newton iterate and retries fresh;
Schur elimination and scaling must never be applied twice to the same matrix.
Policy is dropped at checkpoint serialization and suspended during capacity
searches, which retain their existing independent reuse scope.

## Qualification gates

The mode changes preconditioning and may change roundoff, Newton iteration
counts and adaptive timesteps. Equal output is not assumed. Before examining
paired results, use these conservative short-window screening limits:
maximum cell pressure difference 1000 Pa, saturation difference 1e-5,
maximum well-rate difference / max(1, reference curve absolute peak) <= 1e-3.
These are optimization screening limits, not new convergence tolerances or
proof of physical accuracy. CNV 1e-3 and MB 1e-7 remain unchanged. The initial
strict validator also audits cumulative component balance. The continuation
checks strict convergence but has no independent cumulative balance audit.
A full-history qualification is required before changing any default.

Paired AB/BA runs on RTX 3090: 0–115 days and day 2431–2521 from identical
historical checkpoint. Both variants have identical source; only mode differs.
Step-call timings exclude setup and audit/output. No concurrent GPU tests.

## Outcome: not qualified for default use

Two pairs: early median **12.2898 → 11.6680 s** (5.06% faster), late
**18.5527 → 18.5599 s** (no gain). Early adaptive work: 10 hierarchy builds,
59 updates, no fresh retry, 69 Newton iterations vs 68 for the control. Late:
15 builds, 87 updates, no fresh retry, 102 Newton iterations vs 101.
All windows accepted 17 steps with no failures.

Both repeats gave identical numerical differences. Early max pressure
difference **5270.39 Pa**, max saturation difference **0.000850640**; late
pressure **39.98 Pa**, saturation **0.000025496**. Worst normalized well-rate
differences: early **1.03046e-5**, late **5.06936e-7**. Both windows failed
the predeclared conservative field screen. This does not establish that one
preconditioner gives more physically accurate results; a tighter reference
is needed to determine discretization/nonlinear error. Initial balance and
strict convergence audits passed.

The policy is retained as explicit experimental opt-in only. Default remains
`step`. Full 3312-day qualification was not run because the short-window
gates already failed. Numerical source is identical between compared modes
and matches production; late source hashes were rechecked after runs.
After benchmarks, the run launcher gained explicit propagation of a requested
AMG mode on resume and an effective-mode metadata field (fresh benchmark
execution is unaffected). The exact benchmark launcher versions are archived.
