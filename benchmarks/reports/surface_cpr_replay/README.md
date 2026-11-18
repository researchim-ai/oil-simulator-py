# Solver-owned CPR graph replay

Opt-in `simulation.cpr_reuse_graph=true`; production default remains false.
All current matrix values, pressure weights/scaling, inverse diagonal, AMG
operators, prolongation/restriction, equilibration and Chebyshev coefficients
are copied into private graph buffers. Sparse index contents are checked before
reuse. Structural changes rebuild one retained graph. Unsupported hierarchies,
input budget overflow and capture errors use the existing CPR path. Checkpoints
do not retain graph handles. Different streams do not share the replay.

## Measured scope

65,536 cells, 64 wells, coupled synthetic gathering/facility model, 30 days,
reports every 5 days, operating events at days 10 and 20. RTX 3090 Ti selected
by UUID, allocator fraction 0.12. Shared machine: these are observations, not a
performance guarantee or a Norne benchmark.

Two opposite-order pairs: 24.435 → 22.294 s and 22.125 → 20.877 s.
Median 23.280 → 21.585 s, about 7.3% less elapsed solve time (1.079x).
Both versions: 29 FIM evaluations, 70 linear solves, 9 accepted steps, no
rejected steps. Graph captures 34 → 16, eager CPR calls 70 → 16;
new workspace: 54 refreshes, 112 replays, zero fallbacks.
Owned input buffers: 61,662,096 bytes. The 128 MiB default input budget excludes
graph scratch memory, allocator reservation and CUDA context.

Final fields, accepted states, surface results, material accounts and withdrawal
match exactly. Intermediate trial certificates differ at floating-point
roundoff: first month pair maximum relative difference 8.49e-16. The original
strict comparator rejected the initial five-day pair on these differences;
that failure is retained in `pairs/` and `pairs.log`, not reclassified as a
bitwise pass. The explicit `compare_roundoff.py` checks every trial-trace entry
with a 2e-14 relative gate, then requires exact equality for accepted physical
states using the original comparator. Independent material audits and original
engineering budgets pass both month pairs. No physical or convergence tolerance
was loosened. Five-day timing alone showed no benefit (7.407 → 7.547 s).

## Evidence and limits

`month/` and `month_repeat/` contain source-guarded protocols, summaries,
trajectories, full logs, engineering comparisons and exact-state checks.
`run_counted.py SOURCE_ROOT ...forecast arguments...` instruments both versions
identically and prints CPR counters. Configuration, schedule, budgets, source
archives, incremental patch and tests are included. Run with explicit GPU UUID,
`PYTHONPATH=SOURCE_ROOT/src:SOURCE_ROOT/scripts`, and the forecast runner options
recorded in protocols; set config PVT path to the local checkout when relocating.

The final source archive additionally rejects nonfinite Chebyshev spectra
instead of silently replacing the smoother with Jacobi. This guard was added
after timings and verified in `tests_gpu_final.log`; original measured source
hashes remain in protocols. New GPU tests cover numeric refresh, coarse solve,
Chebyshev updates, sparse pattern/size changes, equilibration, output ownership,
budget/capture fallback, checkpoint isolation, stream isolation and invalid
configuration. Existing CUDA graph and FIM/surface/AMG regressions also pass.

This establishes a limited execution optimization, not independent physical
validation, long-run memory qualification, or a universal speedup. Default-on
promotion requires wider difficult-case/Norne and memory-scaling qualification.
