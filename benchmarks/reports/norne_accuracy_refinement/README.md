# Norne: numerical reference refinement for AMG comparison

Purpose: compare the ordinary and adaptive AMG modes to a more tightly
converged solution, rather than assuming the ordinary mode is exact.
Production solver tolerances/defaults are not changed by this experiment.

## Controlled quantities

All runs use the same accepted 17-step grid per window from the earlier
control: 0–115 days and day 2431–2521. Both original modes used identical
step clocks. The new runner prescribes those clocks, rejects any failed step
instead of silently subdividing, exports float64 fields, checks finite and
physical saturations, and audits cumulative component balance relative to the
window's initial masses and cumulative-well counters. Source, launcher, deck
INCLUDE files, checkpoint and timestep grid are hashed; hashes are checked
again after successful completion. RTX 3090 is selected explicitly.

The new ordinary control reproduces all early stored field snapshots at their
original float32 precision, and late final float64 fields exactly. This rules
out the fixed-grid driver as the source of previously observed mode differences.

## Failed stronger references

An initial experiment tightened both reservoir and well criteria. On the first
day the maximum well-rate residual stagnated near 2.5e-7 while the requested
criterion was 1e-7; reservoir CNV was already around 1e-8. That run failed and
is not a reference. Subsequent runs hold the original well criteria fixed.

With CNV=1e-5, MB=1e-9 and tightened linear solves, the early window failed on
day 41 with dt=10 days: oil MB plateaued at 1.164e-9 (above 1e-9), despite CNV
below 1e-5 and a converged linear system. Late CNV=1e-5/MB=1e-9 failed at day
2510; gas CNV at termination was 0.01088. Late CNV=1e-6/MB=1e-10 failed on day
2436.2 with oil MB about 1.290e-10. These are nonconvergence observations,
not a claim that the underlying root cause is known or that either AMG mode
has more accurate physics.

A second refinement ladder uses CNV=1e-4 then 1e-5, with MB=1e-8 and fixed
well criteria. Both use linear max/min tolerances 1e-8/1e-9 and true-residual
acceptance 1e-7. This is still stricter than ordinary CNV=1e-3/MB=1e-7.
It measures reservoir algebraic refinement only, conditional on unchanged
well tolerances, mesh, timestep grid, PVT and phase-switching model.

No result here validates spatial or temporal discretization against an exact
solution, experimental data, Eclipse or OPM. Differences between two refined
runs estimate reference stability; they are not rigorous error bounds.

## Early-window comparison (conditional on fixed well criteria)

The moderate CNV ladder completed all 17 steps at both levels. Against the
CNV=1e-5 / MB=1e-8 solution, ordinary/adaptive max pressure differences were
**5310.36 / 134.96 Pa**; pore-volume RMS differences **46.11 / 15.15 Pa**.
Maximum gas-saturation differences were **0.000173891 / 0.000877589**;
pore-volume RMS **1.68163e-6 / 2.95309e-6**. Thus adaptive is closer in pressure,
but farther in gas saturation. Ordinary mode is not an exact reference.

The two refined references differ by up to **55.0 Pa** in pressure and
**0.0000537217** in gas saturation. Water-saturation reference variation is
**0.000131905**, comparable to the ordinary/adaptive water errors; that metric
cannot support a reliable ranking. These observations do not prove monotonic
convergence or an absolute error bound.

Worst well-rate error normalized by max(1, the reference curve peak) was
4.38053e-6 for ordinary and 8.20869e-6 for adaptive. All completed runs passed
the component-balance and strict requested-convergence checks. Refinement
solve-call times were 14.37 and 15.49 s vs 12.55 / 11.88 s for ordinary/adaptive;
these single instrumented runs are not a new speed benchmark.

## Late-window limitation

Both moderate references also failed at day 2510 on the original prescribed
9.66368-day step. Increasing the Newton budget to 80 did not resolve the
coarser reference: gas CNV alternated approximately 0.0109 and 0.0667 while
mass balance remained small. Increasing an iteration budget alone is not a
remedy for this oscillation. A converged whole-window refined reference is
therefore unavailable; no late-window accuracy winner is declared. Resolving
this nonlinear convergence limit (or undertaking a separately labelled
common timestep-refinement experiment) is necessary before a stronger claim.
The default solver and its tolerances remain unchanged.
