# Norne: shared PVT row lookup

The experimental reservoir `DeckPVT` oil and wet-gas branches shared the discrete
row interval lookup and table abscissae for inverse formation-volume factor
and inverse (formation-volume factor × viscosity). Each property retains
its own interpolation arithmetic, preserving forward/reverse AD. No physical
model, convergence tolerance, time-step policy, or persistent CPR setting changes.

## Decision: rejected for production

No repeatable speed benefit. Median step-call times: early **13.602 →
13.778 s**, late **19.589 → 19.837 s** (both about 1.3% slower).
The production source was restored byte-for-byte to its pre-experiment version.
The candidate and its tests are archived, not enabled. Exact numerical
equivalence held in all four comparisons. No full Norne run was justified.

## Qualification

RTX 3090 selected explicitly by UUID; float64. Two sequential pairs in
reference/optimized and optimized/reference order, with no profiler or
concurrent test process. Solver time is the sum of recorded step calls;
imports, deck parsing, initialization and output/audits are excluded.

- Initial window: 0–115 days, the existing report endpoints around 100 days.
- Late continuation: 2431–2521 days, 90 days, from the identical historical
  checkpoint `results/norne_gpu_packets_3090_20261031/checkpoint_day_2431.pt`.
- Both use strict CNV ≤ 1e-3 and MB ≤ 1e-7. Initial runs additionally audit
  cumulative component balance and verify source/input hashes.
- Continuation is a paired regression, not a newly audited full-history run;
  its runner checks strict convergence but has no independent cumulative
  component balance audit. Exact final float64 fields and summary agreement
  do not establish external physical validation.
- Work comparison covers every attempted step's start, length, success and
  Newton count. Saved fields and all SUMMARY arrays are compared exactly.

`results.json` contains the measurements. Subdirectories retain the original
run manifests, statuses, work records and numerical outputs. The archived
reference source and `change.patch` identify the sole production change.
The launcher is archived as `run_pair.py.txt`; it expects the repository root
and existing Norne deck/checkpoint, creates frozen sources under
`.scratch/norne_windows`, and must use fresh output directories.

11 focused PVT tests passed on CPU/CUDA, including exact interpolation,
forward/reverse Jacobians and vmapped JVPs at knots and extrapolation points.

Preliminary profiled runs were used only to locate the hotspot. A preliminary
late candidate overlapped a short CUDA test and is excluded from measurements.
No full 3312-day runtime or surface-network speedup is inferred here.
