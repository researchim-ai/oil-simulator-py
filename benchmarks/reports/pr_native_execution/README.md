# Native PR kernels and embedded Radau control

This measures complete CPU pipe transients, including the initial wave and
all accepted steps. It does not measure one isolated residual or equate this
pipe with a whole coupled field. Cold initialization/compilation, integration,
and independent auditing are accounted separately. Warm numbers mean fresh
physical states with a reusable execution context, not continuation past the
startup transient.
The recorded environment is Torch 2.7.0+cu126 with one CPU thread via the
OMP/MKL settings below; these are CPU throughput figures, not GPU timings.

## Qualified standalone results

For one complete simulated second, three warm CPU repeats give:

* Two-phase pipe: existing BE **26.6449 s**, native embedded Radau **1.43679 s**
  (18.54×). Native samples are 1.41766 / 2.27900 / 1.43679 s on the shared host;
  the reported number is the median, not the best sample. Sixteen accepted
  nonlinear stage systems; zero rejections, phase fallbacks or execution
  fallbacks. Five native kernels are reused across all fresh starts.
* Single-phase BDF wave: existing execution **1.50817 s**, native **0.607262 s**
  (2.48×). Same 47 accepted steps and tolerances. Endpoint differences are
  7.92e-8 Pa / 4.55e-13 K / 4.95e-13 m/s.

Two-phase temporal refinement uses original eager derivatives and unfused full
EOS, maximum step /4 and tolerances /64. At accepted trajectory checkpoints the
maximum pressure discrepancy improves **12.4775 → 1.28492 Pa**, temperature
7.72e-5 → 7.95e-6 K, velocity 1.443e-4 → 8.385e-5 m/s. Maximum requested-field
error norm improves **0.1443 → 0.08385**; this explicitly checks startup, rather
than only the almost-equal end state. All measured and refined trajectories
pass separate NumPy EOS/FV/stage-ledger audits. Raw endpoint energy ledger
defects are at most about 1.2e-7 J; stage balances are also checked with the
stated floating-point/transfer budgets.

**Cold cost is substantial.** The first native two-phase calculation in the
process takes 18.8741 s initialization plus 43.5147 s integration/compilation
(62.3888 s total), using the local disk compiler cache. Warm initialization is
another 0.083–0.091 s beyond the 1.43679 s integration median. Single-phase
native first initialization/integration is 2.40172 + 3.60194 s. Warm throughput
is not a promise of that latency for the first project opening. Keep one
execution context for a whole schedule; geometry/phase-structure changes can
require new compilation.

Evidence: [two-phase summary](release_conservative_radau/summary.json),
[single-phase summary](release_bdf/summary.json), with all compressed
trajectories, the refined trajectory and hashes alongside. These are source
versions at measurement; the subsequently wired subcycling selector is checked
separately and does not participate in either standalone timing case.

### Numerical maximum-step budget

An additional input-driven study varies only `maximum_step_s`, retaining
the equations, initial physical state and every error tolerance. Timings here
include model loading/initial-state inspection and the complete schedule:
cap 0.1 s **1.44040 s**, cap 1 s **0.812572 s**, medians of three warm runs.
The latter uses nine accepted steps and actually chooses a largest step of
0.579708 s, rather than being forced to use a one-second step. All audits pass
with zero rejection, phase fallback or compiler fallback. Both cases reuse
the same five kernels, including across the configuration change.

Maximum normalized accepted-checkpoint error stays **0.08385 < 1** in both
cases. Raising the cap increases maximum pressure discrepancy from 1.28492
to 3.11619 Pa, temperature from 7.95e-6 to 1.93e-5 K and endpoint error norm
from 8.52e-5 to 0.01193; the maximum velocity discrepancy remains 8.385e-5 m/s.
This is a disclosed accuracy/step-policy tradeoff within the unchanged
requested budget, not a claim of identical trajectories. Each maximum field
discrepancy remains below the original BE case against the same refinement.
The initial wave remains resolved; a larger maximum is permission for the
adaptive controller, not an instruction to ignore a fast transient.

This gives faster-than-real-time **warm CPU throughput on this two-phase
standalone case**, not a real-time qualification of a whole installation or
an equal-model win over NeqSim. Source-current evidence and every trajectory:
[step-budget summary](adaptive_step_budget/summary.json).

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
TORCHINDUCTOR_CACHE_DIR=/tmp/oil_pr_native_execution \
/home/user/miniconda3/bin/python scripts/benchmark_pr_adaptive_step_budget.py \
  --request benchmarks/reports/pr_native_execution/release_conservative_radau/request.json \
  --reference benchmarks/reports/pr_native_execution/release_conservative_radau/refined.json.gz \
  --maximum-steps 0.1 1 --repeats 3 --output-dir /tmp/pr-step-budget-repeat
```

## Implementation

`NativeExecution` compiles the existing pure residual, forward dense AD or
colored JVP, and exact constrained characteristic tangents. It does not use
lookup tables, freeze phases or omit admissibility/stability/entropy checks.
Changing tensor inputs are guarded and remain live; outputs own their storage.
There are explicit kernel/node/constant-memory admission budgets. Higher AD
uses the original path. Unsupported compilation returns to original AD with
diagnostics; final Newton acceptance always evaluates the original residual.

Reverse-AD compilation produced incorrect scatter pullbacks on the installed
Torch/Inductor combination during a two-phase probe, despite an exact lowered
ATen graph. It is deliberately excluded. Forward/colored results are checked
against eager before admission. Fast-math, in-place buffers and buffer reuse
are disabled. Compilation is a real cold cost and can take minutes for new
structures; this is not a claim of instant first-run latency.

Residual protocol selection occurs outside the compiled function, avoiding
identity guards on freshly created step functions. No entire checkpoint is
captured by the collocation residual: changing wall temperatures are tensor
inputs, so a new model clock does not cause recompilation. The installed
FakeTensor forward-view metadata cache is disabled during native dispatch;
this changes compiler metadata handling, not the equations. Repeated closure
and actual multi-step two-phase tests assert zero execution fallback.

The opt-in Radau IIA control uses one nonlinear two-stage solve and a linear
filtered second-order companion estimate. It removes full-step/two-halfstep
nonlinear repetition. Inventory transfers remain the original Radau stage
quadrature. Sparse systems retain colored Jacobians, BlockJacobi and GMRES;
there is no mandatory global dense Jacobian. The original step-doubling
control remains the default. Existing restrictions on detailed equipment,
slip, populations and hydrostatic bundles are unchanged.

The embedded controller permits at most fivefold growth when its measured
error allows it; legacy control and phase fallback retain the twofold bound.
Each proposed step still solves and checks its actual error. This follows
the ordinary adaptive-integrator practice of bounding an error-based proposal
rather than imposing a long doubling ladder: for comparison, the primary
[SciPy Radau implementation](https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/radau.py)
sets a tenfold maximum factor. No SciPy integrator is used by the production path.

The refinement work also found a conservative-energy carry bug in the shared
transport residual. It previously reconstructed old internal energy from EOS,
discarding the small difference between that reconstruction and the stored
conservative total energy. The residual now subtracts old kinetic, potential
and surface energies from the checkpoint total. Thermal stocks are extracted
in one device batch, without adding per-cell GPU-to-CPU synchronization.
A regression perturbs an admissible checkpoint energy and checks the actual
physical energy equation responds by exactly that amount.

## Cases and reproduction

`two-phase-pipe`: methane/n-pentane 80/20, PR, 300 K, around 5 MPa;
10 m × 0.1 m, eight cells, two **finite 100 m³ plenums**. Initially the inlet
plenum is 1000 Pa above the pipe/export. Initial velocities are zero and both
phases must be occupied. Maximum step is 0.1 s. Both measured paths use the same
relative/temperature/velocity tolerances: 1e-4 / 0.002 K / 0.001 m/s. Baseline
is the existing backward Euler with step doubling; candidate combines exact
native execution/full-EOS fusion and embedded Radau. CPU chemical fusion is
enabled for both. This is a combined algorithm/execution change, not an
isolated attribution of all speedup to Radau.

`bdf-wave`: periodic single-phase methane PR pipe with absent pentane in its
registry, same geometry, 1000 Pa sinusoidal pressure wave, initially zero
velocity. Both paths use BDF2 and identical stricter single-phase tolerances;
the comparison isolates execution/fused-property changes.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
TORCHINDUCTOR_CACHE_DIR=/tmp/oil_pr_native_execution \
/home/user/miniconda3/bin/python scripts/benchmark_pr_native_execution.py \
  --case two-phase-pipe --integrator radau3-embedded \
  --reference-refinement 4 --repeats 3 --output-dir /tmp/pr-native-radau-repeat

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
TORCHINDUCTOR_CACHE_DIR=/tmp/oil_pr_native_execution \
/home/user/miniconda3/bin/python scripts/benchmark_pr_native_execution.py \
  --case bdf-wave --repeats 3 --output-dir /tmp/pr-native-bdf-repeat
```

Use an empty cache directory to measure a disk-cache cold compilation.
`release_*/summary.json` contains source hashes, all timings, execution
fallbacks and endpoint differences. Compressed complete trajectories and
their model inputs are retained. Timing excludes independent NumPy audits;
the saved results must pass them before a summary is published.

The refined two-phase run divides maximum step by four and all temporal
tolerances by 64. Accepted coarse checkpoints are compared against linear
interpolation of that refined trajectory, including startup. This is numerical
temporal refinement, not experimental validation or spatial convergence.

The independent collocation ledger includes a bounded sixteen-ULP allowance
from the two published stocks when subtracting large inventories. The original
absolute and relative transfer budgets remain. In the 100 m³ plenum a nominally
zero transfer differed by four ULPs (1.16e-10 mol); energy subtraction reached
about twelve ULPs and exceeded the old absolute floor. The auditor reports the raw residual
and rejects a deliberately injected 0.01 J stage error. This allowance does
not alter integration tolerances or EOS/phase acceptance.

## External comparison and limits

The previous [NeqSim experiment](../external_transient_pipes/README.md)
completed one model second in 1.55266 warm CPU seconds for an eight-cell
two-phase pipe. Its imposed reservoirs, separate phase transport/mass transfer
and experimental energy recovery differ from our finite plenums and HEM
equilibrium model. It is an engineering throughput reference, **not an
equal-physics accuracy/performance ranking**. No whole-facility, drift-flux,
film or GPU speedup is inferred from these standalone measurements.

`*_pilot` directories are exploratory evidence. Their source versions changed
during development, and early Radau results failed the original absolute-only
large-stock ledger test. They are not qualified release timing series.
`release_radau` also retains a failed qualification attempt: a compiler
forward-view-layout assertion caused fallback, and its large-stock audit
exceeded the provisional eight-ULP allowance. A directory without a qualified
`summary.json` must not be used as the final speed result.
