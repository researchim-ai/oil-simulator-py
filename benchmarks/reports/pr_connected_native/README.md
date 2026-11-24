# Exact native execution on the connected PR facility

Follow-up source-bound preparation telemetry, private replay storage and the
input-view layout repair are documented in
[the preparation/replay report](../pr_kernel_preparation/README.md). This
directory retains the earlier measurement and full-cycle qualification at
their recorded source hashes; the follow-up does not silently replace them.

## Completed paired operating interval

This is the existing 16-node subcycled facility with a gravitational separator,
two pump connections, a compressor and six control channels. It contains the
original 0.2 m long, 1 mm bore gas recycle. Geometry, physical inventories,
commands and state/time tolerances are unchanged between measured policies.

The supplied schedule consists of three complete reporting intervals, totaling
**0.04864483384 s**, starting from the existing checkpoint at **0.70623809959 s**.
This is an operating segment, **not a cold startup or full operating cycle**.

Three warm fresh-start measurements including model loading/inspection:

* Existing backward Euler: 18.0733 / 18.1643 / 18.2205 CPU seconds;
  median **18.1643 s**.
* Embedded Radau fast dynamics plus native execution:
  11.7743 / 7.42759 / 7.16791 CPU seconds; median **7.42759 s**.
* Observed whole-schedule speedup: **2.4455×**. This is much smaller than the
  standalone pipe speedup and **is not real-time facility operation**.

The native series keeps twelve compiled kernels, without compiler fallback,
disabled execution or phase fallback. Accepted fast records fall from 24 to
12; six accepted macro halfsteps remain. All six slow stages and all fast
stage quadratures pass the separate NumPy physical audit. Final component
ledger residuals are at most 1.02e-10 mol; energy is 1.12e-7 J. The state
difference at the three matching reporting times is 0.02493 / 0.05650 / 0.08726
in the original requested error norm. This is agreement within that budget,
not experimental validation or proof that BE is an exact solution.

Cold preparation remains expensive: the first complete native calculation is
**399.19 s**, including model inspection and native/property preparation;
recorded native compilation accounts for 279.70 s. The first BE run is 22.12 s.
The warm throughput assumes one execution context retained for the schedule
and repeated project evaluations. Measurements use Torch 2.7.0+cu126, CPU,
OMP/MKL thread counts one, on a shared host. Audits are outside timing.

[Qualified results and source hashes](steady_release/summary.json),
with every complete trajectory and exact requests beside the summary.
The subsequently added cancellation propagation affects exceptional exits,
not any equations or the measured successful execution path.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
TORCHINDUCTOR_CACHE_DIR=/tmp/oil_pr_native_execution \
/home/user/miniconda3/bin/python scripts/benchmark_pr_fast_integrator.py \
  benchmarks/reports/pr_connected_native/steady_embedded.json \
  --schemes backward_euler radau3-embedded --native-kernels \
  --maximum-native-kernels 16 --compiled-cpu-chemical --compiled-full-eos \
  --reverse-dense-jacobian --repeats 3 --output-dir /tmp/pr-connected-repeat
```

## Problems found and corrected

The original execution wrapper pooled unrelated Jacobian functions into one
Dynamo guard cache. On the facility it hit the recompile limit and disabled
all acceleration. The bounded cache now separates operator code, role and
shape using distinct tracing frames. Unsupported/admission-limited operators
use original AD and are negatively cached; validated unrelated operators stay
active. A genuinely wrong numerical result still disables native execution.
Both operator contexts and admitted native kernels have explicit limits.

Gravitational sampling captured a Python height and the complete separator
state. Height is now a changing tensor input and only the phase-root tag is
captured, instead of the checkpoint. Twelve samples across four elevations
and three temperatures reuse one Jacobian kernel and match original values.

A focused 268-unknown Jacobian reproduction found **identical** original AD,
Dynamo and lowered ATen results. Native code differed by 2.741e-9 in one
entry near −11.912, with matrix scale 28496.996: 9.62e-14 of that scale.
This was a strict verification threshold crossing from roundoff, not the
earlier large reverse-AD miscompile. Dense linearization verification retains
the existing entrywise 3e-10 + 2e-10*abs(reference) criterion, adding only
64*FP64_epsilon*row_maximum for cancellation. Residual/state/physical
acceptance tolerances remain unchanged; final Newton acceptance is original
eager computation. Tests reject substantial tangent errors, NaNs and errors
in small rows. Only one verification in the successful series needs this
additional allowance.

[Numerical diagnosis](jacobian_diagnosis.json),
[saved matrices](coupled_jacobian.pt.gz),
[focused reproduction source](reproduce_jacobian.py.txt).

## Full original cold operating cycle

`cold_cycle_embedded.json` preserves the initial time **zero**, every physical
stock, geometry and command of
`configs/surface_pr_integrated_facility_subcycled.json`. Only the fast
integrator selector and corresponding enclosing checkpoint identifier change.
[Input identity check](cycle_identity.json) records both input hashes.
The original schedule is 0.2 s startup, 0.2 s isolation, 0.6 s restart.

The **complete original cycle now passes** both the independent physical audit
and the operational audit. The joined trajectory reaches exactly **1.0 s**:
startup 0–0.2 s, isolation 0.2–0.4 s, restart 0.4–1.0 s. All three machines
and all five separator ports transfer material during startup and restart;
their isolated transfers are exactly zero. The audit checks 270 split/slow
steps and 550 local steps. Final component residuals are at most
**3.79e-10 mol**, with energy residual **5.03e-7 J**. Sources remain unchanged,
with no explicit solver migration, native fallback or numerical disable.

This qualification required two processes: the first exhausted its 900 s
budget at 0.4 s; the second resumed the accepted state and finished in
556.81 s. Their accumulated solver wall time is **1456.83 s (24.28 min)**,
excluding independent audits and the gap between processes. Recorded native
compilation totals 608.63 s; tracing, model preparation, verification and
integration also contribute. This interrupted cold execution is **not a
warm throughput benchmark** or evidence of real-time performance.
At restart the accepted step briefly falls to 9.55 microseconds, then grows
back to about 16 ms. There is no persistent microstep stall in this cycle.

[Full-cycle qualification](cold_cycle_restart.json) records audits, machine
and aperture throughputs, execution statistics and exact source hashes.
[Condensed cycle evidence](cold_cycle_summary.json) includes the step history
and timing scope. The remaining performance work is cold compilation and the
cost of the coupled slow solves; the standalone pipe timing does not remove
these costs.

The journaled driver supports a bounded native context, fused properties and
separate deadline propagation. Accepted physical steps are retained even if
the process fails or exceeds its wall-time budget; a recovered prefix must
pass its own independent audit and does not qualify the whole cycle.
`cold_cycle.json` is completion metadata, and `cold_cycle.accepted.jsonl`
contains durable accepted work. Use its actual completion/physical/operating
audit flags, never infer completion from a running log or this steady-segment
benchmark.

Large journals and raw trajectories are retained locally but excluded from Git
in this directory. Their lossless `.gz` archives are the portable evidence;
`archives.json` records compressed and original SHA-256 hashes. Restore any
required raw file with `gzip -dk <file>.gz` before replaying/auditing it.
The final full-cycle trajectory archive contains all three operating stages.
[Regression checks](regression_checks.json): 31 CPU tests and one CUDA
stratified-column test pass; these are correctness checks, not GPU speed data.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
TORCHINDUCTOR_CACHE_DIR=/tmp/oil_pr_native_execution \
/home/user/miniconda3/bin/python scripts/run_pr_checkpointed_facility.py \
  benchmarks/reports/pr_connected_native/cold_cycle_embedded.json \
  --native-kernels --maximum-native-kernels 32 --compiled-cpu-chemical \
  --compiled-full-eos --reverse-dense-jacobian --budget-s 900 \
  --output /tmp/pr-cold-cycle.json
```

Early shared-cache and `steady_pair` observations are diagnosis only: the
former disabled acceleration, and the latter failed native verification then
its source-stability check during development. They are not qualified speedup
evidence. The accepted numerical policies remain opt-in; no Norne/reservoir,
GPU throughput or competing full-facility speedup is established here.
