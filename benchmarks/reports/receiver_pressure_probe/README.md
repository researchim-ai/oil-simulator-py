# Complete pressure residuals for local receiver coupling

## Change

Previously a locally integrated pressure trial stopped at the first feed-pressure
violation. The coupling iteration then had an intermediate-time pressure bound,
not an endpoint residual. Repeated small pressure corrections could replay much
of the same interval.

For the single-vessel subcycling path, the driver can now finish a trial to
obtain its endpoint pressure while collecting the maximum feed-pressure bound.
Such a trial is **always rejected**, including when only a subsequently discarded
coarse step violated the bound. Its endpoint is only a predictor for a new,
independently computed trial. Component, energy, entropy, EOS and time-error
checks remain in place. No physical acceptance tolerance was widened.

If a step of an already pressure-infeasible probe fails thermodynamic or
nonlinear checks, abort it immediately. Do not adaptively halve that invalid
probe in an attempt to obtain an endpoint. The existing pressure-bound fallback
handles this case. The observer stores only the worst violation (bounded memory).

The accepted trajectory must still satisfy every pressure and physical check.
Neither the probe state nor its cumulative transfers are published. The driver
retains transactional rollback and the same outer step-doubling error check.
Normal standalone vessel integration still rejects bad feed pressure immediately.
Local subcycling remains opt-in and restricted to one gravitational receiver.

## Evidence and reproduction

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src:scripts:tests /home/user/miniconda3/bin/python \
benchmarks/reports/receiver_pressure_probe/compare.py
python benchmarks/reports/receiver_pressure_probe/summarize.py
```

The common prepared initial state, 0.05 s duration, 0.9 export opening and
all tolerances match `../receiver_newton_reuse/compare.py`. The comparison
reference is the archived optimized-Newton run there. This is a shared-host
CPU observation, not an isolated AB/BA experiment or a GPU speed guarantee.
Small regression jobs overlapped the current measurement.

`initial-interrupted.log` is an incomplete diagnostic attempt before the fast
failure guard was added. It is not a completed benchmark and has no timing result.
The completed run is `comparison.json`; `comparison.log` includes
per-probe duration, completion and rejection information. `summary.json` compares
physical receiver/reservoir fields and the number of pressure trials.

## Completed result

The 0.05 s run completed and passed the independent audit. Compared with the
archived optimized-Newton reference, CPU time was **244.339 → 170.806 s**
(30.1% less time, observed 1.431×). Nonlinear solves fell **2,218 → 1,574**;
residual evaluations **15,590 → 11,064**. Pressure trials on the two accepted
outer half intervals fell from **6/3 to 5/2**. The timing includes the discarded
outer coarse interval as well, whereas those trial counts cover accepted stages.

Final reservoir pressure, saturations and dissolved gas matched exactly.
Receiver differences: component moles 3.90e-11 mol, energy 6.21e-8 J,
temperature 7.75e-11 K. Wall temperature and interface positions matched exactly.
Both runs used identical tolerances. The complete reports and field comparisons
are retained, not just successful timing summaries.

This still takes minutes for a small example. It remains slower than the
previously archived ordinary coupled integration with short global steps;
this result does not justify enabling local subcycling by default.

## Verification

`tests/test_receiver_pressure_probe.py` checks rejection of discarded-step
violations, missing-endpoint fallback, a real low-pressure feed, and immediate
termination when another physical check fails in a pressure-infeasible trial.
Existing receiver subcycling tests check independent scalar audits and rollback.
`regressions.log` covers standalone flow/thermal vessels and coupled receiver
commands. The independent audit also runs on the completed benchmark.

## Remaining scope

This reduces wasted pressure probes; it does not replace nested backward-Euler
step doubling with a higher-order integrator or introduce a global network DAE.
It does not remove all repeated trajectories or qualify full-field performance.
