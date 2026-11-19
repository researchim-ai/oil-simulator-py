# Adaptive operational forecast: qualified numerical refinement

## Model and result

The existing black-oil FIM, physical gathering tree and quasi-steady
`SurfaceFacility` were reused. The synthetic model has 1,024 reservoir cells,
16 wells, separator stages, a water pump/network, gas compression/export and
capacity limits. Gas export opening changes on day 10 and restores on day 20.
This is a 30-day operational forecast, not resolved transient PR vessels/pipes.

The selected configuration is
`configs/fim_blackoil_facility_depletion_adaptive.json`. It uses the existing
endpoint-rate controller with relative tolerance 0.005, startup limit 0.5 h,
and maximum timestep 0.5 day. Initial reservoir dt remains 4.32 s. The rate
indicator is a heuristic; the qualification below is essential.

- Candidate: **105.366 s CPU**, 240 accepted steps, 983 reservoir evaluations.
- Reference: **153.371 s CPU**, 454 accepted steps, 1,647 reservoir evaluations.
- Actual candidate dt: median **2.632 h**, maximum **6.540 h**.
- **96.95%** of simulated time was covered by steps longer than two hours.
- Both forecasts land exactly on the day-10 and day-20 event boundaries.
- Same physical model, PVT, Newton tolerances, source versions and budgets.

The reference halves maximum dt, startup rate-controller limit, and BOTH rate
indicator tolerances. Lowering only maximum dt is insufficient when that cap
never binds. The release checker now rejects unchanged/weakened adaptive
criteria, changed growth policies and different controller families, and still
requires an actual increase in accepted temporal resolution.

## Accuracy gate

The pre-existing `configs/fim_facility_forecast_budgets.json` was unchanged.
All independent conservation/control checks and numerical budgets passed:

- Maximum final reservoir pressure difference: **229.222 Pa**, budget 1,000 Pa.
- Maximum final water/gas saturation differences: **2.192e-5 / 8.631e-5**,
  each budget 2e-4.
- Total component withdrawal relative difference: **0.1040%**, budget 0.2%.
- Maximum well BHP difference: **261.330 Pa**, budget 1,000 Pa.
- Scaled well component-rate difference: **0.1725%**, budget 0.2%.
- Maximum relative cumulative well mass difference: **0.1674%**, budget 0.2%.
- Well-mode mismatches: **zero**.

These are differences between discretizations, not error against an exact
solution or an experimental calibration. Timings are shared-host CPU observations
with one Torch thread. This is not a Norne/GPU or arbitrary-network claim.
Quasi-steady modelling omits vessel inventories, waves and detailed control
transients; its applicability must be assessed for the engineering task.

## Artifacts and rejected candidates

The accepted candidate was reused from
`../operational_adaptive_forecast_fast/refined`; the additional reference is
`refined/` here. Their trajectory checksums and source/PVT guards are checked
by the existing qualification tools. `finish.py` reconstructs the final gate
and timestep statistics without rerunning the already completed candidate.

The relative-0.01 candidate completed in 51.030 s but FAILED the unchanged
accuracy gate: maximum cumulative well mass difference 0.3759%. Its full result
is retained in `../operational_adaptive_forecast_fast`; it is not the selected
configuration. The relative-0.001 pilot in `../operational_adaptive_forecast`
was interrupted after excessive early-time work; it is not a completed forecast.

## Reproduce

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src:scripts python scripts/validate_operational_forecast.py \
  --config configs/fim_blackoil_facility_depletion_adaptive.json \
  --days 30 --output /tmp/adaptive-operational-validation
```

A fresh output directory is required. The command runs both discretizations
and exits unsuccessfully if qualification fails. The existing fixed-cap profile
remains available and unchanged. No commits or pushes were performed.

47 tests passed: operational rate-refinement safeguards, existing refinement
and engineering gates, rate-control rejection/rollback and scheduled events.
