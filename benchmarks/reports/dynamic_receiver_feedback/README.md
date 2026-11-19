# Existing two-way FIM / PR receiver coupling: verification and retry control

The two-way coupling existed before this change. This work adds measured temporal-error feedback to the retry step; it does not introduce another physical solver or connect the quasi-steady operational facility to PR automatically.

## Runs

- `schedule.json` / `result.json`: existing manufactured three-cell, two-producer case, three 0.02 s intervals: export open, closed, reopened. Independent scalar audit accepted all intervals. Source hashes are embedded in the result. CPU runtime approximately 4 s per interval, not a large-model performance qualification.
- `retry_probe.py` / `retry.json`: existing physical model with every temporal tolerance explicitly divided by 100. Initial 0.02 s interval rejected at error ratio 12.481009. Retry predictor selects 0.0050950406 s instead of the driver's 0.01 s cut. Retry accepted at ratio 0.812943 with independent scalar audit. No accuracy tolerance was relaxed. This demonstrates a valid retry, not a measured overall speedup.
- An exploratory 0.2 s trial with the original open outlet hit `EquipmentOperatingError: Outlet outside low-pressure-drop/EOS domain`. It is not a successful temporal-refinement run, and the closure restriction remains enforced.

## Verification

32 tests passed, one GPU test skipped, one spatial-batch test excluded in the receiver/time-controller suite. Two additional graph pressure/balance and command-validation tests passed. Restart reproduces the complete physical report and vessel state exactly; stability-cache work counters may differ after tensor restoration.

The methane/ethane stock assay is synthetic. These results do not validate field oil/water characterization, injection, reservoir thermal dynamics, or long-term runtime. Existing multi-stage black-oil operational forecast and the dynamic PR branch remain separate selectable drivers.

Reproduce the scheduled run from repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts python scripts/run_dynamic_receiver.py benchmarks/reports/dynamic_receiver_feedback/schedule.json --output /tmp/receiver-feedback-new.json
```

The retry probe uses the existing test fixture and additionally needs `tests` on `PYTHONPATH`; it writes `/tmp/dynamic-receiver-feedback/retry.json`.
