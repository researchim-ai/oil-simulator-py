# Rejected transport-only flash experiment

Not promoted. Production source was restored before implementing the separate
pipe sensitivity/port reuse change. The archived `experiment.patch` and source
archive are experimental, not the current production implementation.

Skipping unused phase-mass reporting tensors in internal quadrature preserved
values, first/second derivatives and full compared trajectories. A 128-stream x
512-point CUDA microcase reduced peak additional tensor allocation from
25,768,448 to 24,195,584 bytes (6.1%). However, four full-month GPU pairs gave
only 1.7% lower combined median with two pairs showing no improvement. The
strong CPU pair was 21.559 → 22.268 s. An isolated property/pressure-derivative
microcase showed a small CPU benefit and no CUDA benefit. These observations
do not justify promoting the extra API/path for acceleration.

Configurations, audited trajectories, tests, timing samples and memory samples
are retained. No tolerances were relaxed; the experiment is not evidence of a
full-model speedup or a new physical validation.
