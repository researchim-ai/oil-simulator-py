# Current external measurement challenge

The unchanged full experimental protocol completed and returned exit code 1.
This is an expected negative qualification, not an interrupted calculation.

Methane/ethane VLE: 17/17 states computed; mean pressure error 1.022%, maximum
2.629%. Fifteen pressure values and sixteen vapor-composition values are within
their archived marginal bounds. This does not imply joint statistical coverage.
Nine pure-fluid saturation comparisons against NIST correlations have maximum
relative discrepancy 1.548%; these correlations are not raw measurements.

The annular-pressure criterion still fails. MAE is 58.84/48.28 Pa under the
two predeclared dry roughness baselines; none of the 16 windows meets ±4 Pa.
FETRig remains conditional: six cases, 31 input-steady windows, but units and
CCI/LabView time alignment are not independently confirmed. Therefore the
whole-plant experimental qualification remains false.

An additional real CLI check after the exit-status repair is saved as
annular_gate_after_fix.json/.log. The standalone annular validator now returns
nonzero for a failed physical qualification, just as the aggregate validator
already did. `--report-only` explicitly allows report generation with exit zero
without changing the negative qualification in JSON. Three exit-contract tests
plus the existing experimental/conditioning tests passed (12 total).

The aggregate run predates this CLI-only edit; its recorded source hashes remain
unchanged as provenance. The post-fix annular report records the new validator hash.
No coefficients were fitted, no experimental windows removed and no budgets
relaxed. Independent numerical agreement does not remove the physical blockers.
