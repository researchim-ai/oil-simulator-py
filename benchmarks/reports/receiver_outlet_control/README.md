# Explicit outlet commands versus autonomous control

A real FIM/PR receiver test reproduced nonzero export with `outlet_openings={"export": 0}`: the pressure regulator recomputed its own opening after the boundary adapter assigned the manual position. The level regulator had the same precedence problem. See `before.log`.

The boundary adapter now disables automatic opening computation only in the commanded trial model. The persistent vessel/controller remains unchanged; omitting the command restores automatic operation. Qualified vessel/port names behave the same way in the vessel graph. Independent scalar audits reconstruct the same declared command semantics without calling the solver's adapter. A deliberately falsified command must fail the audit.

Tests exercise closure, partial/full opening, automatic restoration, actual FIM component/energy exports, and the graph audit. This does not implement actuator travel times, PID bumpless transfer, or general controller dynamics.

## Throughflow pilot (not completed)

`interrupted_throughflow_config.json` increases the original manufactured outlet conductance from 1e-10 to 3e-6 m3/(s Pa), retains the 5% pressure-drop validity domain, and requests 0.5 s open / 0.02 s closed / 0.5 s open. It is a distinct synthetic equipment specification, not a solver optimization or calibrated field model.

The initial 0.02 s trial was rejected at temporal error 3.639; the next 0.0094355 s trial was rejected at 1.208. The driver then tried approximately 0.0047177 s. The run was deliberately interrupted to fix the independently found outlet-command defect. No completed trajectory or independent audit exists for this longer pilot. Its log is retained as diagnostic evidence only. No long-run speedup or production readiness is claimed.

## Final validation

Seven new tests passed. The receiver/network regression run passed eleven tests and exposed one legacy assertion comparing cache-work counters after restart. All physical report fields were identical. That assertion now checks exact vessel state and the entire physical report, excluding only `property_evaluations`; the affected restart/rollback test passed on rerun. Across the combined suites, eighteen distinct CPU tests passed. One GPU test was skipped and the unrelated spatial-batch benchmark was excluded. Ruff and diff whitespace checks passed.
