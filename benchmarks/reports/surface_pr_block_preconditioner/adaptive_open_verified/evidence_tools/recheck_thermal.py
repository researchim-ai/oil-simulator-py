import gzip
import hashlib
import json
import statistics
from pathlib import Path

from benchmark_pr_adaptive_preconditioner import (
    ThermalReference,
    audit_results,
    digest,
    pair_errors,
    prepare,
)

from surface import PREquilibriumTransportState

root = Path.cwd()
directory = root / "benchmarks/reports/surface_pr_block_preconditioner/adaptive"
model, state, schedule = prepare("thermal32", True)
reference = ThermalReference(model, state)
reference.solve(0.02)
certificates = {}
rows = []
pairs = {}
for path in sorted(directory.glob("thermal32_pair*.json.gz")):
    data = json.loads(gzip.decompress(path.read_bytes()))
    payload = {k: data[k] for k in ("case", "model", "initial", "schedule", "reports")}
    key = digest(payload)
    row = data["measurement"]
    assert key == row["trajectory_sha256"] == row["audit_certificate"]
    assert (
        data["model"] == model.to_dict()
        and data["initial"] == json.loads(json.dumps(state.to_dict()))
        and data["schedule"] == schedule
    )
    previous = state
    results = []
    for report in data["reports"]:
        results.append((previous, report))
        previous = PREquilibriumTransportState.from_dict(report["state"])
    assert previous.time_s == 0.02
    assert row["accepted_substeps"] == sum(len(r["steps"]) for _, r in results)
    assert row["jacobian_builds"] == sum(r["jacobian_builds"] for r in data["linear_solves"])
    if key not in certificates:
        checked = audit_results("thermal32", model, results, schedule)
        errors, _ = reference.errors(results)
        assert errors == row["errors_vs_independent_ode"]
        certificates[key] = {
            "independently_audited_steps": checked,
            "errors_vs_independent_ode": errors,
        }
    rows.append(row)
    pairs.setdefault(row["pair"], {})[row["mode"]] = results
assert len(rows) == 6 and set(pairs) == {0, 1, 2}
comparisons = [
    {"pair": i, "errors_at_report_times": pair_errors(p["diagonal"], p["block"])}
    for i, p in sorted(pairs.items())
]
medians = {
    mode: statistics.median(r["elapsed_s"] for r in rows if r["mode"] == mode)
    for mode in ("diagonal", "block")
}
report = {
    "status": "revalidated_archival_trajectories",
    "source_manifest_available": False,
    "timing_provenance_qualified": False,
    "limitation": "Original interrupted series has no completed source manifest. Timings retained as historical observations only; current-source independent re-audit does not attest to original timed source or environment.",
    "runs": rows,
    "median_elapsed_s": medians,
    "historical_speedup": medians["diagonal"] / medians["block"],
    "comparisons": comparisons,
    "audit_certificates": certificates,
    "reference_refinement_defect_k": reference.refinement_defect_k,
    "revalidation_source_sha256": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "src/surface").glob("*.py"))
    },
}
output = directory / "revalidated_thermal.json"
with output.open("x") as f:
    json.dump(report, f, indent=2, allow_nan=False)
print(
    json.dumps(
        {"medians": medians, "certificates": certificates, "qualified_original_timing": False}
    )
)
