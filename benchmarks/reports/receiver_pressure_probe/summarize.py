"""Compare stored physical fields after both independently audited runs."""
import json
from pathlib import Path

root = Path(__file__).parent
old = json.loads((root.parent / "receiver_newton_reuse" / "comparison.json").read_text())[-1]
new = json.loads((root / "comparison.json").read_text())[0]
assert old["accepted"] and new["accepted"]
assert old["audit"]["accepted"] and new["audit"]["accepted"]


def maximum_difference(a, b):
    if isinstance(a, list):
        assert len(a) == len(b)
        return max((maximum_difference(x, y) for x, y in zip(a, b)), default=0.)
    return abs(a - b)


summary = {
    "previous_wall_s": old["wall_s"],
    "completed_probe_wall_s": new["wall_s"],
    "observed_speedup": old["wall_s"] / new["wall_s"],
    "previous_counters": old["counters"],
    "completed_probe_counters": new["counters"],
    "reservoir_max_absolute_differences": {
        key: maximum_difference(old["reservoir"][key], new["reservoir"][key])
        for key in old["reservoir"]
    },
    "receiver_differences": {},
}
summary["accepted_stage_pressure_trials"] = {
    "previous": [len(stage["pressure_trials"]) for stage in old["report"]["stages"]],
    "completed_probe": [len(stage["pressure_trials"]) for stage in new["report"]["stages"]],
}
a, b = old["state"], new["state"]
assert a["model_id"] == b["model_id"]
for key in ("time_s", "wall_temperature_k"):
    summary["receiver_differences"][key] = maximum_difference(a[key], b[key])
for key in ("component_moles", "total_energy_j", "temperature_k", "interface_heights_m"):
    summary["receiver_differences"][key] = maximum_difference(a["column"][key], b["column"][key])
assert a["column"]["phase_roots"] == b["column"]["phase_roots"]
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
