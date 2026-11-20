"""An incomplete release schedule must fail before launching a solver."""

import json
from pathlib import Path

import pytest
from validate_pr_integrated_facility import requested_operation


def test_full_liquid_mixture_fixture_has_the_complete_release_cycle():
    cfg = json.loads(
        (
            Path(__file__).parents[1] / "configs/surface_pr_integrated_facility_liquid_mixture.json"
        ).read_text()
    )
    assert requested_operation(cfg) == [0.2, 0.2, 0.6]
    assert all(
        p["equilibrium_liquid_mixture"]
        for p in cfg["model"]["plant"]["transport"]["pump_connections"]
    )


@pytest.mark.parametrize(
    "schedule",
    [
        [{"duration_s": 1e-4}],
        [{"duration_s": 0.1}] * 3,
        [{"duration_s": 0.2}, {"duration_s": 0.0}, {"duration_s": 1.0}],
    ],
)
def test_short_or_invalid_schedule_is_not_a_full_cycle(schedule):
    with pytest.raises(ValueError):
        requested_operation({"schedule": schedule})


def test_operating_flow_accounting_is_identical_for_local_and_monolithic_records():
    from copy import deepcopy

    from validate_pr_integrated_facility import coverage, operational_audit

    config = json.loads(
        (Path(__file__).parents[1] / "configs/surface_pr_integrated_facility.json").read_text()
    )
    names = coverage(config)["machines"]
    port_names = [p["name"] for p in config["model"]["plant"]["ports"]]

    def record(dt, flow):
        return {
            "dt_s": dt,
            "transport": {
                "pump_connections": {n: {"mass_rate_kg_s": flow} for n in names},
                "compressors": {},
            },
            "columns": {"ports": {n: {"flux": [flow, 0.0, 0.0, 0.0, 0.0]} for n in port_names}},
        }

    initial = {"plant": {"transport": {"time_s": 0.0}}}
    intervals = [
        {
            "steps": [{"plant": {"steps": [record(dt, flow)]}}],
            "state": {"plant": {"transport": {"time_s": sum([0.2, 0.2, 0.6][: i + 1])}}},
        }
        for i, (dt, flow) in enumerate(((0.2, 1.0), (0.2, 0.0), (0.6, 1.0)))
    ]
    report = {"initial": initial, "intervals": intervals}
    reference = operational_audit(config, report)
    split = deepcopy(report)
    for interval in split["intervals"]:
        control = interval["steps"][0]
        slow = {"operator": "slow", "report": {"plant": control.pop("plant")}}
        control["local_subcycling"] = {
            "stages": [{"operator": "fast", "report": {}}, slow, {"operator": "fast", "report": {}}]
        }
    config["model"]["model"] = "pr_subcycled_surface_v1"
    assert operational_audit(config, split) == reference
    split["intervals"][0]["steps"][0]["local_subcycling"]["stages"].reverse()
    # Symmetric labels alone are insufficient to identify a swapped slow stage.
    stages = split["intervals"][0]["steps"][0]["local_subcycling"]["stages"]
    stages[0], stages[1] = stages[1], stages[0]
    with pytest.raises(ValueError, match="reordered"):
        operational_audit(config, split)
