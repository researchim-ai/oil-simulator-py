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
