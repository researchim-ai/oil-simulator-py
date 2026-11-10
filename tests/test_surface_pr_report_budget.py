"""The last allowed attempt may finish the interval; no extra attempt is charged."""

import pytest

from surface.liquid import HydraulicConvergenceError


@pytest.mark.parametrize("kind", ["axial", "column", "facility"])
def test_last_allowed_attempt_is_accepted_and_an_incomplete_interval_is_not(kind):
    if kind == "axial":
        from test_surface_pr_capillary_contact import setup

        model, state = setup()
        commands = {"initial_step_s": 1e-7}
    elif kind == "column":
        from test_surface_pr_capillary_column import setup

        model, state = setup()
        commands = {}
    else:
        from test_surface_pr_capillary_facility import setup

        model, state = setup()
        commands = {}
    snapshot = state.to_dict()
    final, report = model.advance(state, 1e-7, maximum_attempts=1, **commands)
    assert final.time_s == 1e-7
    assert len(report["steps"]) == 1 and not report["rejected"]
    with pytest.raises(HydraulicConvergenceError, match="attempt budget"):
        model.advance(state, 2e-7, maximum_attempts=1, **commands)
    assert state.to_dict() == snapshot
