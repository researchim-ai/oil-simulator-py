"""Certified cancellation noise cannot mask descent of a stricter equation."""

import math

import pytest
import torch

from surface.torch_nonlinear import TorchNewton


@pytest.mark.parametrize("certified", [True, False])
def test_line_search_respects_each_rows_explicit_budget(certified):
    floor, width = 1e-10, 1e-11
    start = torch.zeros(2, dtype=torch.float64)

    def residual(point):
        # There is an irreducible, explicitly certified defect in the first
        # equation. Reaching the exact root of the second increases that
        # defect while staying within its own allowed arithmetic scale.
        return torch.stack((floor * (2 - torch.cos(point[1] / width)), point[1] - math.pi * width))

    outputs = []
    for weighted in (False, True):
        solver = TorchNewton("cpu", maximum_iterations=40)
        solver._roundoff_line_search = weighted
        bounds = start.new_tensor([4 * floor if certified else 0.0, 0.0])
        result = solver.solve(
            residual,
            start,
            start.new_full((2,), -1.0),
            start.new_full((2,), 1.0),
            tolerance=3e-12,
            stagnation_tolerance=1e-11,
            roundoff_tolerance=lambda bounds=bounds: bounds,
        )
        outputs.append(result)
    assert bool(outputs[1][2]) is certified
    if certified:
        assert not bool(outputs[0][2])
        assert abs(float(outputs[1][1][1])) <= 1e-11
        assert float(outputs[1][1][0]) <= 4 * floor


@pytest.mark.parametrize("enabled,closed", [(True, False), (True, True), (False, False)])
def test_single_liquid_pump_only_certifies_live_enthalpy_rows(enabled, closed):
    from test_surface_pr_pump_connection import setup

    from surface.pr_pump_connection import PumpConnectionBundle

    model, state, _ = setup(backpressure=1.1e6 if closed else 1e6)
    openings = {**model.default_openings, "booster": enabled}
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, *_ = model._problem(state, seeds, 1e-5, openings, [0.0, 0.0])
    bundle = evaluate.pump_connections
    offset = bundle.entries[0][3]
    with torch.no_grad():
        original_row, original = evaluate(y)
        bundle._enthalpy_roundoff = False
        legacy_row, legacy = evaluate(y)
    torch.testing.assert_close(original_row, legacy_row, rtol=0, atol=0)
    bounds = original["roundoff_bounds"][offset : offset + PumpConnectionBundle.count]
    assert bool((bounds <= 1e-10).all())
    assert float(bounds[1]) == float(bounds[4]) == 0
    assert not bool(legacy["roundoff_bounds"].any())
    if not enabled:
        assert not bool(bounds.any())
    else:
        assert float(bounds[0]) > 0
        valve_closed = bool(original["pump_connections"]["booster"]["valve_closed"])
        assert valve_closed is closed
        assert bool((bounds[2:4] > 0).all()) is (not valve_closed)
