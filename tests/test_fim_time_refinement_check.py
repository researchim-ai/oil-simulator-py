"""Check the diagnostic against an independent first-order ODE example."""

import json
import math

import pytest
from check_fim_time_refinement import difference_order


def test_first_order_backward_euler_decay_has_a_resolved_decreasing_three_grid_error():
    values = [(1 + h) ** (-1 / h) for h in (0.1, 0.05, 0.025)]
    assert all(abs(v - math.exp(-1)) > 0 for v in values)
    result = difference_order(*[[v] for v in values])
    assert result["decreases"] and result["resolved"]
    json.dumps(result, allow_nan=False)
    assert 0.85 < result["observed_order"] < 1.1


def test_roundoff_plateau_is_not_reported_as_a_convergence_order():
    result = difference_order([1.0], [1.0], [1.0])
    assert not result["resolved"] and not result["decreases"]
    assert result["observed_order"] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="Finite"):
        difference_order([1.0], [2.0], [math.nan])
