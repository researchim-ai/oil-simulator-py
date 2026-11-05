"""No unresolved ULP-sized tail, and no absolute-epsilon interval truncation."""

import math
from dataclasses import replace

import pytest
from test_surface_compositional_separator import binary

from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork
from surface.pr_gravity_transport import PRGravityTransportNetwork
from surface.time_interval import interval_step_s


def test_roundoff_tail_is_included_in_solved_step():
    now, end, proposed = math.nextafter(1.5e-7, -math.inf), 2e-7, 5e-8
    assert proposed < end - now
    assert interval_step_s(now, end, proposed) == end - now
    assert interval_step_s(now, end, proposed / 2) == proposed / 2


def test_small_physical_interval_is_not_hidden_by_absolute_epsilon():
    assert interval_step_s(0.0, 1e-25, 1e-26) == 1e-26
    assert interval_step_s(0.0, 1e-25, 1e-25) == 1e-25
    now = 1e9
    end = now + 1.0
    step = 100 * math.ulp(now)
    assert interval_step_s(now, end, step) == step


@pytest.mark.parametrize("coupled", [False, True])
def test_real_nested_interval_finishes_exactly_after_decimal_time_additions(coupled):
    transport = PREquilibriumTransportNetwork(
        binary(),
        [PREquilibriumNode("closed", 0.001)],
        initial_step_s=5e-8,
        minimum_step_s=1e-12,
        maximum_step_s=1e-6,
    )
    state = transport.initialize(
        {"closed": {"pressure_pa": 1e6, "temperature_k": 350.0, "mole_fractions": [1.0, 0.0]}}
    )
    state = replace(state, time_s=math.nextafter(1.5e-7, -math.inf))
    if coupled:
        model = PRGravityTransportNetwork(transport, {}, [])
        state = model.initialize(state, {})
    else:
        model = transport
    report = model.advance(state, 2e-7 - state.time_s)
    final = report["state"]["transport"] if coupled else report["state"]
    assert final["time_s"] == 2e-7
    assert len(report["steps"]) == 2
    assert math.fsum(step["dt_s"] for step in report["steps"]) == pytest.approx(5e-8, rel=1e-15)
    assert max(map(abs, report["conservation_residuals"])) < 1e-8
