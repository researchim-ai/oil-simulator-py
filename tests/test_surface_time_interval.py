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


def test_odd_ulp_step_splits_without_losing_its_final_clock_quantum():
    from surface.time_interval import split_step_s

    now = 1e8
    unit = math.ulp(now)
    step = 65 * unit
    first, second = split_step_s(now, step)
    assert first != second
    assert first + second == step
    assert now + first + second == now + step
    assert interval_step_s(now, now + 1.0, 1e-5) == (now + 1e-5) - now


@pytest.mark.parametrize("coupled", [False, True])
def test_heated_physical_interval_uses_same_duration_as_large_epoch_clock(coupled):
    now = 1e8
    unit = math.ulp(now)
    duration = 129 * unit
    transport = PREquilibriumTransportNetwork(
        binary(),
        [PREquilibriumNode("closed", 0.001)],
        initial_step_s=65 * unit,
        minimum_step_s=1e-10,
        maximum_step_s=1e-5,
    )
    state = transport.initialize(
        {"closed": {"pressure_pa": 1e6, "temperature_k": 350.0, "mole_fractions": [1.0, 0.0]}}
    )
    state = replace(state, time_s=now)
    if coupled:
        model = PRGravityTransportNetwork(transport, {}, [])
        state = model.initialize(state, {})
    else:
        model = transport
    report = model.advance(
        state, duration, heat_w={"closed": 10.0}, utility_temperature_k={"closed": 400.0}
    )
    final = report["state"]["transport"] if coupled else report["state"]
    assert final["time_s"] == now + duration
    assert math.fsum(step["dt_s"] for step in report["steps"]) == duration
    assert report["total_heat_j"] == pytest.approx(10 * duration, rel=1e-14, abs=1e-18)
    assert max(map(abs, report["conservation_residuals"])) < 1e-8


def test_controlled_rotor_and_protection_timer_share_large_epoch_duration():
    from test_surface_pr_motor_control import setup

    from surface.pr_control_channels import PRSurfaceTrip

    trip = PRSurfaceTrip("delay", "motor", "speed", "booster", 90.0, 80.0, delay_s=1e-5)
    model, state, _ = setup((trip,), feedback=False)
    now = 1e8
    unit = math.ulp(now)
    duration = 129 * unit
    state = replace(
        state,
        plant=replace(state.plant, transport=replace(state.plant.transport, time_s=now)),
        next_step_s=65 * unit,
    )
    report = model.advance(state, duration, targets={"motor": 0.5}, openings={"booster": False})
    final = report["state"]
    assert final["plant"]["transport"]["time_s"] == now + duration
    assert math.fsum(step["dt_s"] for step in report["steps"]) == duration
    assert final["trips"]["delay"]["elapsed_s"] == duration
    speed = dict(final["plant"]["transport"]["rotors"])["booster"]
    expected = 100.0 + (2.0 - 0.001 * 100.0) / 0.001 * (-math.expm1(-0.001 * duration / 0.1))
    assert speed == pytest.approx(expected, abs=1e-10)
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_nonrepresentable_trip_delay_at_large_epoch_reaches_latch_without_tiny_tail():
    from test_surface_pr_motor_control import setup

    from surface.pr_control_channels import PRSurfaceTrip

    trip = PRSurfaceTrip(
        "delay",
        "motor",
        "speed",
        "booster",
        90.0,
        80.0,
        delay_s=1e-5,
        isolate_switches=("booster",),
    )
    model, state, _ = setup((trip,), feedback=False)
    now = 1e8
    state = replace(
        state, plant=replace(state.plant, transport=replace(state.plant.transport, time_s=now))
    )
    report = model.advance(state, 3e-5)
    assert report["state"]["trips"]["delay"]["latched"]
    active = next(s for s in report["steps"] if s["protections"]["delay"]["active"])
    start = active["state"]["plant"]["transport"]["time_s"] - active["dt_s"]
    assert trip.delay_s <= start - now < trip.delay_s + math.ulp(now)
    assert report["state"]["plant"]["transport"]["time_s"] == now + 3e-5
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
