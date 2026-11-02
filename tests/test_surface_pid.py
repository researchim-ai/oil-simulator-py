"""Sampled PID: no derivative kick, saturation tracking, manual transfer."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import PIDController, PIDState


def controller(**kw):
    return PIDController(
        **{
            "proportional_gain": 2.0,
            "integral_gain": 1.0,
            "derivative_gain": 0.0,
            "min_output": -10.0,
            "max_output": 10.0,
            **kw,
        }
    )


def test_proportional_integral_response_and_filtered_derivative_without_setpoint_kick():
    p = controller(derivative_gain=3.0, derivative_filter_s=0.2)
    s = p.initialize(0, setpoint=1, mode="auto")
    r = p.step(s, 1, 0, 0.1)
    assert r["output"] == pytest.approx(0.1)
    assert r["derivative"] == 0
    r = p.step(PIDState(**r["state"]), 2, 0, 0.1)
    assert r["derivative"] == 0  # A setpoint step never enters D.
    r = p.step(PIDState(**r["state"]), 2, 0.3, 0.1)
    assert r["derivative"] == pytest.approx(-3.0)  # Filtered derivative = 1.


def test_manual_auto_transfer_tracks_actual_actuator_without_bump():
    p = controller()
    s = p.initialize(5, output=3)
    r = p.step(s, 20, 6, 0.1, mode="manual", manual_output=4)
    auto = p.step(PIDState(**r["state"]), 30, 7, 0.1, applied_output=2)
    assert auto["output"] == 2
    off = p.step(PIDState(**auto["state"]), 30, 7, 0.1, mode="off")
    assert off["output"] == 0
    auto = p.step(PIDState(**off["state"]), 30, 7, 0.1)
    assert auto["output"] == 0


def test_antiwindup_bounds_integral_and_tracks_external_power_limit():
    p = controller(tracking_time_s=0.2)
    s = p.initialize(0, setpoint=0, mode="auto")
    for _ in range(500):
        r = p.step(s, 100, 0, 0.1, applied_output=1.0)
        s = PIDState(**r["state"])
    assert abs(s.integral_output) < 250  # No accumulation proportional to elapsed time.
    assert r["output"] == 10
    tracked = s.integral_output
    for _ in range(100):
        r = p.step(s, 0, 0, 0.1, applied_output=0.0)
        s = PIDState(**r["state"])
    assert abs(s.integral_output) < abs(tracked)
    assert abs(r["output"]) < 0.001


def test_pid_validation_and_state_identity():
    p = controller()
    with pytest.raises(ValueError):
        p.step(replace(p, integral_gain=2).initialize(0), 1, 0, 0.1)
    with pytest.raises(ValueError):
        replace(p, tracking_time_s=0)
    with pytest.raises(ValueError):
        p.step(p.initialize(0), 1, 0, 0)
    with pytest.raises(ValueError):
        p.step(p.initialize(0), 1, 0, 0.1, mode="typo")
