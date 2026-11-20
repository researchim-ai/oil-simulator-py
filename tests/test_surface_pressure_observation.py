import math

import pytest
import torch

from surface.pressure_observation import PressureTapSpan


def span(**kw):
    data = {
        "length_m": 3.0,
        "diameter_m": 0.085,
        "inlet_area_m2": 0.02,
        "outlet_area_m2": 0.02,
        "elevation_change_m": 3.0,
        "minor_loss_coefficient": 0.32,
        "impulse_line_difference_pa": 0.0,
    }
    data.update(kw)
    return PressureTapSpan(**data)


def test_static_column_and_explicit_impulse_line_cancellation():
    rho = 1.2
    plain = span().predict(0.0, rho, 1.81e-5)
    assert float(plain["sensor_difference_pa"]) == pytest.approx(rho * 9.80665 * 3)
    corrected = span(impulse_line_difference_pa=-rho * 9.80665 * 3).predict(0.0, rho, 1.81e-5)
    assert float(corrected["sensor_difference_pa"]) == pytest.approx(0.0)


def test_bernoulli_term_uses_tap_areas_not_reference_area():
    q, rho = 0.01, 1.2
    base = span().predict(q, rho, 1.81e-5)
    small = span(outlet_area_m2=0.01).predict(q, rho, 1.81e-5)
    expected = 0.5 * rho * ((q / 0.01) ** 2 - (q / 0.02) ** 2)
    assert float(small["sensor_difference_pa"] - base["sensor_difference_pa"]) == pytest.approx(
        expected
    )


def test_wet_replacement_does_not_double_count_gravity_or_local_losses():
    model = span()
    dry = model.predict(0.05, 1.2, 1.81e-5)
    wet = model.predict(
        0.05, 1.2, 1.81e-5, distributed_friction_pa=dry["distributed_friction_pa"] + 17.0
    )
    assert float(wet["sensor_difference_pa"] - dry["sensor_difference_pa"]) == pytest.approx(17.0)


def test_zero_flow_has_finite_laminar_slope():
    q = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    value = span().predict(q, 1.2, 1.81e-5)["sensor_difference_pa"]
    value.backward()
    assert float(q.grad) == pytest.approx(128 * 1.81e-5 * 3 / (math.pi * 0.085**4))


@pytest.mark.parametrize("q", [-0.01, float("nan"), float("inf")])
def test_unmodelled_reverse_flow_and_invalid_inputs_rejected(q):
    with pytest.raises(ValueError):
        span().predict(q, 1.2, 1.81e-5)


def test_batch_matches_scalar():
    batch = span().predict(torch.tensor([0.0, 0.03, 0.06], dtype=torch.float64), 1.2, 1.81e-5)
    for i, q in enumerate([0.0, 0.03, 0.06]):
        scalar = span().predict(q, 1.2, 1.81e-5)
        for key, value in batch.items():
            assert float(value if value.ndim == 0 else value[i]) == pytest.approx(
                float(scalar[key])
            )


def test_negative_loss_and_missing_observation_definition_rejected():
    with pytest.raises(ValueError):
        span(minor_loss_coefficient=-1.0)
    with pytest.raises(ValueError):
        span().predict(0.01, 1.2, 1.81e-5, distributed_friction_pa=-1.0)


def test_imposed_local_resistance_cannot_reduce_wet_pressure_increment():
    dry = span().predict(0.05, 1.2, 1.81e-5)
    baseline = span().predict(
        0.05, 1.2, 1.81e-5, distributed_friction_pa=dry["distributed_friction_pa"] + 35.0
    )
    higher = span(minor_loss_coefficient=1.0).predict(
        0.05, 1.2, 1.81e-5, distributed_friction_pa=dry["distributed_friction_pa"] + 35.0
    )
    assert higher["sensor_difference_pa"] > baseline["sensor_difference_pa"]


def test_finite_inputs_cannot_silently_produce_nonfinite_pressure():
    with pytest.raises(ValueError, match="representable"):
        span().predict(1e200, 1.2, 1.81e-5)
