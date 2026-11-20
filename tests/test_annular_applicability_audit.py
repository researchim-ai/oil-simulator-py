"""Conditional wetting thresholds and independent velocity sensitivity."""

import math

import pytest
from audit_annular_applicability import critical_angle
from diagnose_annular_experiment import independent_film


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 160.0, 180.0])
def test_threshold_inverts_capillary_force(angle):
    sigma = 0.0728
    force = sigma * (1 - math.cos(math.radians(angle)))
    assert critical_angle(force, sigma) == pytest.approx(angle, abs=1e-10)


def test_large_force_is_all_angles_not_nan():
    assert critical_angle(1.0, 0.0728) == 180.0


@pytest.mark.parametrize(
    "force,sigma", [(-1.0, 0.07), (float("nan"), 0.07), (1.0, 0.0), (1.0, float("inf"))]
)
def test_invalid_material_inputs_rejected(force, sigma):
    with pytest.raises(ValueError):
        critical_angle(force, sigma)


def test_reference_velocity_variant_re_solves_film_without_using_pressure_data():
    normal = independent_film(9.0, 0.08, 1.2)
    superficial = independent_film(9.0, 0.08, 1.2, shear_velocity="superficial")
    assert all(math.isfinite(x) for x in (*normal, *superficial))
    assert 0 < superficial[0] < normal[0] < 0.3
    assert superficial[1] > normal[1]
    with pytest.raises(ValueError):
        independent_film(9.0, 0.08, 1.2, shear_velocity="invented")
