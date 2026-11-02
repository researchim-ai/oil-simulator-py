"""Synthetic parameter recovery and rejection of unidentifiable data."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import PropertyModelError
from surface.calibration import fit_drift_flux


def test_exact_recovery_and_holdout_prediction():
    x = np.linspace(-1, 3, 15)
    fit = fit_drift_flux(x, 1.2 * x + 0.3, np.full_like(x, 0.01))
    assert fit.distribution_parameter == pytest.approx(1.2, abs=1e-12)
    assert fit.drift_velocity_m_s == pytest.approx(0.3, abs=1e-12)
    assert fit.predict([0.5, 2.5]) == pytest.approx([0.9, 3.3])
    assert fit.rms_error_m_s < 1e-12
    with pytest.raises(PropertyModelError, match="range"):
        fit.predict(4)


def test_weighted_measurements_and_active_parameter_bound():
    x = np.arange(5, dtype=float)
    y = 1.1 * x + 0.2
    y[-1] += 2
    sigma = np.array([0.01, 0.01, 0.01, 0.01, 100])
    fit = fit_drift_flux(x, y, sigma)
    assert fit.distribution_parameter == pytest.approx(1.1, abs=1e-6)
    bound = fit_drift_flux(x, 3 * x + 0.2, np.ones(5))
    assert bound.distribution_parameter == pytest.approx(2)
    assert bound.active_bounds[0] == 1


def test_insufficient_or_degenerate_observations_fail():
    with pytest.raises(PropertyModelError, match="identifiable"):
        fit_drift_flux([1, 1, 1], [1, 2, 3], [0.1] * 3)
    with pytest.raises(ValueError):
        fit_drift_flux([0, 1, 2], [1, 2, 3], [0.1, 0, 0.1])
    with pytest.raises(ValueError):
        fit_drift_flux([0, 1], [1, 2], [1, 1])
