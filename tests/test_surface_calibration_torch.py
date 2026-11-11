"""Exact convex calibration against a separate bounded optimizer and holdout."""

import numpy as np
import pytest
import torch
from scipy.optimize import lsq_linear

from surface.calibration import fit_drift_flux, fit_validate_drift_flux


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_weighted_box_solution_matches_independent_optimizer_on_every_bound(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(177)
    x = np.linspace(-2, 4, 24)
    sigma = np.exp(rng.normal(0, 1, len(x)))
    for c, v in [(1.2, 0.1), (-2.0, 20.0), (4.0, -30.0), (0.4, 0.2), (1.0, -50.0)]:
        y = c * x + v + rng.normal(0, 0.01, len(x))
        expected = lsq_linear(
            np.column_stack((x, np.ones_like(x))) / sigma[:, None],
            y / sigma,
            bounds=([0.5, -10.0], [2.0, 10.0]),
            tol=1e-13,
            max_iter=1000,
        )
        actual = fit_drift_flux(
            *(torch.tensor(a, device=device, dtype=torch.float64) for a in (x, y, sigma))
        )
        assert [actual.distribution_parameter, actual.drift_velocity_m_s] == pytest.approx(
            expected.x, rel=2e-9, abs=1e-9
        )
        prediction = actual.predict(torch.tensor([0.0, 1.0], device=device))
        assert prediction.device.type == device


def test_holdout_does_not_leak_into_fit_and_bad_predictions_remain_visible():
    x = np.linspace(0, 4, 9)
    y = 1.2 * x + 0.3
    sigma = np.ones(9) * 0.01
    kwargs = {"training_indices": [0, 2, 4, 6, 8], "validation_indices": [1, 3, 5, 7]}
    good = fit_validate_drift_flux(x, y, sigma, **kwargs)
    corrupted = y.copy()
    corrupted[[1, 3, 5, 7]] += 1.0
    bad = fit_validate_drift_flux(x, corrupted, sigma, **kwargs)
    assert bad["fit"] == good["fit"]
    assert good["validation_standardized_rms"] < 1e-10
    assert bad["validation_standardized_rms"] == pytest.approx(100.0)
    with pytest.raises(ValueError, match="partition"):
        fit_validate_drift_flux(
            x, y, sigma, training_indices=[0, 1, 2], validation_indices=[2, 3, 4]
        )
