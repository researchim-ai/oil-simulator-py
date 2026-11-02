"""Identifiable weighted fitting of explicitly supplied drift-flux observations."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .liquid import PropertyModelError


@dataclass(frozen=True)
class DriftFluxFit:
    distribution_parameter: float
    drift_velocity_m_s: float
    rms_error_m_s: float
    max_error_m_s: float
    velocity_domain_m_s: tuple
    observations: int
    design_condition_number: float
    active_bounds: tuple

    def predict(self, mixture_velocity_m_s, extrapolate=False):
        x = np.asarray(mixture_velocity_m_s, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Calibration queries must be finite")
        if not extrapolate and (
            np.any(x < self.velocity_domain_m_s[0]) or np.any(x > self.velocity_domain_m_s[1])
        ):
            raise PropertyModelError("Velocity outside calibrated range")
        return self.distribution_parameter * x + self.drift_velocity_m_s


def fit_drift_flux(
    mixture_velocity_m_s,
    gas_velocity_m_s,
    standard_deviation_m_s,
    distribution_bounds=(0.5, 2.0),
    drift_bounds=(-10.0, 10.0),
):
    x, y, s = (
        np.asarray(a, dtype=float)
        for a in (mixture_velocity_m_s, gas_velocity_m_s, standard_deviation_m_s)
    )
    if (
        x.ndim != 1
        or y.shape != x.shape
        or s.shape != x.shape
        or len(x) < 3
        or not all(np.isfinite(a).all() for a in (x, y, s))
        or np.any(s <= 0)
    ):
        raise ValueError("Need >=3 finite observations and positive measurement deviations")
    for b in (distribution_bounds, drift_bounds):
        if len(b) != 2 or not np.isfinite(b).all() or b[0] >= b[1]:
            raise ValueError("Invalid calibration bounds")
    if distribution_bounds[0] <= 0:
        raise ValueError("Distribution parameter must stay positive")
    design = np.column_stack([x, np.ones_like(x)]) / s[:, None]
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= 1e-12 * singular[0]:
        raise PropertyModelError(
            "Drift-flux parameters are not identifiable from these observations"
        )
    result = lsq_linear(
        design,
        y / s,
        bounds=(
            [distribution_bounds[0], drift_bounds[0]],
            [distribution_bounds[1], drift_bounds[1]],
        ),
        tol=1e-12,
        max_iter=200,
    )
    if not result.success:
        raise PropertyModelError("Drift-flux calibration did not converge")
    residual = result.x[0] * x + result.x[1] - y
    return DriftFluxFit(
        float(result.x[0]),
        float(result.x[1]),
        float(np.sqrt(np.mean(residual**2))),
        float(np.max(abs(residual))),
        (float(x.min()), float(x.max())),
        len(x),
        float(singular[0] / singular[-1]),
        tuple(map(int, result.active_mask)),
    )
