"""Identifiable weighted fitting of explicitly supplied drift-flux observations."""

from dataclasses import dataclass

import numpy as np
import torch

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
        if isinstance(mixture_velocity_m_s, torch.Tensor):
            x = mixture_velocity_m_s
            if not bool(torch.isfinite(x).all().cpu()):
                raise ValueError("Calibration queries must be finite")
            outside = (x < self.velocity_domain_m_s[0]) | (x > self.velocity_domain_m_s[1])
            if not extrapolate and bool(outside.any().cpu()):
                raise PropertyModelError("Velocity outside calibrated range")
            return self.distribution_parameter * x + self.drift_velocity_m_s
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
    inputs = (mixture_velocity_m_s, gas_velocity_m_s, standard_deviation_m_s)
    devices = {v.device for v in inputs if isinstance(v, torch.Tensor)}
    if len(devices) > 1:
        raise ValueError("Calibration tensors must share one device")
    device = next(iter(devices), torch.device("cpu"))
    x, y, s = (torch.as_tensor(v, dtype=torch.float64, device=device) for v in inputs)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or s.shape != x.shape
        or len(x) < 3
        or not bool(
            (
                torch.isfinite(x).all()
                & torch.isfinite(y).all()
                & torch.isfinite(s).all()
                & (s > 0).all()
            ).cpu()
        )
    ):
        raise ValueError("Need >=3 finite observations and positive measurement deviations")
    for bounds in (distribution_bounds, drift_bounds):
        if len(bounds) != 2 or not all(np.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("Invalid calibration bounds")
    if distribution_bounds[0] <= 0:
        raise ValueError("Distribution parameter must stay positive")
    design = torch.stack((x, torch.ones_like(x)), -1) / s[:, None]
    singular = torch.linalg.svdvals(design)
    if bool((singular[-1] <= 1e-12 * singular[0]).cpu()):
        raise PropertyModelError(
            "Drift-flux parameters are not identifiable from these observations"
        )
    target = y / s
    unconstrained = torch.linalg.lstsq(design, target).solution
    lower = x.new_tensor([distribution_bounds[0], drift_bounds[0]])
    upper = x.new_tensor([distribution_bounds[1], drift_bounds[1]])
    # For two coefficients the optimum lies in the interior or on one of
    # four box edges; edge projection includes all corners. No iteration,
    # SciPy dependency or implicit CPU copy of the observation array.
    candidates = []
    if bool(((unconstrained >= lower) & (unconstrained <= upper)).all().cpu()):
        candidates.append(unconstrained)
    for fixed in range(2):
        free = 1 - fixed
        for value in (lower[fixed], upper[fixed]):
            coordinate = (design[:, free] * (target - design[:, fixed] * value)).sum() / design[
                :, free
            ].square().sum()
            row = torch.zeros(2, dtype=x.dtype, device=device)
            row[fixed] = value
            row[free] = coordinate.clamp(lower[free], upper[free])
            candidates.append(row)
    matrix = torch.stack(candidates)
    errors = (matrix @ design.T - target).square().sum(-1)
    coefficients = matrix[errors.argmin()]
    active = torch.where(coefficients == lower, -1, torch.where(coefficients == upper, 1, 0))
    residual = coefficients[0] * x + coefficients[1] - y
    # Only the small final report crosses the device boundary.
    report = (
        torch.cat(
            (
                coefficients,
                torch.stack(
                    (
                        residual.square().mean().sqrt(),
                        residual.abs().max(),
                        x.min(),
                        x.max(),
                        singular[0] / singular[-1],
                    )
                ),
                active.to(x.dtype),
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    return DriftFluxFit(
        report[0],
        report[1],
        report[2],
        report[3],
        (report[4], report[5]),
        len(x),
        report[6],
        tuple(int(v) for v in report[7:]),
    )


def fit_validate_drift_flux(
    mixture_velocity_m_s,
    gas_velocity_m_s,
    standard_deviation_m_s,
    *,
    training_indices,
    validation_indices,
    distribution_bounds=(0.5, 2.0),
    drift_bounds=(-10.0, 10.0),
    allow_validation_extrapolation=False,
):
    """Fit only declared training observations; disclose untouched holdout error.

    No universal acceptance threshold is invented. The returned metrics must
    be assessed against the experiment's stated uncertainties and intended
    operating envelope. A successful fit is not experimental validation.
    """
    from dataclasses import asdict

    values = (mixture_velocity_m_s, gas_velocity_m_s, standard_deviation_m_s)
    devices = {v.device for v in values if isinstance(v, torch.Tensor)}
    if len(devices) > 1:
        raise ValueError("Calibration tensors must share one device")
    device = next(iter(devices), torch.device("cpu"))
    x, y, s = (torch.as_tensor(v, device=device, dtype=torch.float64) for v in values)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or s.shape != x.shape
        or not bool(
            (
                torch.isfinite(x).all()
                & torch.isfinite(y).all()
                & torch.isfinite(s).all()
                & (s > 0).all()
            ).cpu()
        )
    ):
        raise ValueError("Finite matching observations and positive uncertainties required")
    train, validation = tuple(training_indices), tuple(validation_indices)
    if (
        any(type(i) is not int for i in (*train, *validation))
        or len(train) < 3
        or not validation
        or len(set(train)) != len(train)
        or len(set(validation)) != len(validation)
        or set(train) & set(validation)
        or set(train) | set(validation) != set(range(len(x)))
    ):
        raise ValueError("Disjoint complete training/validation partition required")
    if type(allow_validation_extrapolation) is not bool:
        raise ValueError("Explicit validation extrapolation switch required")
    ti = torch.tensor(train, device=device)
    vi = torch.tensor(validation, device=device)
    fit = fit_drift_flux(x[ti], y[ti], s[ti], distribution_bounds, drift_bounds)
    error = fit.predict(x[vi], extrapolate=allow_validation_extrapolation) - y[vi]
    standardized = error / s[vi]
    summary = (
        torch.stack(
            (
                error.square().mean().sqrt(),
                error.abs().max(),
                standardized.square().mean().sqrt(),
                standardized.abs().max(),
                x[vi].min(),
                x[vi].max(),
            )
        )
        .cpu()
        .tolist()
    )
    return {
        "fit": asdict(fit),
        "training_indices": train,
        "validation_indices": validation,
        "validation_rms_error_m_s": summary[0],
        "validation_max_error_m_s": summary[1],
        "validation_standardized_rms": summary[2],
        "validation_standardized_max": summary[3],
        "validation_velocity_range_m_s": tuple(summary[4:]),
        "validation_extrapolation_allowed": allow_validation_extrapolation,
        "calibration_device": str(device),
        "qualification": "Bounded Torch weighted fit on declared training data, disjoint untouched holdout; no automatic model acceptance or field-calibration claim",
    }
