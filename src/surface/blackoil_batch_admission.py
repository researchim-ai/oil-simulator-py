"""Validate a device-local pipe batch with one host status packet.

Retains scalar pressure/composition, PVTO coverage and moving-bubble continuity
checks. Values used only for safe lookup never relax the admission predicates.
"""

from numbers import Real

import numpy as np
import torch

from .liquid import PropertyModelError


def _tensor(value, model):
    if torch.is_tensor(value):
        return value.to(device=model.rho.device, dtype=torch.float64).detach()
    return torch.as_tensor(np.asarray(value), device=model.rho.device, dtype=torch.float64)


@torch.no_grad()
def validate_inputs(model, boundaries, rates, count):
    """Return detached float64 boundary/masses; one D2H status read per batch."""
    if not torch.is_tensor(boundaries) and any(
        isinstance(p, (bool, np.bool_)) or not isinstance(p, Real) for p in boundaries
    ):
        raise ValueError("Pressure boundaries must be real numbers, not booleans")
    if torch.is_tensor(boundaries) and boundaries.dtype in (
        torch.bool,
        torch.complex64,
        torch.complex128,
    ):
        raise ValueError("Pressure boundaries must be real numbers")
    p, m = _tensor(boundaries, model), _tensor(rates, model)
    if p.shape != (count,) or m.shape != (count, 3) or count < 1:
        raise ValueError("One boundary and three component masses per pipe required")
    low, high = model.pressure_bounds_pa
    pressure_ok = torch.isfinite(p).all() & (p > 0).all() & (p >= low).all() & (p <= high).all()
    total = m.sum(1)
    mass_ok = (
        torch.isfinite(m).all() & (m >= 0).all() & torch.isfinite(total).all() & (total > 0).all()
    )
    q = m / model.rho
    oil = q[:, 0] > 0
    ratio = torch.where(oil, q[:, 2] / torch.where(oil, q[:, 0], torch.ones_like(q[:, 0])), 0.0)
    rs = model.columns["Rs"]
    under = oil & (ratio < rs[-1])
    needs_continuity = oil & (ratio > rs[0]) & (ratio < rs[-1])
    pure_water = (m[:, 0] == 0) & (m[:, 2] == 0)
    coverage = ~under
    water_ok = ~pure_water
    continuous = ~needs_continuity
    if len(model.oil_rs) >= 2:
        coverage |= (ratio >= model.oil_rs[0]) & (ratio <= model.oil_rs[-1])
        water_ok |= (model.oil_rs[0] <= 0) & (model.oil_rs[-1] >= 0)
        # Avoid invalid arithmetic in inactive NaN/Inf/out-of-range rows.
        # Such rows still fail their original domain predicates above.
        lookup = torch.where(needs_continuity, ratio, rs[0]).detach()
        i = (torch.searchsorted(rs, lookup.contiguous(), right=True) - 1).clamp(0, len(rs) - 2)
        span = rs[i + 1] - rs[i]
        bubble = model.pressures[i] + (lookup - rs[i]) * (
            model.pressures[i + 1] - model.pressures[i]
        ) / torch.where(span > 0, span, torch.ones_like(span))
        j = (torch.searchsorted(model.oil_rs, lookup.contiguous(), right=True) - 1).clamp(
            0, len(model.oil_rs) - 2
        )
        weight = (lookup - model.oil_rs[j]) / (model.oil_rs[j + 1] - model.oil_rs[j])
        matches = torch.ones_like(oil)
        for key, column in [("Bo", 1), ("mu_o", 2)]:
            values = torch.stack(
                [
                    model.interpolate(bubble, table[0], table[column])
                    for table in model.oil_tables.values()
                ],
                1,
            )
            lo = values.gather(1, j[:, None]).squeeze(1)
            hi = values.gather(1, (j + 1)[:, None]).squeeze(1)
            under_value = lo * (1 - weight) + hi * weight
            saturated = model.interpolate(bubble, model.pressures, model.columns[key])
            matches &= torch.isclose(saturated, under_value, rtol=1e-7, atol=1e-10)
        continuous |= matches
    packet = (
        torch.stack([pressure_ok, mass_ok, coverage.all(), continuous.all(), water_ok.all()])
        .detach()
        .cpu()
        .tolist()
    )
    messages = [
        "Surface pressure outside finite positive PVT coverage",
        "Expected finite nonnegative component masses with positive finite total",
        "Undersaturated surface oil requires covering extended PVTO Rs",
        "Saturated and undersaturated PVTO must be continuous at the stream bubble pressure",
        "Pure-water composition derivatives require covering zero-GOR oil PVTO",
    ]
    for ok, message in zip(packet, messages):
        if not ok:
            raise PropertyModelError(message)
    return p, m
