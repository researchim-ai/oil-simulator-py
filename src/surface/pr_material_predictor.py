"""Positive implicit frozen-carrier guesses for newly reachable components.

This bounded M-matrix solve is a nonlinear INITIAL GUESS only. It never
updates physical stocks, accepted fluxes, energy targets or balance thresholds.
The final common PR Newton retains every original material/energy equation.
"""

import torch


def carrier_inventory_guess(stocks, edges, incoming, outgoing, dt, *, maximum_cells=256):
    count = len(stocks)
    if count > maximum_cells:
        return None
    total = stocks.sum(1)
    if not bool((torch.isfinite(stocks).all() & (stocks >= 0).all() & (total > 0).all()).cpu()):
        return None
    matrix = torch.eye(count, dtype=stocks.dtype, device=stocks.device)
    if edges:
        # Dense tensor assembly is bounded. Each column is a signed incidence
        # vector, so internal frozen transfers conserve each component. No
        # CUDA scatter atomics or host copies of numerical edge rates.
        incidence = stocks.new_zeros((count, 2 * len(edges)))
        donor = stocks.new_zeros((2 * len(edges), count))
        rates = []
        for k, (a, b, rate) in enumerate(edges):
            incidence[a, 2 * k], incidence[b, 2 * k] = 1.0, -1.0
            incidence[b, 2 * k + 1], incidence[a, 2 * k + 1] = 1.0, -1.0
            donor[2 * k, a], donor[2 * k + 1, b] = 1 / total[a], 1 / total[b]
            rates.extend((rate.clamp_min(0), (-rate).clamp_min(0)))
        matrix = matrix + dt * ((incidence * torch.stack(rates)[None, :]) @ donor)
    matrix = matrix + torch.diag(dt * outgoing / total)
    rhs = stocks + dt * incoming
    with torch.no_grad():
        values, info = torch.linalg.solve_ex(matrix, rhs, check_errors=False)
        valid = (
            (info == 0)
            & torch.isfinite(values).all()
            & (values >= 0).all()
            & torch.isfinite(rhs).all()
            & (incoming >= 0).all()
            & (outgoing >= 0).all()
        )
        # A failed/ill-conditioned predictor simply keeps the original guess.
        residual = (matrix @ values - rhs).abs().max()
        valid = valid & (residual <= 1e-12 * rhs.abs().sum().clamp_min(1.0))
    return values if bool(valid.cpu()) else None
