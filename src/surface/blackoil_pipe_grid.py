"""Pressure knots of the two PVTO tables used by each stream.

Topology is built once from the validated immutable PVT snapshot on the host;
all per-stream choices are device-local. Common PVT knots and the moving
bubble pressure remain mandatory. No phase or accuracy tolerance is changed.
"""

import numpy as np
import torch


class PipePressureGrid:
    def __init__(self, model):
        self.rho, self.oil_rs = model.rho, model.oil_rs
        self.global_knots = model._fixed_pressures
        self.maximum_fixed_knots = len(model._fixed_values)
        self.uniform_global = True
        self.pair_knots = None
        if len(model.oil_rs) < 2:
            return
        # PVTTable is the validated source snapshot; its CPU axes are retained
        # by BlackOilMixture.to(). Do not copy device axis values to the host.
        base = tuple(float(v) for v in np.asarray(model.pvt.P) * 1e6)
        ratios = tuple(float(v) for v in model.pvt.oil_rs)
        axes = [
            tuple(float(v) for v in np.asarray(model.pvt.oil_tables[r]["pressure_MPa"]) * 1e6)
            for r in ratios
        ]
        pairs = [tuple(sorted(set(base + a + b))) for a, b in zip(axes, axes[1:])]
        self.uniform_global = all(p == model._fixed_values for p in pairs)
        if self.uniform_global:
            return
        self.maximum_fixed_knots = max(map(len, pairs))
        packed = np.full((len(pairs), self.maximum_fixed_knots), np.inf, dtype=np.float64)
        for i, knots in enumerate(pairs):
            packed[i, : len(knots)] = knots
        self.pair_knots = torch.as_tensor(packed, dtype=torch.float64, device=model.rho.device)

    def fixed(self, masses):
        if self.uniform_global:
            return self.global_knots[None, :].expand(len(masses), -1)
        q = masses / self.rho
        ratio = torch.where(
            q[:, 0] > 0, q[:, 2] / torch.where(q[:, 0] > 0, q[:, 0], torch.ones_like(q[:, 0])), 0.0
        )
        index = (torch.searchsorted(self.oil_rs, ratio.detach(), right=True) - 1).clamp(
            0, len(self.oil_rs) - 2
        )
        return self.pair_knots[index]


def get_pressure_grid(model):
    """Cache topology only on this validated property snapshot, never globally."""
    grid = getattr(model, "_pipe_pressure_grid", None)
    if grid is None:
        grid = model._pipe_pressure_grid = PipePressureGrid(model)
    return grid
