"""Select only the two PVTO Rs tables used by each independent stream.

Ragged pressure axes are padded once per pipe batch, never per quadrature point.
Pressure interpolation keeps the scalar operation ordering and right-knot rule.
No node/point by all-Rs-table tensor is materialized.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


class PackedPVTO:
    def __init__(self, model):
        tables = tuple(model.oil_tables.values())
        if len(tables) < 2:
            raise ValueError("At least two covering PVTO Rs tables required")
        self.axis = pad_sequence(
            [t[0] for t in tables], batch_first=True, padding_value=float("inf")
        )
        self.columns = tuple(pad_sequence([t[k] for t in tables], batch_first=True) for k in (1, 2))
        self.lengths = torch.tensor(
            [len(t[0]) for t in tables], dtype=torch.long, device=model.rho.device
        )

    def interpolate(self, pressure, indices, column):
        count = pressure.shape[0]
        if indices.shape != (count,) or indices.dtype != torch.long or column not in (1, 2):
            raise ValueError("One long table index per stream and column 1/2 required")
        axis = self.axis[indices]
        values = self.columns[column - 1][indices]
        p = pressure.reshape(count, -1).contiguous()
        i = (torch.searchsorted(axis, p, right=True) - 1).clamp_min(0)
        i = torch.minimum(i, self.lengths[indices, None] - 2)
        a, b = axis.gather(1, i), axis.gather(1, i + 1)
        lo, hi = values.gather(1, i), values.gather(1, i + 1)
        return (lo + (p - a) * (hi - lo) / (b - a)).reshape_as(pressure)
