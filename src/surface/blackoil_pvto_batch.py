"""Select only the two PVTO Rs tables used by each independent stream.

Ragged pressure axes are padded once per pipe batch, never per quadrature point.
Pressure interpolation keeps the scalar operation ordering and right-knot rule.
No node/point by all-Rs-table tensor is materialized.
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from .blackoil_pressure_lookup import PressureLookup


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

    def prepare_lookup(self, pressure, indices):
        return PressureLookup(pressure, self.axis[indices], self.lengths[indices])

    def interpolate(self, pressure, indices, column, *, lookup=None):
        count = pressure.shape[0]
        if indices.shape != (count,) or indices.dtype != torch.long or column not in (1, 2):
            raise ValueError("One long table index per stream and column 1/2 required")
        if lookup is None:
            lookup = self.prepare_lookup(pressure, indices)
        return lookup.interpolate(self.columns[column - 1][indices])
