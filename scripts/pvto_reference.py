"""Benchmark-only prior all-table interpolation, shared by two gathers."""

import torch


class AllTablesPVTO:
    def __init__(self, model):
        self.model = model
        self.cached = None
        self.calls = 0
        self.largest_full_table_bytes = 0

    def interpolate(self, pressure, indices, column):
        if self.cached is None:
            matrix = torch.stack(
                [
                    self.model.interpolate(pressure, t[0], t[column])
                    for t in self.model.oil_tables.values()
                ],
                -1,
            )
            self.cached = (pressure, column, matrix)
            self.largest_full_table_bytes = max(
                self.largest_full_table_bytes, matrix.numel() * matrix.element_size()
            )
        else:
            p, c, matrix = self.cached
            if p is not pressure or c != column:
                raise RuntimeError(
                    "Reference requires consecutive lo/hi gathers on one pressure tensor"
                )
            self.cached = None
        shape = (len(indices),) + (1,) * (pressure.ndim - 1)
        self.calls += 1
        return matrix.gather(-1, indices.reshape(shape).expand_as(pressure)[..., None]).squeeze(-1)
