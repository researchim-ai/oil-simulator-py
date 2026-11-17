"""One pressure interval search shared by immutable property columns.

The lookup lives for one property evaluation, not across pressure trials.
Arithmetic keeps scalar lo + delta * (hi-lo) / span ordering in float64.
"""

import torch


class PressureLookup:
    def __init__(self, pressure, axis, lengths=None):
        if axis.ndim not in (1, 2):
            raise ValueError("One common axis or one padded axis per stream required")
        if axis.ndim == 2 and (
            pressure.ndim == 0
            or pressure.shape[0] != axis.shape[0]
            or lengths is None
            or lengths.shape != (axis.shape[0],)
        ):
            raise ValueError("Padded axes require matching stream count and row lengths")
        self.axis_shape = axis.shape
        self.shape = pressure.shape
        self.batched = axis.ndim == 2
        if self.batched:
            p = pressure.reshape(pressure.shape[0], -1).contiguous()
            self.i = (torch.searchsorted(axis, p, right=True) - 1).clamp_min(0)
            self.i = torch.minimum(self.i, lengths[:, None] - 2)
            a, b = axis.gather(1, self.i), axis.gather(1, self.i + 1)
        else:
            p = pressure
            self.i = (torch.searchsorted(axis, p.contiguous(), right=True) - 1).clamp(
                0, len(axis) - 2
            )
            a, b = axis[self.i], axis[self.i + 1]
        self.delta = p - a
        self.span = b - a

    def interpolate(self, values):
        if values.shape != self.axis_shape:
            raise ValueError("Property values must match the lookup pressure axes")
        if self.batched:
            lo, hi = values.gather(1, self.i), values.gather(1, self.i + 1)
        else:
            lo, hi = values[self.i], values[self.i + 1]
        return (lo + self.delta * (hi - lo) / self.span).reshape(self.shape)
