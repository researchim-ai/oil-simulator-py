"""Compact Gauss points while retaining collapsed-span first/second AD terms.

One interval-count status read per pressure trial is reused by all Gauss orders.
Active intervals and collapsed midpoint contributions share one property pass.
"""

import torch


class CompactQuadraturePlan:
    def __init__(self, points):
        self.half = (points[:, 1:] - points[:, :-1]) / 2
        self.mid = (points[:, 1:] + points[:, :-1]) / 2
        self.lower = points[:, 0]
        self.active = self.half > 0
        self.intervals = self.half.shape[1]
        self.maximum_active = int(self.active.sum(1).amax().detach().cpu().item())
        if 0 < self.maximum_active < self.intervals:
            index = torch.argsort((~self.active).to(torch.int8), dim=1, stable=True)[
                :, : self.maximum_active
            ]
            self.active_half = self.half.gather(1, index)
            self.active_mid = self.mid.gather(1, index)
            if self.maximum_active < self.intervals:
                # Padded slots are evaluated again by the collapsed-span
                # branch. Suppress their AD contribution here, so a zero
                # interval with distinct endpoint derivatives is counted once.
                present = self.active.gather(1, index)
                self.active_half = torch.where(present, self.active_half, 0.0)
                self.active_mid = torch.where(present, self.active_mid, self.lower[:, None])
        self.point_evaluations = 0

    def evaluate(self, kernel, masses, resistance, nodes, weights):
        n = len(masses)
        if self.maximum_active == self.intervals:
            p = self.mid[:, :, None] + self.half[:, :, None] * nodes
            values = kernel._integrand(p, self.half[:, :, None], masses, resistance)
            self.point_evaluations += p.numel()
            return torch.stack([(self.half * (weights * v).sum(-1)).sum(-1) for v in values], 1)
        p_parts, width_parts = [], []
        if self.maximum_active:
            p = self.active_mid[:, :, None] + self.active_half[:, :, None] * nodes
            p_parts.append(p.reshape(n, -1))
            width_parts.append(self.active_half[:, :, None].expand_as(p).reshape(n, -1))
        collapsed = self.maximum_active < self.intervals
        if collapsed:
            # A zero half-width can carry derivatives if distinct symbolic
            # endpoints happen to coincide. Never replace it with a constant.
            zero_half = torch.where(self.active, torch.zeros_like(self.half), self.half)
            p_parts.append(torch.where(self.active, self.lower[:, None], self.mid))
            width_parts.append(zero_half)
        p = torch.cat(p_parts, 1)[:, None, :]
        widths = torch.cat(width_parts, 1)[:, None, :]
        values = kernel._integrand(p, widths, masses, resistance)
        self.point_evaluations += p.numel()
        active_points = self.maximum_active * len(nodes)
        result = []
        for value in values:
            value = value.reshape(n, -1)
            total = masses.new_zeros(n)
            if self.maximum_active:
                gauss = value[:, :active_points].reshape(n, self.maximum_active, len(nodes))
                total = (self.active_half * (weights * gauss).sum(-1)).sum(-1)
            if collapsed:
                total = total + (zero_half * weights.sum() * value[:, active_points:]).sum(-1)
            result.append(total)
        return torch.stack(result, 1)


def make_quadrature_plan(points):
    return CompactQuadraturePlan(points)
