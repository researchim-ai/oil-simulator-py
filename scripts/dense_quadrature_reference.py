"""Benchmark/test reference: Gauss nodes on every interval, including zero spans."""

import torch


class DenseQuadraturePlan:
    def __init__(self, points):
        self.half = (points[:, 1:] - points[:, :-1]) / 2
        self.mid = (points[:, 1:] + points[:, :-1]) / 2
        self.point_evaluations = 0

    def evaluate(self, kernel, masses, resistance, nodes, weights):
        p = self.mid[:, :, None] + self.half[:, :, None] * nodes
        values = kernel._integrand(p, self.half[:, :, None], masses, resistance)
        self.point_evaluations += p.numel()
        return torch.stack([(self.half * (weights * v).sum(-1)).sum(-1) for v in values], 1)
