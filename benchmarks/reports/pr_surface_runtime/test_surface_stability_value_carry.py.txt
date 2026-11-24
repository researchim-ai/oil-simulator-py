"""Selected line-search values can be reused without changing TPD acceptance."""

import pytest
import torch
from test_surface_stability_curvature import RegularSolution

from surface.equilibrium_minimization_batch import _first_acceptable, _minimize
from surface.equilibrium_stability_batch import can_reuse_values, stability_candidates


class Counted(RegularSolution):
    def __init__(self, interaction):
        super().__init__(interaction)
        self.calls = 0

    def tensor_homogeneous(self, *args):
        self.calls += 1
        return super().tensor_homogeneous(*args)


def inputs(f):
    point = torch.tensor([[-10.0], [-1.0], [0.0], [1.0], [10.0]], dtype=torch.float64)
    p = point.new_full((5,), 1e6)
    t = point.new_full((5,), 350.0)
    z = point.new_tensor([0.5, 0.5])
    mu = z.log() + f.tensor_homogeneous(p[0], t[0], z)["log_fugacity_coefficients"]
    return p, t, mu.expand(5, -1), torch.arange(2), point


@pytest.mark.parametrize("interaction", [0.0, 1.5, 3.0])
def test_newton_keeps_every_minimum_and_saddle_with_fewer_eos_calls(interaction):
    results, counts = [], []
    for enabled in (False, True):
        f = Counted(interaction)
        f._reuse_stability_values = enabled
        result = stability_candidates(f, *inputs(f))
        results.append(result)
        counts.append(f.calls)
    for a, b in zip(results[0], results[1]):
        if isinstance(a, dict):
            for name in a:
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=1e-12)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
    assert counts[1] < counts[0]
    if interaction == 3.0:
        assert not bool(results[1][-1][2])  # The stationary saddle remains rejected.


@pytest.mark.parametrize("width", [1, 4, 12])
def test_bfgs_carries_matching_rows_through_staged_search_and_compaction(width):
    results, counts = [], []
    for enabled in (False, True):
        f = Counted(1.5)
        f._reuse_stability_values = enabled
        f._minimization_line_width = width
        results.append(_minimize(f, *inputs(f)))
        counts.append(f.calls)
    torch.testing.assert_close(results[1], results[0], rtol=0, atol=1e-12)
    assert counts[1] < counts[0]


def test_rejected_or_finished_rows_retain_their_own_scores():
    point = torch.zeros((3, 1), dtype=torch.float64)
    p = point.new_tensor([0.0, 1.0, 2.0])
    t, mu = p + 350, point.clone()
    value, grad, residual = p * 0, -torch.ones_like(point), torch.ones_like(point)
    valid = torch.ones(3, dtype=torch.bool)
    done = torch.tensor([False, False, True])

    def evaluate(v, pp, tt, chem):
        x = v[..., 0]
        finite = torch.where(pp < 1, -x, torch.nan)
        return finite, torch.zeros_like(v), torch.zeros_like(v), None, {"valid": x > 0}

    chosen, carried = _first_acceptable(
        evaluate,
        point,
        -grad,
        p,
        t,
        mu,
        value,
        grad,
        residual,
        done,
        point.new_tensor([1.0, 0.5, 0.25]),
        1,
        current_values=(value, grad, residual, valid),
    )
    torch.testing.assert_close(chosen[:, 0], point.new_tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(carried[0], point.new_tensor([-1.0, 0.0, 0.0]))
    torch.testing.assert_close(carried[1][:, 0], point.new_tensor([0.0, -1.0, -1.0]))
    torch.testing.assert_close(carried[2][:, 0], point.new_tensor([0.0, 1.0, 1.0]))
    assert bool(carried[3].all())


def test_differentiable_inputs_never_reuse_detached_values():
    f = RegularSolution(0.0)
    p, t, mu, _, v = inputs(f)
    assert can_reuse_values(f, v, p, t, mu)
    assert not can_reuse_values(f, v.requires_grad_(), p, t, mu)
