"""Compaction preserves endpoint AD even when intervals have zero value."""

import pytest
import torch
from dense_quadrature_reference import DenseQuadraturePlan
from test_surface_blackoil_properties import model

from surface.blackoil_pipe import BlackOilPipe, _gauss
from surface.blackoil_pipe_batch import PipeBatch
from surface.blackoil_quadrature_plan import CompactQuadraturePlan


class ToyKernel:
    def _integrand(self, p, widths, masses, resistance):
        return p.square() + 1, torch.exp(p), p.pow(3) + 2


@pytest.mark.parametrize(
    "points", [[1.0, 1.0, 2.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.5, 2.0, 2.5, 3.0]]
)
def test_zero_width_values_first_and_second_derivatives_match_dense(points):
    x = torch.tensor([points], dtype=torch.float64, requires_grad=True)
    mass = torch.ones(1, 3, dtype=torch.float64)
    nodes, weights = _gauss(32)
    outputs = [
        plan(x).evaluate(ToyKernel(), mass, None, nodes, weights)
        for plan in (DenseQuadraturePlan, CompactQuadraturePlan)
    ]
    torch.testing.assert_close(*outputs, rtol=1e-13, atol=1e-13)
    gradients = []
    hessians = []
    for value in outputs:
        g = torch.autograd.grad(value.sum(), x, create_graph=True, retain_graph=True)[0]
        gradients.append(g)
        rows = []
        for component in g.flatten():
            rows.append(torch.autograd.grad(component, x, retain_graph=True)[0])
        hessians.append(torch.stack(rows))
    torch.testing.assert_close(*gradients, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(*hessians, rtol=1e-11, atol=1e-11)
    if len(set(points)) == 1:
        assert gradients[1].abs().max() > 1  # naive removal would lose these terms


@pytest.mark.parametrize("middle", [[1.0, 1.2, 1.4, 1.7, 2.0], [1.0, 1.0, 1.4, 1.4, 2.0]])
def test_mixed_active_counts_are_padded_without_changing_integrals(middle):
    p = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 2.0], middle, [1.0, 1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    masses = torch.ones(3, 3, dtype=torch.float64)
    nodes, weights = _gauss(16)
    a = DenseQuadraturePlan(p).evaluate(ToyKernel(), masses, None, nodes, weights)
    plan = CompactQuadraturePlan(p)
    b = plan.evaluate(ToyKernel(), masses, None, nodes, weights)
    torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)
    ga = torch.autograd.grad(a.sum(), p, create_graph=True, retain_graph=True)[0]
    gb = torch.autograd.grad(b.sum(), p, create_graph=True, retain_graph=True)[0]
    torch.testing.assert_close(ga, gb, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(ga.square().sum(), p)[0],
        torch.autograd.grad(gb.square().sum(), p)[0],
        rtol=1e-11,
        atol=1e-11,
    )


def test_coincident_bubble_or_pvt_boundary_matches_original_pipe_jacobian():
    m = model()
    kernel = PipeBatch([BlackOilPipe("p", 300, 0.08, 30, 1e-5)], m)
    for lower, upper, rates in [
        (3e6, 3.2e6, [0.85, 2, 0.06]),
        (2.8e6, 3e6, [0.85, 2, 0.06]),
        (1e6, 1.2e6, [0.85, 2, 0.144]),
    ]:
        x = torch.tensor([lower, upper, *rates, 0.0], dtype=torch.float64, requires_grad=True)
        points = kernel._pressure_points(x[:1], x[1:2], x[2:5][None, :])
        nodes, weights = _gauss(32)
        a = DenseQuadraturePlan(points).evaluate(kernel, x[2:5][None, :], x[5:6], nodes, weights)
        b = CompactQuadraturePlan(points).evaluate(kernel, x[2:5][None, :], x[5:6], nodes, weights)
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-9)
        torch.testing.assert_close(
            torch.autograd.grad(a.sum(), x, retain_graph=True)[0],
            torch.autograd.grad(b.sum(), x)[0],
            rtol=1e-9,
            atol=1e-7,
        )


def test_actual_point_evaluations_shrink_and_no_tiny_interval_is_dropped():
    points = torch.tensor([[1.0, 1.0, 1.0 + 1e-12, 1.1, 1.1, 1.1]], dtype=torch.float64)
    plan = CompactQuadraturePlan(points)
    assert plan.maximum_active == 2
    nodes, weights = _gauss(32)
    plan.evaluate(ToyKernel(), torch.ones(1, 3, dtype=torch.float64), None, nodes, weights)
    assert plan.point_evaluations == 2 * 32 + 5
    assert plan.point_evaluations < 5 * 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_compaction_values_and_derivatives_match_dense():
    x = torch.tensor(
        [[1.0, 1.0, 2.0, 2.0, 3.0]], dtype=torch.float64, device="cuda", requires_grad=True
    )
    mass = torch.ones(1, 3, dtype=torch.float64, device="cuda")
    nodes, weights = _gauss(32, "cuda")
    a = DenseQuadraturePlan(x).evaluate(ToyKernel(), mass, None, nodes, weights)
    b = CompactQuadraturePlan(x).evaluate(ToyKernel(), mass, None, nodes, weights)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(
        torch.autograd.grad(a.sum(), x, retain_graph=True)[0], torch.autograd.grad(b.sum(), x)[0]
    )


def test_pressure_trial_reuses_one_plan_for_adaptive_gauss_orders(monkeypatch):
    from surface import blackoil_pipe_batch

    m = model()
    kernel = PipeBatch([BlackOilPipe("p", 300, 0.08, 30)], m)
    factory = blackoil_pipe_batch.make_quadrature_plan
    calls = []

    def counted(points):
        calls.append(tuple(points.shape))
        return factory(points)

    monkeypatch.setattr(blackoil_pipe_batch, "make_quadrature_plan", counted)
    lower, upper = (
        torch.tensor([1.2e6], dtype=torch.float64),
        torch.tensor([1.3e6], dtype=torch.float64),
    )
    rates = torch.tensor([[0.85, 2, 0.144]], dtype=torch.float64)
    _, order, _ = kernel.checked(lower, upper, rates, torch.zeros(1, dtype=torch.float64))
    assert order >= 32 and len(calls) == 1


def test_one_planning_read_for_whole_batch(monkeypatch):
    original = torch.Tensor.item
    reads = []

    def counted(value, *args):
        reads.append(tuple(value.shape))
        return original(value, *args)

    monkeypatch.setattr(torch.Tensor, "item", counted)
    points = torch.tensor([[1.0, 1.0, 2.0, 2.0, 3.0]], dtype=torch.float64).expand(512, -1)
    plan = CompactQuadraturePlan(points)
    assert plan.maximum_active == 2
    assert reads == [()]
