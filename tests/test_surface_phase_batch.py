"""Grouped EOS preserves root selection, mixed supports and nested device AD."""

import math

import pytest
import torch
from test_surface_compositional_separator import binary
from test_surface_pr_equilibrium_transport import pipe

from surface.pr_phase_batch import PhaseEOSBatch, PhaseEOSGroups

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_shared_column_temperature_retains_all_roots_and_temperature_derivative(device):
    f = binary(device=device)
    tags = ("liquid", "vapor", "liquid", "stable")
    plan = PhaseEOSGroups(f, tags)
    y = f.tensor([math.log(1e6)] * 4 + [196.2 / f.tref] + [-0.2, 0.0, 0.2, 0.1])

    def values(z, grouped):
        p, t = z[:4].exp(), z[4] * f.tref
        x = torch.softmax(torch.stack((z[5:], torch.zeros_like(z[5:])), -1), -1)
        if grouped:
            q = plan.properties(p, t, x)
        else:
            rows = [f.tensor_homogeneous(p[i], t, x[i], tag) for i, tag in enumerate(tags)]
            q = {key: torch.stack([r[key] for r in rows]) for key in rows[0]}
        return torch.cat([value.flatten() for value in q.values() if value.dtype != torch.bool])

    torch.testing.assert_close(values(y, True), values(y, False), rtol=2e-13, atol=2e-11)
    torch.testing.assert_close(
        torch.func.jacfwd(lambda z: values(z, True))(y),
        torch.func.jacfwd(lambda z: values(z, False))(y),
        rtol=3e-12,
        atol=1e-9,
    )
    samples = torch.stack((y, y + 0.0001))
    torch.testing.assert_close(
        torch.vmap(lambda z: values(z, True))(samples),
        torch.stack([values(z, False) for z in samples]),
        rtol=3e-12,
        atol=1e-9,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_roots_supports_and_nested_forward_ad(device):
    f = binary(device=device)
    a = torch.tensor([0, 1], device=f.device)
    b = torch.tensor([1], device=f.device)
    plan = PhaseEOSBatch(f, [(0, a, ("liquid", "vapor")), (7, b, ("stable",))])
    y = f.tensor(
        [
            math.log(0.4),
            math.log(0.01),
            math.log(0.01),
            math.log(0.58),
            math.log(1e6),
            196.2 / f.tref,
            0,
            math.log(3),
            math.log(1.1e6),
            320 / f.tref,
            0,
        ]
    )

    def reference(z):
        rows = []
        for offset, active, tags in plan.entries:
            m, c = len(tags), len(active)
            n = z[offset : offset + m * c].exp().reshape(m, c)
            total = n.sum(-1)
            x = z.new_zeros((m, len(f.names))).index_copy(-1, active, n / total[:, None])
            values = [
                f.tensor_homogeneous(
                    z[offset + m * c].exp(), z[offset + m * c + 1] * f.tref, x[i], tag
                )
                for i, tag in enumerate(tags)
            ]
            rows.append((total, x, {k: torch.stack([q[k] for q in values]) for k in values[0]}))
        return rows

    def packed(z, batch):
        rows = plan.evaluate(z) if batch else reference(z)
        return torch.cat(
            [
                torch.cat([v.reshape(-1) for k, v in q.items() if v.dtype != torch.bool])
                for _, _, q in rows
            ]
        )

    actual, expected = plan.evaluate(y), reference(y)
    for (_, x, q), (_, xx, qq) in zip(actual, expected):
        torch.testing.assert_close(x, xx, rtol=0, atol=0)
        for key in q:
            torch.testing.assert_close(q[key], qq[key], rtol=2e-13, atol=2e-11)
    torch.testing.assert_close(
        torch.func.jacfwd(lambda z: packed(z, True))(y),
        torch.func.jacfwd(lambda z: packed(z, False))(y),
        rtol=3e-12,
        atol=1e-9,
    )
    samples = torch.stack((y, y + 0.0001, y - 0.0001))
    torch.testing.assert_close(
        torch.vmap(lambda z: packed(z, True))(samples),
        torch.stack([packed(z, False) for z in samples]),
        rtol=3e-12,
        atol=1e-9,
    )


def test_complete_transport_residual_and_jacobian_match_scalar_reference():
    model, state = pipe(temperature=196.2, velocity=0.1)
    seeds = [model._seed(i, v) for i, (_, v) in enumerate(state.nodes)]
    args = (state, seeds, 1e-4, {"internal": True}, [0.0, 0.0])
    y, _, _, batch, *_ = model._problem(*args)
    scalar_y, _, _, scalar, *_ = model._problem(*args, _batch_phase_eos=False)
    assert torch.equal(y, scalar_y)
    torch.testing.assert_close(batch(y)[0], scalar(y)[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        torch.func.jacfwd(lambda z: batch(z)[0])(y),
        torch.func.jacfwd(lambda z: scalar(z)[0])(y),
        rtol=1e-12,
        atol=1e-11,
    )
