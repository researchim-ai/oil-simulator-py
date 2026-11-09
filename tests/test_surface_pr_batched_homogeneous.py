"""Batch EOS must preserve scalar PR values, derivatives and absent inventory."""

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid

from surface.liquid import PropertyModelError
from surface.pr_batched_homogeneous import homogeneous_batch


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_mixed_roots_and_temperatures_match_scalar_kernel_and_gradients(device):
    f = fluid(device=device)
    p = f.tensor([1e6, 2e6, 1.1e6]).requires_grad_()
    t = f.tensor([200.0, 320.0, 290.0]).requires_grad_()
    x = f.tensor([[0.0, 1.0], [0.5, 0.5], [0.2, 0.8]])
    roots = ["liquid", "vapor", "stable"]
    batched = homogeneous_batch(f, p, t, x, roots)
    scalar = [f.tensor_homogeneous(p[i], t[i], x[i], root) for i, root in enumerate(roots)]
    for key in batched:
        torch.testing.assert_close(
            batched[key], torch.stack([s[key] for s in scalar]), rtol=1e-12, atol=1e-12
        )
    for key in ("molar_volume_m3_mol", "molar_internal_energy_j_mol"):
        a = torch.autograd.grad(batched[key].sum(), (p, t), retain_graph=True)
        b = torch.autograd.grad(
            torch.stack([s[key] for s in scalar]).sum(), (p, t), retain_graph=True
        )
        for aa, bb in zip(a, b):
            torch.testing.assert_close(aa, bb, rtol=1e-10, atol=1e-12)


def test_batch_rejects_one_invalid_phase_without_returning_partial_result():
    f = fluid()
    with pytest.raises(PropertyModelError, match="outside domain"):
        homogeneous_batch(
            f, [1e6, 1e6], [320.0, float("nan")], [[0.5, 0.5], [0.5, 0.5]], ["vapor"] * 2
        )


def test_absent_slots_do_not_enter_eos_batch(monkeypatch):
    from test_surface_pr_active_multifluid_pipe import phase, setup

    import surface.pr_active_multifluid_pipe as module

    pipe, v = setup(count=3)
    cells = [{"gas": phase(pipe.fluid, 1e6, 320.0, "vapor", v, x=(0.5, 0.5))} for _ in range(3)]
    original = module.homogeneous_batch
    calls = []

    def capture(fluid, p, t, x, roots, **kwargs):
        calls.append(tuple(roots))
        return original(fluid, p, t, x, roots, **kwargs)

    monkeypatch.setattr(module, "homogeneous_batch", capture)
    state = pipe.initialize(cells)
    pipe.inspect(state)
    assert calls and all(roots == ("vapor",) * 3 for roots in calls)
