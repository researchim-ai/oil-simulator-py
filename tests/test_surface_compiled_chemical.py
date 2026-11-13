"""CUDA fusion preserves eager acceptance, sensitivities and model parameters."""

import json
from pathlib import Path

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid

from surface import chemical_kernel
from surface.equilibrium_properties import EquilibriumFluid
from surface.equilibrium_stability_batch import stability_candidates

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@CUDA
def test_compiled_chunks_keep_trace_species_pure_points_and_full_root_choice():
    f = fluid("cuda")
    p = torch.linspace(9e5, 2e6, 71, device=f.device, dtype=torch.float64)
    t = torch.linspace(310.0, 350.0, 71, device=f.device, dtype=torch.float64)
    z = torch.linspace(0.1, 0.9, 71, device=f.device, dtype=torch.float64)
    x = torch.stack((z, 1 - z), -1)
    x[0] = x.new_tensor([1.0, 1e-30])
    x[1] = x.new_tensor([0.0, 1.0])
    with torch.no_grad():
        actual = f.tensor_chemical_batch(p, t, x)
        full = torch.vmap(f.tensor_homogeneous)(p, t, x)
    assert f._chemical_compiled_batches == 2
    assert f._chemical_compiled_points == 71
    assert f._chemical_compile_failure is None
    for key, value in actual.items():
        torch.testing.assert_close(value, full[key], rtol=0, atol=2e-12)


@CUDA
def test_compiled_kernel_observes_changed_interaction_parameters():
    f = fluid("cuda")
    p, t, x = map(f.tensor, (1e6, 320.0, [0.2, 0.8]))
    with torch.no_grad():
        before = f.tensor_chemical(p, t, x)["log_fugacity_coefficients"].clone()
        f.kij[0, 1] = f.kij[1, 0] = 0.1
        actual = f.tensor_chemical(p, t, x)
        full = f.tensor_homogeneous(p, t, x)
    assert not torch.equal(before, actual["log_fugacity_coefficients"])
    assert f._chemical_compiled_batches == 2 and f._chemical_compile_failure is None
    for key, value in actual.items():
        torch.testing.assert_close(value, full[key], rtol=0, atol=2e-12)


@CUDA
def test_sensitivities_never_enter_compiled_trial_path(monkeypatch):
    f = fluid("cuda")
    p, t, x = map(f.tensor, (1e6, 320.0, [0.2, 0.8]))

    def forbidden(*args):
        raise AssertionError("AD reached compiled chemical trial")

    monkeypatch.setattr(chemical_kernel, "_compiled", forbidden)
    monkeypatch.setattr(chemical_kernel, "_compiled_jacobian", forbidden)
    for method in (torch.func.jacfwd, torch.func.jacrev):
        expected = method(lambda v: f.tensor_homogeneous(p, t, v)["log_fugacity_coefficients"])(x)
        actual = method(lambda v: f.tensor_chemical(p, t, v)["log_fugacity_coefficients"])(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.no_grad():
        actual = torch.func.jacfwd(
            lambda v: f.tensor_chemical_batch(p.expand(2), t.expand(2), v)[
                "log_fugacity_coefficients"
            ]
        )(x.expand(2, -1).clone())
        assert bool(torch.isfinite(actual).all())
        f.kij.requires_grad_(True)
        f.tensor_chemical(p, t, x)
        f.tensor_chemical_jacobian(
            p.reshape(1),
            t.reshape(1),
            x.log().reshape(1, -1),
            torch.arange(2, device=f.device),
            (x[:1].log() - x[1:].log()).reshape(1, 1),
        )
    assert f._chemical_compiled_batches == 0
    assert f._chemical_jacobian_compiled_batches == 0


@CUDA
@pytest.mark.parametrize("support", [[0, 1], [1, 2], [0, 1, 2]])
def test_compiled_reverse_jacobian_matches_reference_forward_ad(support):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(
        (root / "configs/surface_pr_integrated_facility_liquid_mixture.json").read_text()
    )
    f = EquilibriumFluid.from_dict(cfg["model"]["plant"]["transport"]["fluid"], device="cuda")
    p = torch.linspace(9e5, 2e6, 71, device=f.device, dtype=torch.float64)
    t = torch.linspace(310.0, 350.0, 71, device=f.device, dtype=torch.float64)
    point = torch.linspace(-1, 1, 71 * (len(support) - 1), device=f.device, dtype=torch.float64)
    point = point.reshape(71, -1)
    point[0, 0] = -70
    point[1, -1] = 70
    mu = p.new_zeros((71, len(support)))
    active = torch.tensor(support, dtype=torch.long, device=f.device)
    with torch.no_grad():
        expected = chemical_kernel._jacobian(f, p, t, mu, active, point)
        actual = f.tensor_chemical_jacobian(p, t, mu, active, point)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, rtol=5e-12, atol=2e-12)
    assert f._chemical_jacobian_compiled_batches == 2
    assert f._chemical_jacobian_compiled_points == 71
    assert f._chemical_jacobian_compile_failure is None


@CUDA
def test_derivative_compile_failure_keeps_forward_ad_on_same_device(monkeypatch):
    f = fluid("cuda")
    p, t, x = map(f.tensor, ([1e6], [320.0], [[0.2, 0.8]]))
    mu = x.log()
    active = torch.arange(2, device=f.device)
    point = (x[:, :1].log() - x[:, 1:].log()).contiguous()

    def unavailable(*args, **kwargs):
        raise torch._dynamo.exc.Unsupported("injected derivative compiler unavailable")

    monkeypatch.setattr(chemical_kernel, "_compiled_jacobian", unavailable)
    with torch.no_grad(), pytest.warns(RuntimeWarning, match="eager CUDA retained"):
        actual = f.tensor_chemical_jacobian(p, t, mu, active, point)
    expected = chemical_kernel._jacobian(f, p, t, mu, active, point)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == "cuda"
    assert f._chemical_compile_failure is None
    assert "injected derivative compiler unavailable" in f._chemical_jacobian_compile_failure


@CUDA
def test_compiler_unavailable_retains_eager_cuda_and_records_reason(monkeypatch):
    f = fluid("cuda")
    p, t, x = map(f.tensor, (1e6, 320.0, [0.2, 0.8]))

    def unavailable(*args):
        raise torch._dynamo.exc.Unsupported("injected compiler unavailable")

    monkeypatch.setattr(chemical_kernel, "_compiled", unavailable)
    with torch.no_grad(), pytest.warns(RuntimeWarning, match="eager CUDA retained"):
        actual = f.tensor_chemical(p, t, x)
    assert "injected compiler unavailable" in f._chemical_compile_failure
    assert all(v.device.type == "cuda" for v in actual.values())
    with torch.no_grad():
        eager = f.tensor_chemical(p, t, x)
    for key, value in actual.items():
        torch.testing.assert_close(value, eager[key], rtol=0, atol=0)


@CUDA
def test_unexpected_kernel_failure_is_not_silently_accepted(monkeypatch):
    f = fluid("cuda")
    p, t, x = map(f.tensor, (1e6, 320.0, [0.2, 0.8]))

    def broken(*args):
        raise RuntimeError("unexpected device failure")

    monkeypatch.setattr(chemical_kernel, "_compiled", broken)
    with torch.no_grad(), pytest.raises(RuntimeError, match="unexpected device failure"):
        f.tensor_chemical(p, t, x)
    assert f._chemical_compile_failure is None


def test_candidate_acceptance_is_rechecked_with_full_eager_chemical_potentials(monkeypatch):
    f = fluid()
    p, t, x = map(f.tensor, (1e6, 320.0, [0.2, 0.8]))
    mu = x.log() + f.tensor_homogeneous(p, t, x)["log_fugacity_coefficients"]
    original = f.tensor_chemical_batch

    def biased(*args):
        result = original(*args)
        result["log_fugacity_coefficients"] = result["log_fugacity_coefficients"] + x.new_tensor(
            [1e-7, 0.0]
        )
        return result

    monkeypatch.setattr(f, "tensor_chemical_batch", biased)
    _value, _x, q, accepted = stability_candidates(
        f,
        p.reshape(1),
        t.reshape(1),
        mu.reshape(1, -1),
        torch.arange(2),
        (x[:1].log() - x[1:].log()).reshape(1, 1),
    )
    assert bool(q["valid"][0]) and not bool(accepted[0])
