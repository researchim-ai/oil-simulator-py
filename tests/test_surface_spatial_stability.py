"""Spatial batching must confirm physics or retain the full pointwise check."""

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid

import surface.equilibrium_properties as properties_module
from surface.equilibrium_cache import cached_flashes
from surface.liquid import PropertyModelError
from surface.separator_geometry import SeparatorGeometry
from surface.stratified_pr import StratifiedPRColumn

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def states(f, n=9):
    p = torch.linspace(9e5, 2e6, n, device=f.device, dtype=torch.float64)
    t = torch.linspace(310, 350, n, device=f.device, dtype=torch.float64)
    a = torch.linspace(0.1, 0.9, n, device=f.device, dtype=torch.float64)
    x = torch.stack((a, 1 - a), -1)
    q = torch.vmap(f.tensor_homogeneous)(p, t, x)
    return p, t, x, q


@pytest.mark.parametrize("device", DEVICES)
def test_distinct_ptx_rows_chunking_and_all_original_seeds(device):
    f = fluid(device)
    p, t, x, q = states(f)
    with cached_flashes(f) as cache:
        chunked = f.homogeneous_stability_mask(
            p, t, x, q["log_fugacity_coefficients"], maximum_seed_rows=14
        )
        assert bool(chunked.all().cpu())
        assert cache.spatial_stability_batches == 5
        assert cache.spatial_stability_points == 9
        assert cache.stability_seeds == 9 * 7
    single = f.homogeneous_stability_mask(p, t, x, q["log_fugacity_coefficients"])
    assert torch.equal(single, chunked)
    for i in range(len(p)):
        reference = f.flash_pt(p[i], t[i], x[i])
        assert reference["phase_count"] == 1
        assert torch.allclose(
            reference["phases"][0]["log_fugacity_coefficients"],
            q["log_fugacity_coefficients"][i],
            atol=1e-10,
            rtol=0,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_unstable_feed_and_wrong_root_fugacity_never_confirmed(device):
    f = fluid(device)
    p = f.tensor([1e6, 1.2e6, 1.5e6])
    t = f.tensor([196.2044254, 325, 340])
    x = f.tensor([[0.5, 0.5], [0.3, 0.7], [0.7, 0.3]])
    q = torch.vmap(f.tensor_homogeneous)(p, t, x)
    mask = f.homogeneous_stability_mask(p, t, x, q["log_fugacity_coefficients"])
    assert not bool(mask[0].cpu())
    assert f.flash_pt(p[0], t[0], x[0])["phase_count"] == 2
    assert bool(mask[1:].all().cpu())
    bad = q["log_fugacity_coefficients"].clone()
    bad[1, 0] += 0.001
    wrong = f.homogeneous_stability_mask(p, t, x, bad)
    assert not bool(wrong[1].cpu())


@pytest.mark.parametrize("device", DEVICES)
def test_variable_support_and_too_small_budget_delegate_without_guessing(device):
    f = fluid(device)
    p, t, x, q = states(f, 3)
    x[0] = f.tensor([1, 0])
    q = torch.vmap(f.tensor_homogeneous)(p, t, x)
    assert not bool(
        f.homogeneous_stability_mask(p, t, x, q["log_fugacity_coefficients"]).any().cpu()
    )
    p, t, x, q = states(f, 3)
    assert not bool(
        f.homogeneous_stability_mask(p, t, x, q["log_fugacity_coefficients"], maximum_seed_rows=1)
        .any()
        .cpu()
    )


@pytest.mark.parametrize("device", DEVICES)
def test_failed_batch_factorization_keeps_pointwise_physical_checks(device, monkeypatch):
    f = fluid(device)
    p, _, x, q = states(f, 3)
    t = f.tensor(320)
    q = torch.vmap(lambda pp, xx: f.tensor_homogeneous(pp, t, xx))(p, x)
    column = StratifiedPRColumn(f, SeparatorGeometry(0.1, 2, "vertical"))

    def fail(*args, **kwargs):
        raise torch.linalg.LinAlgError("injected batch failure")

    monkeypatch.setattr(properties_module, "stability_candidates", fail)
    assert not bool(
        f.homogeneous_stability_mask(p, t, x, q["log_fugacity_coefficients"]).any().cpu()
    )
    column._check_locals([p], t, [x], [q])
    bad = {**q, "log_fugacity_coefficients": q["log_fugacity_coefficients"].clone()}
    bad["log_fugacity_coefficients"][1, 0] += 0.001
    with pytest.raises(PropertyModelError, match="topology"):
        column._check_locals([p], t, [x], [bad])


@pytest.mark.parametrize("device", DEVICES)
def test_pure_support_and_invalid_batch_domains(device):
    f = fluid(device)
    p, t, x, q = states(f, 3)
    x = f.tensor([[0, 1], [0, 1], [0, 1]])
    q = torch.vmap(f.tensor_homogeneous)(p, t, x)
    assert bool(f.homogeneous_stability_mask(p, t, x, q["log_fugacity_coefficients"]).all().cpu())
    with pytest.raises(ValueError, match="per point"):
        f.homogeneous_stability_mask(p, t, x[:2], q["log_fugacity_coefficients"])
    with pytest.raises(PropertyModelError, match="domain"):
        f.homogeneous_stability_mask(-p, t, x, q["log_fugacity_coefficients"])
    empty = f.homogeneous_stability_mask(p[:0], t[:0], x[:0], q["log_fugacity_coefficients"][:0])
    assert empty.numel() == 0 and empty.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_spatial_confirmation_keeps_scalar_pr_alpha_domain(device):
    from surface.equilibrium_properties import EquilibriumFluid
    from surface.peng_robinson import PengRobinsonFluid

    f = EquilibriumFluid(
        [PengRobinsonFluid(n, 500, 5e6, 0.2, 0.1, 100, 250, 5000) for n in ("a", "b")],
        [[0, 0], [0, 0]],
        device=device,
    )
    p, t, x = f.tensor([1e5]), f.tensor([4500]), f.tensor([[0.5, 0.5]])
    q = torch.vmap(f.tensor_homogeneous)(p, t, x)
    assert bool(torch.isfinite(q["log_fugacity_coefficients"]).all().cpu())
    with pytest.raises(PropertyModelError, match="domain"):
        f.flash_pt(p[0], t[0], x[0])
    with pytest.raises(PropertyModelError, match="domain"):
        f.homogeneous_stability_mask(p, t, x, q["log_fugacity_coefficients"])
