"""Shared searches preserve scalar interpolation arithmetic and pressure AD."""

import pytest
import torch
from test_surface_blackoil_properties import model

from surface.blackoil_pipe_batch import tensor_state
from surface.blackoil_pressure_lookup import PressureLookup
from surface.blackoil_pvto_batch import PackedPVTO


@pytest.mark.parametrize("shape", [(), (5,), (2, 3, 4)])
def test_common_lookup_values_and_hessian_match_scalar(shape):
    m = model()
    p = torch.full(shape, 2.4e6, dtype=torch.float64, requires_grad=True)
    lookup = PressureLookup(p, m.pressures)
    old = []
    new = []
    for key in ("Bo", "Bw", "Bg", "mu_o", "mu_g", "mu_w", "Rs"):
        old.append(m.interpolate(p, m.pressures, m.columns[key]))
        new.append(lookup.interpolate(m.columns[key]))
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    a = sum(v.square().sum() for v in old)
    b = sum(v.square().sum() for v in new)
    ga = torch.autograd.grad(a, p, create_graph=True)[0]
    gb = torch.autograd.grad(b, p, create_graph=True)[0]
    torch.testing.assert_close(ga, gb, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        torch.autograd.grad(ga.sum(), p)[0],
        torch.autograd.grad(gb.sum(), p)[0],
        rtol=1e-13,
        atol=1e-13,
    )


def test_one_base_and_two_pvto_searches_per_property_evaluation(monkeypatch):
    m = model()
    packed = PackedPVTO(m)
    original = torch.searchsorted
    calls = []

    def counted(axis, p, *a, **kw):
        calls.append(tuple(axis.shape))
        return original(axis, p, *a, **kw)

    monkeypatch.setattr(torch, "searchsorted", counted)
    p = torch.full((128, 1, 32), 2.4e6, dtype=torch.float64)
    rates = torch.tensor([[0.85, 2, 0.06]], dtype=torch.float64).expand(128, -1)
    tensor_state(m, p, rates, pvto=packed)
    # One common pressure axis, one Rs selection, two pressure-table searches.
    assert len(calls) == 4
    assert sum(len(shape) == 2 for shape in calls) == 2


def test_one_lookup_is_reused_for_both_pvto_columns(monkeypatch):
    m = model()
    packed = PackedPVTO(m)
    p = torch.full((4, 32), 2.4e6, dtype=torch.float64)
    indices = torch.tensor([0, 1, 0, 1])
    lookup = packed.prepare_lookup(p, indices)

    def forbidden(*a, **kw):
        raise AssertionError("Repeated pressure search forbidden")

    monkeypatch.setattr(torch, "searchsorted", forbidden)
    for column in (1, 2):
        result = packed.interpolate(p, indices, column, lookup=lookup)
        assert result.shape == p.shape


def test_lookup_is_scoped_to_current_pressure_and_not_cached_across_trials():
    m = model()
    rates = torch.tensor([[0.85, 2, 0.06]], dtype=torch.float64)
    a = tensor_state(m, torch.tensor([1.2e6], dtype=torch.float64), rates)
    b = tensor_state(m, torch.tensor([3.4e6], dtype=torch.float64), rates)
    assert float(a["specific_volume"][0]) != float(b["specific_volume"][0])
    torch.testing.assert_close(
        b["specific_volume"],
        m.tensor_state(torch.tensor([3.4e6], dtype=torch.float64), rates[0])["specific_volume"],
    )


def test_metadata_shape_checks_fail_explicitly():
    p = torch.ones(2, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="row lengths"):
        PressureLookup(p, torch.ones(2, 3, dtype=torch.float64))
    lookup = PressureLookup(p, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="match"):
        lookup.interpolate(torch.ones(4, dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_shared_searches_on_cuda_match_cpu():
    cpu = model()
    gpu = cpu.to("cuda")
    p = torch.full((8, 32), 2.4e6, dtype=torch.float64)
    for name in ("Bo", "Bg", "mu_o"):
        a = PressureLookup(p, cpu.pressures).interpolate(cpu.columns[name])
        b = PressureLookup(p.cuda(), gpu.pressures).interpolate(gpu.columns[name])
        torch.testing.assert_close(a, b.cpu())
