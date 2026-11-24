"""Transient residuals must release their captured states without cyclic GC."""

import gc
import weakref

import torch
from check_pr_collocation import thermal_fixture
from test_surface_pr_surface_control import controlled

from surface import pr_collocation
from surface.cuda_execution import declare_runtime_residual
from surface.pr_collocation import PRCollocation
from surface.pr_gravity_transport import _ColumnCoupling


def test_runtime_protocol_keeps_live_defaults_and_ad_without_self_alias():
    coefficient = torch.tensor(2.0, dtype=torch.float64)

    def residual(x, *, bias=coefficient):
        return x.square() + bias

    declare_runtime_residual(residual)
    assert residual.runtime_residual is not residual
    assert residual.runtime_residual.__code__ is residual.__code__
    x = torch.tensor([3.0], dtype=torch.float64)
    coefficient.add_(4.0)
    torch.testing.assert_close(residual(x), residual.runtime_residual(x), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.func.jacfwd(residual)(x),
        torch.func.jacfwd(residual.runtime_residual)(x),
        rtol=0,
        atol=0,
    )


def test_runtime_protocol_does_not_retain_captured_tensor_until_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        tensor = torch.ones(128, dtype=torch.float64)

        def residual(x, *, coefficient=tensor):
            return x + coefficient.sum()

        declare_runtime_residual(residual)
        refs = [weakref.ref(tensor), weakref.ref(residual), weakref.ref(residual.runtime_residual)]
        del tensor, residual
        assert all(reference() is None for reference in refs)
    finally:
        if enabled:
            gc.enable()


def test_radau_stage_and_embedded_residuals_release_without_collection(monkeypatch):
    physics, state = thermal_fixture()
    model = PRCollocation(physics, scheme="radau3", error_control="embedded")
    refs = []
    original = pr_collocation.declare_runtime_residual

    def watched(function):
        result = original(function)
        refs.extend([weakref.ref(function), weakref.ref(function.runtime_residual)])
        return result

    monkeypatch.setattr(pr_collocation, "declare_runtime_residual", watched)
    enabled = gc.isenabled()
    gc.disable()
    try:
        model._step(state, 0.01, physics.default_openings, [0.0, 0.0], [320.0, 360.0], {})
        assert len(refs) == 4
        assert all(reference() is None for reference in refs)
    finally:
        if enabled:
            gc.enable()


def test_column_couplings_release_without_collection(monkeypatch):
    model, state = controlled()
    refs = []
    original = _ColumnCoupling.__init__

    def watched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        refs.append(weakref.ref(self))

    monkeypatch.setattr(_ColumnCoupling, "__init__", watched)
    enabled = gc.isenabled()
    gc.disable()
    try:
        report = model.advance(state, 1e-7, targets={"v": 0.5})
        assert report["attempts"] == 1
        assert refs and all(reference() is None for reference in refs)
    finally:
        if enabled:
            gc.enable()
