"""Opt-in CPU fusion retains root choice, live parameters and local derivatives."""

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid

from surface import chemical_kernel


@pytest.fixture
def f(monkeypatch):
    result = fluid()
    monkeypatch.setattr(result, "_compiled_chemical_cpu", True)
    return result


@pytest.mark.parametrize("compact", [False, True])
def test_cpu_fusion_preserves_chunked_roots_and_parameter_updates(f, compact):
    f._compact_chemical_cpu = compact
    p = torch.linspace(9e5, 2e6, 71, dtype=torch.float64)
    t = torch.linspace(310.0, 350.0, 71, dtype=torch.float64)
    z = torch.linspace(0.1, 0.9, 71, dtype=torch.float64)
    x = torch.stack((z, 1 - z), -1)
    x[0] = x.new_tensor([1.0, 1e-30])
    x[1] = x.new_tensor([0.0, 1.0])
    with torch.no_grad():
        for _ in range(2):
            expected = f.tensor_homogeneous(p, t, x, _chemical_only=True)
            actual = f.tensor_chemical_batch(p, t, x)
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-11, atol=2e-12)
            f.kij[0, 1] = f.kij[1, 0] = 0.1
        empty = f.tensor_chemical_batch(p[:0], t[:0], x[:0])
    assert empty["valid"].shape == (0,)
    assert f._chemical_compiled_points == 142
    assert f._chemical_compile_failure is None


def test_cpu_fused_local_jacobian_matches_independent_forward_ad(f):
    p = torch.linspace(9e5, 2e6, 71, dtype=torch.float64)
    t = torch.linspace(310.0, 350.0, 71, dtype=torch.float64)
    point = torch.linspace(-60.0, 60.0, 71, dtype=torch.float64).reshape(-1, 1)
    mu = p.new_zeros((71, 2))
    active = torch.arange(2)
    with torch.no_grad():
        expected = chemical_kernel._jacobian(f, p, t, mu, active, point)
        actual = f.tensor_chemical_jacobian(p, t, mu, active, point)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    assert f._chemical_jacobian_compiled_points == 71
    assert f._chemical_jacobian_compile_failure is None


def test_cpu_fusion_bypassed_for_sensitivities_and_default_policy(f, monkeypatch):
    p, t, x = map(f.tensor, (1e6, 320.0, [0.2, 0.8]))

    def forbidden(*args):
        raise AssertionError("Differentiation entered detached compiled trial")

    monkeypatch.setattr(chemical_kernel, "_compiled_cpu", forbidden)
    for method in (torch.func.jacfwd, torch.func.jacrev):
        expected = method(lambda v: f.tensor_homogeneous(p, t, v)["log_fugacity_coefficients"])(x)
        actual = method(lambda v: f.tensor_chemical(p, t, v)["log_fugacity_coefficients"])(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    f._compiled_chemical_cpu = False
    with torch.no_grad():
        f.tensor_chemical(p, t, x)
    assert not f.chemical_kernel_statistics()["enabled"]
    assert f._chemical_compiled_batches == 0


def test_cpu_compiler_failure_explicitly_falls_back_on_same_device(f, monkeypatch):
    p, t, x = map(f.tensor, (1e6, 320.0, [0.2, 0.8]))

    def unavailable(*args):
        raise torch._dynamo.exc.Unsupported("injected CPU compiler failure")

    monkeypatch.setattr(chemical_kernel, "_compiled_cpu", unavailable)
    with torch.no_grad(), pytest.warns(RuntimeWarning, match="eager CPU retained"):
        actual = f.tensor_chemical(p, t, x)
    expected = f.tensor_homogeneous(p, t, x, _chemical_only=True)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert "injected CPU compiler failure" in f._chemical_compile_failure


@pytest.mark.parametrize("support", [[0, 1], [1, 2], [0, 1, 2]])
def test_cpu_fused_ternary_support_and_trace_derivative(support, monkeypatch):
    import json
    from pathlib import Path

    from surface.equilibrium_properties import EquilibriumFluid

    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/surface_pr_integrated_facility_liquid_mixture.json"
        ).read_text()
    )
    f = EquilibriumFluid.from_dict(cfg["model"]["plant"]["transport"]["fluid"])
    monkeypatch.setattr(f, "_compiled_chemical_cpu", True)
    p, t = f.tensor([9e5, 2e6]), f.tensor([310.0, 350.0])
    active = torch.tensor(support)
    point = torch.linspace(-70.0, 70.0, 2 * (len(support) - 1), dtype=torch.float64).reshape(2, -1)
    mu = p.new_zeros((2, len(support)))
    with torch.no_grad():
        actual = f.tensor_chemical_jacobian(p, t, mu, active, point)
        expected = chemical_kernel._jacobian(f, p, t, mu, active, point)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    assert f._chemical_jacobian_compile_failure is None
    assert f._chemical_jacobian_compiled_points == 2


def test_cpu_derivative_compiler_failure_is_recorded(f, monkeypatch):
    p, t = f.tensor([1e6]), f.tensor([320.0])
    mu, point, active = f.tensor([[0.0, 0.0]]), f.tensor([[0.5]]), torch.arange(2)

    def unavailable(*args):
        raise torch._dynamo.exc.Unsupported("injected CPU derivative compiler failure")

    monkeypatch.setattr(chemical_kernel, "_compiled_cpu_jacobian", unavailable)
    with torch.no_grad(), pytest.warns(RuntimeWarning, match="eager CPU retained"):
        actual = f.tensor_chemical_jacobian(p, t, mu, active, point)
    expected = chemical_kernel._native_jacobian(f, p, t, mu, active, point)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "injected CPU derivative compiler failure" in f._chemical_jacobian_compile_failure


def test_cpu_fused_empty_and_one_component_tangents(f):
    active = torch.tensor([0])
    with torch.no_grad():
        result = f.tensor_chemical_jacobian(
            f.tensor([1e6]),
            f.tensor([320.0]),
            f.tensor([[0.0]]),
            active,
            torch.empty((1, 0), dtype=torch.float64),
        )
        empty = f.tensor_chemical_jacobian(
            f.tensor([]),
            f.tensor([]),
            torch.empty((0, 2), dtype=torch.float64),
            torch.arange(2),
            torch.empty((0, 1), dtype=torch.float64),
        )
    assert result.shape == (1, 0, 0)
    assert empty.shape == (0, 1, 1)


@pytest.mark.parametrize("count", [1, 8, 9, 64, 65, 72])
def test_small_cpu_tiles_preserve_real_properties_and_logical_row_count(f, count):
    f._compact_chemical_cpu = True
    p = torch.linspace(2e5, 3e6, count, dtype=torch.float64)
    t = torch.linspace(190.0, 370.0, count, dtype=torch.float64)
    z = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    x = torch.stack((z, 1 - z), -1)
    with torch.no_grad():
        reference = f.tensor_homogeneous(p, t, x, _chemical_only=True)
        actual = f.tensor_chemical_batch(p, t, x)
    for key in actual:
        torch.testing.assert_close(actual[key], reference[key], rtol=2e-11, atol=2e-12)
    assert f._chemical_compile_failure is None
    assert f._chemical_compiled_points == count
    full, tail = divmod(count, 64)
    expected = full * 64 + (0 if not tail else 8 if tail <= 8 else 64)
    assert f._chemical_padded_points == expected
