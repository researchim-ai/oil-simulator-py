"""Factored quadratic mixing retains PR caloric and composition derivatives."""

import json
from pathlib import Path

import pytest
import torch

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_attraction import attraction


def fluid():
    data = json.loads(
        (Path(__file__).parent / "data/surface/pr_restart_check_valve.json").read_text()
    )
    return EquilibriumFluid.from_dict(data["model"]["transport"]["fluid"])


@pytest.mark.parametrize("phase", ["stable", "liquid", "vapor"])
@pytest.mark.parametrize("shared_temperature", [False, True])
def test_all_properties_and_supports_match_reference(phase, shared_temperature):
    f = fluid()
    p = f.tensor([1e5, 1e6, 2e6, 4e6, 1.2e6])
    t = f.tensor(350.0) if shared_temperature else f.tensor([280.0, 350.0, 400.0, 320.0, 365.0])
    x = f.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.01, 0.1, 0.89], [0.2, 0.7, 0.1]]
    )
    f._factored_attraction = False
    expected = f.tensor_homogeneous(p, t, x, phase, derivative_order=2)
    f._factored_attraction = True
    actual = f.tensor_homogeneous(p, t, x, phase, derivative_order=2)
    for key in expected:
        torch.testing.assert_close(
            actual[key], expected[key], rtol=2e-11, atol=2e-9, equal_nan=True
        )


@pytest.mark.parametrize("phase", ["stable", "liquid", "vapor"])
def test_first_and_second_state_derivatives(phase):
    f = fluid()
    y = f.tensor([13.8, 350.0, 0.2, 0.7, 0.1])

    def values(v):
        q = f.tensor_homogeneous(v[0].exp(), v[1], v[2:], phase, derivative_order=2)
        return torch.cat(
            (
                torch.stack(
                    [
                        q[k]
                        for k in (
                            "molar_enthalpy_j_mol",
                            "molar_entropy_j_mol_k",
                            "density_kg_m3",
                            "homogeneous_molar_cp_j_mol_k",
                            "frozen_sound_speed_m_s",
                        )
                    ]
                ),
                q["log_fugacity_coefficients"],
            )
        )

    f._factored_attraction = False
    first = torch.func.jacfwd(values)(y)
    second = torch.func.jacfwd(torch.func.jacfwd(values))(y)
    f._factored_attraction = True
    torch.testing.assert_close(torch.func.jacfwd(values)(y), first, rtol=2e-9, atol=2e-8)
    torch.testing.assert_close(
        torch.func.jacfwd(torch.func.jacfwd(values))(y), second, rtol=2e-8, atol=2e-7
    )


@pytest.mark.parametrize("parameter", ["a0", "kappa", "tc", "kij"])
def test_parameter_gradients_bypass_prepared_constants(parameter):
    f = fluid()
    p, t, x = f.tensor(1e6), f.tensor(350.0), f.tensor([0.2, 0.7, 0.1])
    f.tensor_homogeneous(p, t, x)  # prepare before enabling parameter sensitivities
    v = getattr(f, parameter).requires_grad_(True)
    gradients = []
    for enabled in (False, True):
        f._factored_attraction = enabled
        q = f.tensor_homogeneous(p, t, x)
        gradients.append(torch.autograd.grad(q["molar_enthalpy_j_mol"], v)[0])
    torch.testing.assert_close(gradients[1], gradients[0], rtol=2e-10, atol=2e-8)


def test_in_place_and_replaced_parameters_invalidate_constants():
    f = fluid()
    p, t, x = f.tensor(1e6), f.tensor(350.0), f.tensor([0.2, 0.7, 0.1])
    f.tensor_homogeneous(p, t, x)
    for replace in (False, True):
        if replace:
            f.a0 = f.a0 * 1.01
        else:
            f.kij[0, 1] += 0.013  # asymmetric trial must retain original equations
        f._factored_attraction = False
        old = f.tensor_homogeneous(p, t, x)
        f._factored_attraction = True
        new = f.tensor_homogeneous(p, t, x)
        for key in old:
            torch.testing.assert_close(new[key], old[key], rtol=2e-11, atol=2e-9)


def test_scalar_temperature_broadcast_and_chemical_pair_matrix():
    f = fluid()
    t = f.tensor(350.0)
    x = f.tensor([[0.2, 0.7, 0.1], [1.0, 0.0, 0.0]])
    a, at, att, ax, pairs = attraction(f, t, x, pair_matrix=True)
    torch.testing.assert_close(a, (x * ax).sum(-1))
    torch.testing.assert_close(ax, (x[..., None, :] @ pairs).squeeze(-2))
    derivative = torch.func.jacfwd(lambda tt: attraction(f, tt, x)[0])(t)
    curvature = torch.func.jacfwd(torch.func.jacfwd(lambda tt: attraction(f, tt, x)[0]))(t)
    torch.testing.assert_close(at, derivative, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(att, curvature, rtol=2e-12, atol=2e-12)


def test_first_use_during_transforms_never_saves_transient_constants():
    from torch.fx.experimental.proxy_tensor import make_fx

    f = fluid()
    t, x = f.tensor(350.0), f.tensor([0.2, 0.7, 0.1])
    traced = make_fx(
        lambda tt: attraction(f, tt, x)[0], tracing_mode="fake", _allow_non_fake_inputs=True
    )(t)
    assert not hasattr(f, "_attraction_coefficients")
    torch.testing.assert_close(traced(t), attraction(f, t, x)[0])
    del f._attraction_coefficients
    torch.func.jacfwd(lambda tt: attraction(f, tt, x)[0])(t)
    saved = getattr(f, "_attraction_coefficients", None)
    if saved is not None:
        assert all(not torch._C._functorch.is_functorch_wrapped_tensor(v) for v in saved[2])
