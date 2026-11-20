"""Chemical-only EOS keeps root choice, admissibility and fugacity derivatives."""

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "temperature,composition",
    [(320.0, [0.2, 0.8]), (196.2044254, [0.5, 0.5]), (320.0, [1.0, 1e-30])],
)
def test_chemical_eos_is_identical_to_full_properties(device, temperature, composition):
    f = fluid(device)
    p, t, x = map(f.tensor, (1e6, temperature, composition))
    x = x / x.sum()
    full = f.tensor_homogeneous(p, t, x)
    chemical = f.tensor_chemical(p, t, x)
    for key, value in chemical.items():
        torch.testing.assert_close(value, full[key], rtol=0, atol=0)
    pt = torch.stack((p, t))
    for method in (torch.func.jacfwd, torch.func.jacrev):
        expected = method(
            lambda v: f.tensor_homogeneous(v[0], v[1], x)["log_fugacity_coefficients"]
        )(pt)
        actual = method(lambda v: f.tensor_chemical(v[0], v[1], x)["log_fugacity_coefficients"])(pt)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
        expected = method(lambda v: f.tensor_homogeneous(p, t, v)["log_fugacity_coefficients"])(x)
        actual = method(lambda v: f.tensor_chemical(p, t, v)["log_fugacity_coefficients"])(x)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("device", DEVICES)
def test_chemical_eos_retains_full_root_validity_under_vmap(device):
    f = fluid(device)
    p = f.tensor([1e6, 1.01e6, 1e6])
    t = f.tensor([320.0, 196.2044254, 196.2044254])
    x = f.tensor([[0.2, 0.8], [0.5, 0.5], [1.0, 0.0]])
    full = torch.vmap(f.tensor_homogeneous)(p, t, x)
    chemical = torch.vmap(f.tensor_chemical)(p, t, x)
    for key, value in chemical.items():
        torch.testing.assert_close(value, full[key], rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["stable", "liquid", "vapor"])
def test_native_property_batch_and_derivatives_match_scalar_calls(phase):
    f = fluid()
    p = f.tensor([1e6, 2e6, 1e6, 5e6, 1e6])
    t = f.tensor([320.0, 250.0, 196.2044254, 350.0, 190.0])
    x = f.tensor([[0.2, 0.8], [0.1, 0.9], [0.5, 0.5], [1.0, 0.0], [0.5, 0.5]])
    batch = f.tensor_homogeneous(p, t, x, phase)
    scalar = [f.tensor_homogeneous(p[i], t[i], x[i], phase) for i in range(len(p))]
    for key in batch:
        expected = torch.stack([q[key] for q in scalar])
        torch.testing.assert_close(batch[key], expected, rtol=1e-11, atol=1e-11)
    actual = torch.func.jvp(
        lambda pp, tt: f.tensor_homogeneous(pp, tt, x, phase)["molar_enthalpy_j_mol"],
        (p, t),
        (torch.ones_like(p), torch.ones_like(t)),
    )[1]
    expected = torch.stack(
        [
            torch.func.jvp(
                lambda pp, tt, i=i: f.tensor_homogeneous(pp, tt, x[i], phase)[
                    "molar_enthalpy_j_mol"
                ],
                (p[i], t[i]),
                (p.new_tensor(1.0), t.new_tensor(1.0)),
            )[1]
            for i in range(len(p))
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_native_chemical_jacobian_is_row_local_and_matches_reference_ad():
    from surface.chemical_kernel import _jacobian, _native_jacobian

    f = fluid()
    p = f.tensor([1e6, 2e6, 5e6])
    t = f.tensor([190.0, 250.0, 350.0])
    point = f.tensor([[-2.0], [0.0], [2.0]])
    active = torch.arange(2)
    chemical = f.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    expected = _jacobian(f, p, t, chemical, active, point)
    actual = _native_jacobian(f, p, t, chemical, active, point)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    for i in range(3):
        single = _native_jacobian(
            f, p[i : i + 1], t[i : i + 1], chemical[i : i + 1], active, point[i : i + 1]
        )
        torch.testing.assert_close(actual[i], single[0], rtol=1e-11, atol=1e-11)


def test_native_single_component_composition_tangent_has_zero_width():
    from surface.chemical_kernel import _native_jacobian

    f = fluid()
    p = f.tensor([1e6, 2e6])
    t = f.tensor([320.0, 350.0])
    result = _native_jacobian(
        f, p, t, f.tensor([[0.0], [0.0]]), torch.tensor([0]), f.tensor([[], []])
    )
    assert result.shape == (2, 0, 0)
