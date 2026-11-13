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
