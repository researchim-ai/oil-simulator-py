"""Full fused PR properties preserve outputs, shapes and parameter sensitivities."""

import json
from pathlib import Path

import pytest
import torch

from surface.equilibrium_properties import EquilibriumFluid

DATA = json.loads((Path(__file__).parent / "data/surface/pr_restart_check_valve.json").read_text())
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def fluid(device):
    return EquilibriumFluid.from_dict(DATA["model"]["transport"]["fluid"], device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("phase", ["stable", "liquid", "vapor"])
def test_full_eos_fusion_preserves_all_properties_and_partial_batches(device, phase):
    f = fluid(device)
    p = f.tensor([1e6, 2e6, 4e6, 1e5, 1.2e6])
    t = f.tensor([350.0, 400.0, 320.0, 280.0, 365.0])
    x = f.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.01, 0.1, 0.89], [0.2, 0.7, 0.1]]
    )
    expected = f.tensor_homogeneous(p, t, x, phase)
    f._compiled_full_eos_cpu = device == "cpu"
    f._compiled_full_eos_cuda = device == "cuda"
    actual = f.tensor_homogeneous(p, t, x, phase)
    assert f._full_eos_compile_failure is None and f._full_eos_compiled_batches > 0
    assert set(actual) == set(expected)
    for k in actual:
        torch.testing.assert_close(actual[k], expected[k], rtol=3e-12, atol=3e-10, equal_nan=True)


@pytest.mark.parametrize("device", DEVICES)
def test_broadcast_api_and_ad_bypass_are_unchanged(device):
    f = fluid(device)
    p, t, x = f.tensor([1e6, 2e6]), f.tensor(350.0), f.tensor([0.0, 0.0, 1.0])
    expected = f.tensor_homogeneous(p, t, x)
    f._compiled_full_eos_cpu = device == "cpu"
    f._compiled_full_eos_cuda = device == "cuda"
    actual = f.tensor_homogeneous(p, t, x)
    for k in expected:
        assert actual[k].shape == expected[k].shape
        torch.testing.assert_close(actual[k], expected[k], rtol=3e-12, atol=3e-10)
    count = f._full_eos_compiled_batches
    point = f.tensor([1e6, 350.0])

    def properties(v):
        q = f.tensor_homogeneous(v[0], v[1], x)
        return torch.stack((q["density_kg_m3"], q["molar_entropy_j_mol_k"]))

    derivative = torch.func.jacfwd(properties)(point)
    assert f._full_eos_compiled_batches == count
    f._compiled_full_eos_cpu = f._compiled_full_eos_cuda = False
    torch.testing.assert_close(derivative, torch.func.jacfwd(properties)(point), rtol=0, atol=0)


def test_model_parameter_gradients_bypass_compilation():
    f = fluid("cpu")
    f._compiled_full_eos_cpu = True
    f.kij.requires_grad_(True)
    q = f.tensor_homogeneous(f.tensor(1e6), f.tensor(350.0), f.tensor([0.2, 0.7, 0.1]))
    gradient = torch.autograd.grad(q["molar_enthalpy_j_mol"], f.kij)[0]
    assert torch.isfinite(gradient).all()
    assert f._full_eos_compiled_batches == 0
