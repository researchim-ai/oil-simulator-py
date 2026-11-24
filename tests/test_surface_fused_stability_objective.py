"""Fused TPD values retain EOS roots, chemical gradients and physical audits."""

import json
from pathlib import Path

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid

from surface import chemical_kernel
from surface.equilibrium_minimization_batch import minimize_stability_candidates
from surface.equilibrium_stability_batch import stability_objective
from surface.pr_surface_control import PRControlledSurface


def mixture(ternary):
    if not ternary:
        result = fluid()
    else:
        config = json.loads(
            (Path(__file__).parents[1] / "configs/surface_pr_integrated_facility.json").read_text()
        )
        result = PRControlledSurface.from_dict(config["model"]).plant.transport.fluid
    result._compiled_chemical_cpu = True
    return result


def compare(expected, actual):
    for a, b in zip(expected[:4], actual[:4]):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=3e-12)
    assert actual[4].keys() == expected[4].keys()
    for name in actual[4]:
        torch.testing.assert_close(actual[4][name], expected[4][name], rtol=2e-11, atol=3e-12)


@pytest.mark.parametrize(
    "shape,active_indices", [((), [0, 1]), ((3, 6), [0, 1, 2]), ((71,), [0, 2])]
)
def test_broadcast_tails_absent_components_and_live_parameters(shape, active_indices):
    f = mixture(2 in active_indices)
    active = torch.tensor(active_indices, dtype=torch.long)
    n = len(active)
    count = max(1, torch.tensor(shape).prod().item())
    point = torch.linspace(-150, 150, count * (n - 1), dtype=torch.float64).reshape(*shape, n - 1)
    p, t, mu = f.tensor(1e6), f.tensor(320.0), f.tensor([0.3] * n)
    with torch.no_grad():
        for _ in range(2):
            f._fused_stability_objective_cpu = False
            expected = stability_objective(f, active)(point, p, t, mu)
            f._fused_stability_objective_cpu = True
            actual = stability_objective(f, active)(point, p, t, mu)
            compare(expected, actual)
            f.kij[0, 1] = f.kij[1, 0] = 0.12
    assert f._stability_objective_compiled_points == 2 * count
    assert f._stability_objective_compile_failure is None
    absent = sorted(set(range(len(f.components))) - set(active_indices))
    if absent:
        assert bool((actual[3][..., absent] == 0).all())


def test_objective_and_local_derivatives_bypass_compilation_for_ad(monkeypatch):
    f = mixture(False)
    active = torch.arange(2)
    p, t, mu, v = f.tensor(1e6), f.tensor(320.0), f.tensor([0.0, 0.0]), f.tensor([0.4])

    def forbidden(*args, **kwargs):
        raise AssertionError("An AD request entered detached fusion")

    monkeypatch.setattr(chemical_kernel, "_compiled_stability_values", {(2, 2, 8): forbidden})
    for transform in (torch.func.jacfwd, torch.func.jacrev):
        f._fused_stability_objective_cpu = False
        expected = transform(lambda x: stability_objective(f, active)(x, p, t, mu)[0])(v)
        f._fused_stability_objective_cpu = True
        actual = transform(lambda x: stability_objective(f, active)(x, p, t, mu)[0])(v)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert f._stability_objective_compiled_batches == 0


def test_unavailable_compiler_retains_reference_and_stops_retrying(monkeypatch):
    f = mixture(False)
    active = torch.arange(2)
    args = f.tensor([[0.5]]), f.tensor(1e6), f.tensor(320.0), f.tensor([0.0, 0.0])
    calls = []
    with torch.no_grad():
        f._fused_stability_objective_cpu = False
        expected = stability_objective(f, active)(*args)

        def unavailable(*args, **kwargs):
            calls.append(True)
            raise torch._dynamo.exc.Unsupported("injected compiler failure")

        monkeypatch.setattr(chemical_kernel, "_compiled_stability_values", {})
        monkeypatch.setattr(torch, "compile", unavailable)
        f._fused_stability_objective_cpu = True
        with pytest.warns(RuntimeWarning, match="Fused TPD objective unavailable"):
            compare(expected, stability_objective(f, active)(*args))
        compare(expected, stability_objective(f, active)(*args))
    assert len(calls) == 1
    assert f._stability_objective_compile_failure is not None


def test_full_multistart_stationarity_and_curvature_audits_are_preserved():
    f = mixture(True)
    z = f.tensor([[0.02, 0.1, 0.88], [0.9, 0.09, 0.01], [0.3, 0.4, 0.3]])
    p, t = f.tensor([1e6, 1e6, 2e6]), f.tensor([350.0, 320.0, 370.0])
    with torch.no_grad():
        q = torch.vmap(f.tensor_homogeneous)(p, t, z)
        mu = z.log() + q["log_fugacity_coefficients"]
        point = z[:, :-1].log() - z[:, -1:].log() + f.tensor([3.0, -3.0])
        active = torch.arange(3)
        f._fused_stability_objective_cpu = False
        expected = minimize_stability_candidates(f, p, t, mu, active, point)
        f._fused_stability_objective_cpu = True
        actual = minimize_stability_candidates(f, p, t, mu, active, point)
    torch.testing.assert_close(actual[-1], expected[-1])
    accepted = actual[-1]
    torch.testing.assert_close(actual[0][accepted], expected[0][accepted], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(actual[1][accepted], expected[1][accepted], rtol=1e-8, atol=1e-10)
    assert f._stability_objective_compiled_batches > 0
