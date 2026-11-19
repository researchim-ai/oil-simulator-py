"""Batched histories are independent and every accepted candidate is audited."""

import json
from pathlib import Path

import pytest
import torch
from test_surface_stability_curvature import RegularSolution

from surface.equilibrium_minimization_batch import minimize_stability_candidates
from surface.equilibrium_stability_batch import stability_candidates, stability_objective


@pytest.mark.parametrize("interaction", [0.0, 1.5, 3.0])
def test_minimization_keeps_certified_candidates_and_does_not_accept_saddle(interaction):
    f = RegularSolution(interaction)
    p = torch.full((5,), 1e6, dtype=torch.float64)
    t = torch.full_like(p, 350.0)
    z = torch.tensor([0.5, 0.5], dtype=torch.float64)
    chemical = z.log() + f.tensor_homogeneous(p[0], t[0], z)["log_fugacity_coefficients"]
    args = (
        f,
        p,
        t,
        chemical.expand(5, -1),
        torch.arange(2),
        torch.tensor([[-10.0], [-1.0], [0.0], [1.0], [10.0]], dtype=torch.float64),
    )
    old = stability_candidates(*args)
    new = minimize_stability_candidates(*args)
    torch.testing.assert_close(new[0][old[-1]], old[0][old[-1]], rtol=0, atol=0)
    torch.testing.assert_close(new[1][old[-1]], old[1][old[-1]], rtol=0, atol=0)
    assert bool(new[-1][old[-1]].all())
    if interaction == 3.0:
        assert not bool(new[-1][2])


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_ternary_unresolved_seeds_are_recovered_without_cross_row_histories(device):
    from surface.pr_surface_control import PRControlledSurface

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/surface_pr_integrated_facility.json").read_text())
    f = PRControlledSurface.from_dict(config["model"], device=device).plant.transport.fluid
    compositions = f.tensor([[0.02, 0.1, 0.88], [0.9, 0.09, 0.01], [0.3, 0.4, 0.3]])
    p = f.tensor([1e6, 1e6, 2e6])
    t = f.tensor([350.0, 320.0, 370.0])
    q = torch.vmap(f.tensor_homogeneous)(p, t, compositions)
    chemical = compositions.log() + q["log_fugacity_coefficients"]
    active = torch.arange(3, device=device)
    point = compositions[:, :-1].log() - compositions[:, -1:].log() + f.tensor([3.0, -3.0])
    batch = minimize_stability_candidates(f, p, t, chemical, active, point)
    for i in range(3):
        single = minimize_stability_candidates(
            f, p[i : i + 1], t[i : i + 1], chemical[i : i + 1], active, point[i : i + 1]
        )
        torch.testing.assert_close(batch[0][i], single[0][0], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(batch[1][i], single[1][0], rtol=1e-9, atol=1e-12)
        assert bool(batch[-1][i]) == bool(single[-1][0])
    selected = batch[-1]
    if bool(selected.any()):
        logits = batch[1][selected, :-1].log() - batch[1][selected, -1:].log()
        _, grad, residual, _, _ = stability_objective(f, active)(
            logits, p[selected], t[selected], chemical[selected]
        )
        assert float(grad.abs().max()) < 2e-9
        assert float(residual.abs().max()) < 1e-9


@pytest.mark.parametrize(
    "temperature,pressure,composition",
    [
        (190.0, 1e6, [0.5, 0.5]),
        (250.0, 2e6, [0.1, 0.9]),
        (320.0, 1e6, [0.9, 0.1]),
        (350.0, 5e6, [0.5, 0.5]),
    ],
)
def test_stability_minimum_matches_original_scalar_search(temperature, pressure, composition):
    from test_surface_equilibrium_thermodynamics import fluid

    f = fluid()
    p, t, z = f.tensor(pressure), f.tensor(temperature), f.tensor(composition)
    chemical = z.log() + f.tensor_homogeneous(p, t, z)["log_fugacity_coefficients"]
    active = torch.arange(2)
    actual = f._stability(p, t, z, chemical, active)
    reference = f._stability(p, t, z, chemical, active, _accelerated=False)
    assert float(actual[0]) == pytest.approx(float(reference[0]), abs=2e-8)


def test_unresolved_starts_are_recovered_but_stationary_saddle_stays_rejected(monkeypatch):
    import surface.equilibrium_minimization_batch as module

    f = RegularSolution(3.0)
    p = torch.full((3,), 1e6, dtype=torch.float64)
    t = torch.full_like(p, 350.0)
    z = torch.tensor([0.5, 0.5], dtype=torch.float64)
    mu = (z.log() + f.tensor_homogeneous(p[0], t[0], z)["log_fugacity_coefficients"]).expand(3, -1)
    args = (
        f,
        p,
        t,
        mu,
        torch.arange(2),
        torch.tensor([[-0.01], [0.0], [0.01]], dtype=torch.float64),
    )
    before = stability_candidates(*args)
    after = minimize_stability_candidates(*args)
    assert not bool(before[-1].any())
    assert after[-1].tolist() == [True, False, True]
    assert bool((after[0][after[-1]] < -1e-3).all())
    monkeypatch.setattr(module, "_minimize", lambda *a: a[-1])
    failed = minimize_stability_candidates(*args)
    # A failed extra predictor does not turn an unresolved start into success.
    torch.testing.assert_close(failed[-1], before[-1])
    torch.testing.assert_close(failed[1], before[1], rtol=0, atol=0)
