"""Staged minimization retains first admissible steps and independent histories."""

import pytest
import torch
from test_surface_stability_curvature import RegularSolution

from surface.equilibrium_minimization_batch import _first_acceptable, _minimize

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", [1, 4, 5, 12])
def test_first_admissible_scale_rejected_rows_and_finished_rows(device, width):
    point = torch.zeros((5, 1), dtype=torch.float64, device=device)
    direction = torch.ones_like(point)
    p = torch.arange(1, 6, dtype=point.dtype, device=device)
    t, mu = torch.ones_like(p), torch.zeros_like(point)
    value, grad, residual = p * 0, -direction, direction
    done = torch.tensor([False, False, False, False, True], device=device)
    scales = point.new_tensor([2.0**-i for i in range(12)])
    evaluated = []

    def evaluate(trials, pressure, temperature, chemical):
        ids = pressure[:, 0].long() - 1
        threshold = point.new_tensor([1.0, 2.0**-5, 2.0**-11, -1.0, 1.0])[ids, None]
        x = trials[..., 0]
        evaluated.append(trials.numel())
        # Early nonfinite trials must remain rejected independently per row.
        cv = torch.where((ids[:, None] == 1) & (x > 0.25), torch.nan, -x)
        return (
            cv,
            torch.zeros_like(trials),
            torch.zeros_like(trials),
            None,
            {"valid": x <= threshold},
        )

    result = _first_acceptable(
        evaluate, point, direction, p, t, mu, value, grad, residual, done, scales, width
    )
    torch.testing.assert_close(
        result[:, 0], point.new_tensor([1.0, 2.0**-5, 2.0**-11, 0, 0]), rtol=0, atol=0
    )
    assert sum(evaluated) <= 60
    if width == 4:
        assert sum(evaluated) == 40  # Full reference evaluates all 5*12 points.
    assert torch.equal(point, torch.zeros_like(point))


@pytest.mark.parametrize("interaction", [0.0, 1.5, 3.0])
@pytest.mark.parametrize("compact", [False, True])
def test_multistart_bfgs_histories_match_full_line_reference(interaction, compact):
    f = RegularSolution(interaction)
    point = torch.tensor(
        [[-15.0], [-1.0], [-0.01], [0.0], [0.01], [1.0], [15.0]], dtype=torch.float64
    )
    p = torch.full((len(point),), 1e6, dtype=point.dtype)
    t = torch.full_like(p, 350.0)
    z = point.new_tensor([0.5, 0.5])
    mu = (z.log() + f.tensor_homogeneous(p[0], t[0], z)["log_fugacity_coefficients"]).expand(
        len(point), -1
    )
    active = torch.arange(2)
    f._minimization_line_width = 12
    reference = _minimize(f, p, t, mu, active, point, _compact=compact)
    f._minimization_line_width = 4
    actual = _minimize(f, p, t, mu, active, point, _compact=compact)
    torch.testing.assert_close(actual, reference, rtol=0, atol=1e-12)


@pytest.mark.parametrize("width", [0, 13, True, 1.5])
def test_invalid_line_width_rejected(width):
    f = RegularSolution(0.0)
    f._minimization_line_width = width
    point = torch.ones((2, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="line-search width"):
        _minimize(f, point[:, 0], point[:, 0], point.expand(2, 2), torch.arange(2), point)


@pytest.mark.parametrize(
    "rows,remaining,expected", [(1, 12, 12), (5, 12, 12), (6, 12, 10), (20, 12, 6), (1, 8, 8)]
)
def test_small_tails_share_one_existing_kernel_tile(rows, remaining, expected):
    from surface.equilibrium_minimization_batch import _group_width

    assert _group_width(rows, remaining, 4, 64) == expected


@pytest.mark.parametrize("device", DEVICES)
def test_real_pr_mixed_compositions_keep_all_certified_minima(device):
    import json
    from pathlib import Path

    from surface.equilibrium_minimization_batch import minimize_stability_candidates
    from surface.equilibrium_properties import EquilibriumFluid

    data = json.loads(
        (Path(__file__).parent / "data/surface/pr_restart_check_valve.json").read_text()
    )
    f = EquilibriumFluid.from_dict(data["model"]["transport"]["fluid"], device=device)
    f._compiled_chemical_cpu = device == "cpu"
    f._compact_chemical_cpu = True
    compositions = f.tensor([[0.02, 0.1, 0.88], [0.9, 0.09, 0.01], [0.3, 0.4, 0.3]]).repeat(4, 1)
    p = f.tensor([1e6, 1e6, 2e6]).repeat(4)
    t = f.tensor([350.0, 320.0, 370.0]).repeat(4)
    with torch.no_grad():
        q = f.tensor_homogeneous(p, t, compositions)
        chemical = compositions.log() + q["log_fugacity_coefficients"]
        active = torch.arange(3, device=device)
        point = compositions[:, :-1].log() - compositions[:, -1:].log() + f.tensor([3.0, -3.0])
        f._minimization_line_width = 12
        reference = minimize_stability_candidates(f, p, t, chemical, active, point)
        f._minimization_line_width = 4
        actual = minimize_stability_candidates(f, p, t, chemical, active, point)
    torch.testing.assert_close(actual[-1], reference[-1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], reference[0], rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(actual[1], reference[1], rtol=1e-8, atol=1e-10)
    assert f._chemical_compile_failure is None


@pytest.mark.parametrize(
    "rows,remaining,expected", [(1, 12, 8), (2, 12, 4), (3, 12, 12), (1, 8, 8)]
)
def test_adaptive_group_uses_small_cpu_tile_capacity(rows, remaining, expected):
    from surface.equilibrium_minimization_batch import _group_width

    assert _group_width(rows, remaining, 4, 64, 8) == expected
