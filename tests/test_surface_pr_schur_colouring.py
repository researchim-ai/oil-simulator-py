"""Graph colouring against an independent exact nonlinear tangent."""

from types import SimpleNamespace

import pytest
import torch

from surface.liquid import PropertyModelError
from surface.pr_facility_schur import FacilitySchurProblem


def problem(device="cpu", boundary=False):
    n, width = 12, 2
    p = object.__new__(FacilitySchurProblem)
    p.names = tuple(str(i) for i in range(n))
    p.nc = width - 1
    p.size = n * width
    p.jacobian_builds = 0
    p.evaluations = 0
    p.facility = SimpleNamespace(
        lines={
            str(i): {"ports": [{"column": str(i)}, {"column": str(i + 1)}]} for i in range(n - 1)
        }
    )
    matrix = torch.zeros((n * width, n * width), device=device, dtype=torch.float64)
    for i in range(n):
        for j in range(max(0, i - 1), min(n, i + 2)):
            matrix[i * width : (i + 1) * width, j * width : (j + 1) * width] = torch.tensor(
                [[3.0 + i + j, 0.7], [-0.4, 2.0 + i]], device=device, dtype=matrix.dtype
            )

    def evaluate(point):
        p.evaluations += 1
        if boundary and float(point[0].cpu()) < 0:
            raise PropertyModelError("Physical component boundary")
        return matrix @ point + 0.1 * point.pow(3), None, {}

    p.evaluate = evaluate
    return p, matrix


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("boundary", [False, True])
def test_coloured_tangent_matches_uncoloured_and_exact_ad_with_fewer_physical_samples(
    boundary, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p, exact = problem(device=device, boundary=boundary)
    point = torch.zeros(p.size, dtype=exact.dtype, device=device)
    residual = p.evaluate(point)[0]
    coloured = p.jacobian(point, residual)
    samples = p.evaluations - 1
    assert len(p._colours) == 3
    assert samples < 2 * p.size
    if boundary:
        assert p.jacobian_group_fallbacks > 0
    torch.testing.assert_close(coloured, exact, atol=2e-9, rtol=1e-10)
    uncoloured = p.jacobian(point, residual, grouped=False)
    torch.testing.assert_close(coloured, uncoloured, atol=1e-12, rtol=1e-12)
    ad = torch.func.jacfwd(lambda x: exact @ x + 0.1 * x.pow(3))(point)
    torch.testing.assert_close(coloured, ad, atol=2e-9, rtol=1e-10)
