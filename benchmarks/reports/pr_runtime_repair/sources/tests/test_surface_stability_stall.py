"""An unchanged TPD predictor may stop; unresolved roots stay unresolved."""
import pytest
import torch

from surface.equilibrium_stability_batch import stability_candidates
from test_surface_stability_curvature import RegularSolution


@pytest.mark.parametrize('interaction', [0., 1.5, 3.])
def test_stall_exit_preserves_minima_and_saddles(interaction):
    fluid = RegularSolution(interaction)
    p = torch.full((3,), 1e6, dtype=torch.float64)
    t = torch.full_like(p, 350.)
    z = torch.tensor([.5, .5], dtype=torch.float64)
    mu = z.log() + fluid.tensor_homogeneous(p[0], t[0], z)['log_fugacity_coefficients']
    args = (fluid, p, t, mu.expand(3, -1), torch.arange(2),
            torch.tensor([[-10.], [0.], [10.]], dtype=torch.float64))
    actual = stability_candidates(*args)
    reference = stability_candidates(*args, _stop_stalled=False)
    for a, b in zip(actual, reference):
        if isinstance(a, dict):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_stalled_invalid_seeds_skip_repeated_work_without_accepting():
    class Invalid(RegularSolution):
        def __init__(self):
            super().__init__(0.)
            self.calls = 0

        def tensor_homogeneous(self, *args):
            self.calls += 1
            result = super().tensor_homogeneous(*args)
            result['valid'] = torch.zeros_like(result['valid'])
            return result

    results, counts = [], []
    for stop in (True, False):
        f = Invalid()
        result = stability_candidates(f, torch.tensor([1e6], dtype=torch.float64),
            torch.tensor([350.], dtype=torch.float64),
            torch.tensor([[-.7, -.7]], dtype=torch.float64), torch.arange(2),
            torch.tensor([[1.]], dtype=torch.float64), _stop_stalled=stop)
        assert not bool(result[-1].any())
        results.append(result)
        counts.append(f.calls)
    assert counts[0] < counts[1] / 2
    for i in (0, 1, 3):
        torch.testing.assert_close(results[0][i], results[1][i], rtol=0, atol=0)
