"""Independent kinematic identities and explicit drift-flux validity bounds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from surface.drift_flux_closure import ZuberFindlayClosure


@pytest.mark.parametrize("velocity", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize(
    "volumes,density", [([0.2, 0.8], [10.0, 800.0]), ([0.2, 0.3, 0.5], [10.0, 800.0, 1000.0])]
)
def test_drift_relation_and_mass_momentum_are_satisfied(velocity, volumes, density):
    c = ZuberFindlayClosure(1.2, 0.15)
    a = torch.tensor(volumes, dtype=torch.float64)
    r = torch.tensor(density, dtype=torch.float64)
    v, valid = c.velocities(a, r, torch.tensor(velocity, dtype=torch.float64))
    assert bool(valid)
    j = float((a * v).sum() / a.sum())
    assert float(v[0]) == pytest.approx(1.2 * j + 0.15, abs=1e-13)
    assert float((a * r * v).sum() / (a * r).sum()) == pytest.approx(velocity, abs=1e-13)
    assert torch.equal(v[1:], v[1].expand_as(v[1:]))


def test_one_phase_limit_and_invalid_envelopes_are_explicit():
    c = ZuberFindlayClosure(1.2, 0.15)
    v, valid = c.velocities(torch.tensor([1.0]), torch.tensor([800.0]), torch.tensor(2.0))
    assert bool(valid) and float(v[0]) == 2.0
    _, valid = c.velocities(
        torch.tensor([0.9, 0.1]), torch.tensor([10.0, 800.0]), torch.tensor(2.0)
    )
    assert not bool(valid)
    with pytest.raises(ValueError):
        ZuberFindlayClosure(1.2, 0.1, maximum_gas_volume_fraction=1.0)
