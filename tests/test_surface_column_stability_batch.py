"""Column collocation batches preserve local stability and pure-root checks."""

import pytest
import torch
from test_surface_equilibrium_thermodynamics import fluid

from surface.liquid import HydraulicConvergenceError, PropertyModelError
from surface.separator_geometry import SeparatorGeometry
from surface.stratified_pr import StratifiedPRColumn

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def points(f, temperature, fractions):
    pressures = f.tensor([1e6 + i * 100 for i in range(len(fractions))])
    x = f.tensor(fractions)
    t = f.tensor(temperature)
    properties = torch.vmap(f.tensor_homogeneous)(pressures, t.expand(len(x)), x)
    return [pressures], t, [x], [properties]


@pytest.mark.parametrize("device", DEVICES)
def test_column_batches_every_unconfirmed_mixture_and_retains_pure_flash(device, monkeypatch):
    f = fluid(device)
    column = StratifiedPRColumn(f, SeparatorGeometry(0.1, 1.0, "vertical"))
    args = points(f, 320.0, [[0.2, 0.8], [0.3, 0.7], [1.0, 0.0]])
    monkeypatch.setattr(
        f, "homogeneous_stability_mask", lambda p, *a: torch.zeros_like(p, dtype=torch.bool)
    )
    original_many, original_flash = f._stability_many, f.flash_pt
    calls, pure = [], []

    def many(requests):
        calls.append(len(requests))
        return original_many(requests)

    def flash(*args, **kwargs):
        pure.append(True)
        return original_flash(*args, **kwargs)

    monkeypatch.setattr(f, "_stability_many", many)
    monkeypatch.setattr(f, "flash_pt", flash)
    column._check_locals(*args)
    assert calls == [2] and len(pure) == 1
    column._batch_local_stability = False
    column._check_locals(*args)
    assert calls == [2] and len(pure) == 2


@pytest.mark.parametrize("device", DEVICES)
def test_column_rejects_unstable_phase_on_both_paths(device, monkeypatch):
    f = fluid(device)
    column = StratifiedPRColumn(f, SeparatorGeometry(0.1, 1.0, "vertical"))
    args = points(f, 196.2044254, [[0.5, 0.5], [0.5, 0.5]])
    monkeypatch.setattr(
        f, "homogeneous_stability_mask", lambda p, *a: torch.zeros_like(p, dtype=torch.bool)
    )
    for enabled in (False, True):
        column._batch_local_stability = enabled
        with pytest.raises(PropertyModelError, match="locally stable phase topology"):
            column._check_locals(*args)


def test_column_never_swallows_unresolved_search_failure(monkeypatch):
    f = fluid()
    column = StratifiedPRColumn(f, SeparatorGeometry(0.1, 1.0, "vertical"))
    args = points(f, 320.0, [[0.2, 0.8], [0.3, 0.7]])
    monkeypatch.setattr(
        f, "homogeneous_stability_mask", lambda p, *a: torch.zeros_like(p, dtype=torch.bool)
    )
    monkeypatch.setattr(
        f, "_stability_many", lambda requests: [HydraulicConvergenceError("search exhausted")]
    )
    with pytest.raises(HydraulicConvergenceError, match="search exhausted"):
        column._check_locals(*args)
