"""Packed diagnostics must retain the pre-optimization convergence decisions."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fim_convergence_reference import legacy_convergence

from simulator.blackoil_fim import I_B, BlackOilFIMSolver


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def data(device, bad_pairs=1):
    g = torch.Generator().manual_seed(411)
    half = 0.1 + 10 * torch.rand(32, generator=g, dtype=torch.float64)
    pv = torch.repeat_interleave(half, 2).to(device)
    v = torch.ones((20, len(pv)), dtype=torch.float64, device=device)
    v[I_B : I_B + 3] = v.new_tensor([1.1, 1.0, 20.0])[:, None]
    s = SimpleNamespace(
        rho_sc=v.new_tensor([850.0, 1000.0, 1.2]),
        tol_cnv=1e-3,
        tol_mb=1e-7,
        max_strict_iter=4,
        relaxed_max_pv_fraction=1.0,
        tol_cnv_relaxed=0.01,
        tol_mb_relaxed=1e-6,
        tol_well_rate=0.001,
        tol_well_bhp=0.1,
        _newton_iter=5,
        _prev_norm=None,
    )
    scale = 86400 * (1 / v[I_B : I_B + 3]).mean(1) / s.rho_sc
    R = torch.zeros((len(pv), 3), dtype=torch.float64, device=device)
    for j in range(bad_pairs):
        R[2 * j] = pv[2 * j] * 0.002 / scale
        R[2 * j + 1] = -R[2 * j]
    ws = {
        "is_rate": torch.tensor([True, False], device=device),
        "r_w": v.new_tensor([0.0, 0.0]),
        "target": v.new_tensor([0.0, 2e6]),
        "is_thp": torch.zeros(2, dtype=torch.bool, device=device),
        "is_stopped": torch.tensor([False, True], device=device),
        "component_rates": v.new_tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
    }
    return s, R.flatten(), pv, v, ws


def compare(s, R, pv, v, ws):
    old, expected = legacy_convergence(s, R, 86400, pv, ws, None, v)
    new, actual = BlackOilFIMSolver.convergence(s, R, 86400, pv, ws, None, v)
    assert old == new
    for key, value in expected.items():
        assert actual[key] == value, key
    assert actual["residual_finite"] == bool(torch.isfinite(R).all())
    return actual


@pytest.mark.parametrize("bad_pairs", [0, 1, 8, 32])
def test_exact_pv_fraction_boundaries(device, bad_pairs):
    s, R, pv, v, ws = data(device, bad_pairs)
    fraction = legacy_convergence(s, R, 86400, pv, ws, None, v)[1]["cnv_bad_pv_fraction"]
    for boundary in [
        fraction,
        math.nextafter(fraction, -math.inf),
        math.nextafter(fraction, math.inf),
    ]:
        s.relaxed_max_pv_fraction = boundary
        compare(s, R, pv, v, ws)
    s._newton_iter = 0
    s._prev_norm = 0.1
    compare(s, R, pv, v, ws)
    compare(s, R, pv, v, None)


@pytest.mark.parametrize("case", ["rate_bad", "bhp_bad", "thp_early", "thp_late", "stopped_cancel"])
def test_well_constraints(device, case):
    s, R, pv, v, ws = data(device, 0)
    if case == "rate_bad":
        ws["r_w"][0] = 0.001
    elif case == "bhp_bad":
        ws["r_w"][1] = 1.0
    elif case.startswith("thp"):
        ws["is_thp"][1] = True
        ws["r_w"][1] = 5000.0
        s._newton_iter = 0 if case == "thp_early" else 5
    else:
        ws["component_rates"][1] = v.new_tensor([1e-6, -1e-6, 0.0])
    compare(s, R, pv, v, ws)


def test_missing_stopped_component_contract(device):
    s, R, pv, v, ws = data(device, 0)
    del ws["component_rates"]
    ws["is_stopped"].zero_()
    compare(s, R, pv, v, ws)
    ws["is_stopped"][1] = True
    with pytest.raises(KeyError):
        legacy_convergence(s, R, 86400, pv, ws, None, v)
    with pytest.raises(KeyError):
        BlackOilFIMSolver.convergence(s, R, 86400, pv, ws, None, v)


def test_nonfinite_residual_packet(device):
    s, R, pv, v, ws = data(device, 0)
    R[0] = float("nan")
    ok, criteria = BlackOilFIMSolver.convergence(s, R, 86400, pv, ws, None, v)
    assert not ok and not criteria["residual_finite"]


def test_well_report_packet_preserves_every_published_value(device):
    from fim_convergence_reference import legacy_store_well_results

    def fake():
        wells = [
            SimpleNamespace(type="producer", eff=1.0),
            SimpleNamespace(type="injector", eff=0.5),
            SimpleNamespace(type="producer", eff=1.0, stopped=True),
        ]
        return SimpleNamespace(
            pvt=SimpleNamespace(rho_sc=(850.0, 1000.0, 1.2)),
            wells=SimpleNamespace(
                wells=wells,
                conn_well=torch.tensor([0, 1, 0, 2, 1, 0], device=device),
                _lift=lambda _i: None,
            ),
        )

    a = fake()
    b = fake()
    q = torch.tensor(
        [
            [1e10, 1.0, -1.0],
            [2.0, -3.0, 4.0],
            [-1e10, 2.0, 3.0],
            [1.0, -1.0, 0.1],
            [-2.0, 3.0, -4.0],
            [1e-6, -1e-6, 0.25],
        ],
        dtype=torch.float64,
        device=device,
    )
    p = torch.tensor([1e6, 2e6, 3e6], dtype=torch.float64, device=device)
    ws = {"q": q}
    for dt in (86400.0, 12345.0):
        legacy_store_well_results(a, ws, p, dt)
        BlackOilFIMSolver._store_well_results(b, ws, p, dt)
        for old, new in zip(a.wells.wells, b.wells.wells):
            assert vars(old) == vars(new)


def test_linear_factor_status_does_not_change_convergence(device):
    s, R, pv, v, ws = data(device, 0)
    ws["_surface_factor_bad"] = torch.tensor(True, device=device)
    criteria = compare(s, R, pv, v, ws)
    assert criteria["unrelaxed_converged"]
    assert ws["_surface_factor_validated"] is False
