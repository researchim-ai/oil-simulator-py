"""Pressure inversion, phase appearance, and AD through moving flash boundaries."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_surface_blackoil_properties import model

from surface.blackoil_pipe import BlackOilPipe
from surface.liquid import PropertyModelError


def test_phase_appearance_and_inverse_pressure():
    m = model()
    p = BlackOilPipe("riser", 300, 0.08, 30, 1e-5)
    rates = [0.85, 2, 0.06]
    r = p.run(3e6, rates, m)
    assert r["inlet_state"]["phase_mass_rates_kg_s"][2] == 0
    assert r["outlet_state"]["phase_mass_rates_kg_s"][2] > 0
    assert abs(r["mechanical_balance_error_w"]) < 1e-5
    inverse = p.required_inlet(r["outlet_pressure_pa"], rates, m)
    assert inverse["inlet_pressure_pa"] == pytest.approx(3e6, abs=0.001)
    assert abs(r["length_residual_m"]) < 1e-5
    fwd = r["pressure_derivatives"]
    back = inverse["pressure_derivatives"]
    assert fwd["boundary_pressure_pa"] * back["boundary_pressure_pa"] == pytest.approx(1, rel=1e-8)
    for comp in ("oil_kg_s", "water_kg_s", "gas_kg_s"):
        assert fwd[comp] + fwd["boundary_pressure_pa"] * back[comp] == pytest.approx(0, abs=0.001)


@pytest.mark.parametrize("outlet", [False, True])
def test_pressure_jacobian_with_moving_bubble_point(outlet):
    m = model()
    p = BlackOilPipe("riser", 300, 0.08, 30, 1e-5)
    run = p.required_inlet if outlet else p.run
    boundary = 2.72e6 if outlet else 3.1e6
    x = np.array([boundary, 0.85, 2, 0.06])
    r = run(x[0], x[1:], m)
    key = "inlet_pressure_pa" if outlet else "outlet_pressure_pa"
    for j, (name, h) in enumerate(
        zip(("boundary_pressure_pa", "oil_kg_s", "water_kg_s", "gas_kg_s"), (2.0, 1e-5, 1e-5, 1e-7))
    ):
        a = x.copy()
        b = x.copy()
        a[j] += h
        b[j] -= h
        fd = (run(a[0], a[1:], m)[key] - run(b[0], b[1:], m)[key]) / (2 * h)
        assert r["pressure_derivatives"][name] == pytest.approx(fd, rel=3e-6, abs=0.002)


def test_outside_pressure_coverage_and_geometry():
    with pytest.raises(PropertyModelError, match="upward"):
        BlackOilPipe("downhill", 100, 0.1, -10)
    with pytest.raises(PropertyModelError, match="coverage"):
        BlackOilPipe("huge_rise", 10000, 0.08, 9000).required_inlet(1e6, [0.85, 2, 0.06], model())


def test_laminar_branch_does_not_evaluate_invalid_turbulent_root():
    import torch

    from surface.blackoil_pipe import _darcy

    x = torch.tensor(
        [1e-12, 1e-6, 0.01, 1.0, 100.0, 1000.0, 2000.0], dtype=torch.float64, requires_grad=True
    )
    f = _darcy(x, 0.0)
    assert torch.isfinite(f).all()
    assert torch.allclose(f, 64 / x, rtol=1e-14)
    grad = torch.autograd.grad(f.sum(), x)[0]
    assert torch.allclose(grad, -64 / x.square(), rtol=1e-14)


def test_single_liquid_laminar_pressure_and_composition_endpoints():
    import torch

    m = model()
    p = BlackOilPipe("laminar", 100, 0.01)
    # Water PVT in the fixture is weakly compressible. Pressure-coordinate
    # inversion is cross-checked in both directions at low Re.
    with torch.no_grad():
        r = p.run(2.4e6, [0, 0.001, 0], m)
    assert 2.39e6 < r["outlet_pressure_pa"] < 2.4e6
    assert abs(r["mechanical_balance_error_w"]) < 1e-8
    back = p.required_inlet(r["outlet_pressure_pa"], [0, 0.001, 0], m)
    assert back["inlet_pressure_pa"] == pytest.approx(2.4e6, abs=0.001)


def test_choking_is_physical_capacity_and_numerical_failure_is_not_hidden(monkeypatch):
    from surface.blackoil_pipe import ChokedFlowError
    from surface.liquid import EquipmentOperatingError, HydraulicConvergenceError

    p = BlackOilPipe("choke", 100, 0.01)
    with pytest.raises(ChokedFlowError):
        p.required_inlet(2e6, [0, 0, 1.0], model())
    assert issubclass(ChokedFlowError, EquipmentOperatingError)
    count = []

    def failure(*_args, **_kwargs):
        count.append(1)
        raise HydraulicConvergenceError("injected quadrature failure")

    monkeypatch.setattr(BlackOilPipe, "_checked_integral", failure)
    with pytest.raises(HydraulicConvergenceError, match="injected"):
        BlackOilPipe("numeric", 300, 0.08, 30).run(3.1e6, [0.85, 2, 0.06], model())
    assert len(count) == 1
