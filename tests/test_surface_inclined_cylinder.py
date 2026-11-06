"""Inclined cut geometry and conserved PR gravity profiles, independently checked."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from reference_inclined_cylinder import cut
from test_surface_stratified_pr import binary

from surface.inclined_cylinder import InclinedCylinderGeometry, geometry_from_dict
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState


@pytest.mark.parametrize("degrees", [0.0, 1e-5, 10.0, 45.0, 89.999, 90.0, -35.0])
def test_volume_centroid_and_gradient_against_independent_axial_integrals(degrees):
    g = InclinedCylinderGeometry(0.1, 0.2, inclination_rad=math.radians(degrees))
    for fraction in (0.001, 0.1, 0.37, 0.5, 0.8, 0.999):
        h = torch.tensor(fraction * g.height_m, dtype=torch.float64)
        volume, moment, area = g.cut_tensor(h)
        expected = cut(g.diameter_m, g.length_m, g.inclination_rad, float(h))
        assert float(volume) == pytest.approx(expected[0], rel=2e-9, abs=1e-14 * g.volume_m3)
        assert float(moment) == pytest.approx(
            expected[1], rel=2e-8, abs=1e-13 * g.volume_m3 * g.height_m
        )
        derivative = torch.func.jacfwd(lambda z: torch.stack(g.cut_tensor(z)[:2]))(h)
        torch.testing.assert_close(derivative, torch.stack((area, h * area)), rtol=2e-8, atol=1e-12)
        inverse = g.level_tensor(volume)
        assert float(inverse) == pytest.approx(float(h), rel=1e-9, abs=1e-13)
    assert g.liquid_volume_m3(g.height_m) == pytest.approx(g.volume_m3, rel=1e-14)
    assert geometry_from_dict(json.loads(json.dumps(g.__dict__))) == g


def test_inverse_volume_derivative_and_positive_exact_moment_quadrature():
    g = InclinedCylinderGeometry(0.1, 0.2, inclination_rad=0.4)
    c = StratifiedPRColumn(binary("cpu"), g, quadrature_order=2, maximum_quadrature_order=16)
    boundaries = torch.tensor([0.13, 0.76], dtype=torch.float64) * g.height_m
    for order in (2, 4, 8, 16):
        hh, ww = c._grid(boundaries, order)
        bounds = torch.cat(
            (boundaries.new_zeros(1), boundaries, boundaries.new_tensor([g.height_m]))
        )
        for i, (h, w) in enumerate(zip(hh, ww)):
            assert bool((w > 0).all())
            torch.testing.assert_close(
                w.sum(),
                g.volume_tensor(bounds[i + 1]) - g.volume_tensor(bounds[i]),
                rtol=1e-12,
                atol=1e-17,
            )
            torch.testing.assert_close(
                (w * h).sum(),
                g.moment_tensor(bounds[i + 1]) - g.moment_tensor(bounds[i]),
                rtol=1e-12,
                atol=1e-17,
            )
    volume = torch.tensor(0.41 * g.volume_m3, dtype=torch.float64)
    h = g.level_tensor(volume)
    derivative = torch.func.jacfwd(g.level_tensor)(volume)
    assert float(derivative * g.cut_tensor(h)[2]) == pytest.approx(1.0, rel=1e-10)


@pytest.mark.parametrize("three_phase", [False, True])
def test_live_pr_profile_energy_restart_and_independent_grid(three_phase):
    from check_stratified_pr_flow import profile

    from surface.equilibrium_properties import EquilibriumFluid

    if three_phase:
        cfg = json.loads(
            (
                Path(__file__).resolve().parents[1] / "configs/surface_equilibrium_process.json"
            ).read_text()
        )
        f = EquilibriumFluid.from_dict(cfg["fluid"])
        temperature = 350.0
        composition = [0.4, 0.3, 0.3]
    else:
        f = binary("cpu")
        temperature = 196.2
        composition = [0.19, 0.81]
    g = InclinedCylinderGeometry(0.1, 0.2, inclination_rad=0.4)
    c = StratifiedPRColumn(f, g, quadrature_order=2, maximum_quadrature_order=16)
    report = c.initialize(1e6, temperature, composition)
    s = StratifiedPRState.from_dict(report["state"])
    assert len(s.phase_roots) == (3 if three_phase else 2)
    assert report["quadrature_error_ratio"] <= 1
    restored = StratifiedPRColumn.from_dict(json.loads(json.dumps(c.to_dict())))
    restored.inspect(StratifiedPRState.from_dict(json.loads(json.dumps(s.to_dict()))))
    _, _, _, amounts, _, entropy = profile(s.to_dict(), json.loads(json.dumps(c.to_dict())))
    totals = amounts.sum(0)
    np.testing.assert_allclose(totals, s.component_moles, rtol=1e-8, atol=1e-9)
    assert entropy == pytest.approx(report["entropy_j_k"], rel=1e-9, abs=1e-8)


def test_initial_profile_rejects_interface_inside_transverse_disk_extent():
    from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork
    from surface.pr_terrain_pipe import initialize_from_gravity_profile
    from surface.separator_geometry import SeparatorGeometry

    f = binary("cpu")
    column = StratifiedPRColumn(f, SeparatorGeometry(0.1, 0.2, "vertical"))
    state = StratifiedPRState.from_dict(column.initialize(1e6, 196.2, [0.19, 0.81])["state"])
    interface = state.interface_heights_m[0]
    node = PREquilibriumNode(
        "inclined",
        math.pi * 0.1**2 / 4 * 0.1,
        "pipe_cell",
        elevation_m=interface - 0.02,
        length_m=0.1,
        diameter_m=0.1,
        elevation_change_m=0.01,
        hydrostatic_reconstruction=True,
    )
    model = PREquilibriumTransportNetwork(f, [node], gravity_m_s2=column.gravity_m_s2)
    with pytest.raises(ValueError, match="phase interfaces"):
        initialize_from_gravity_profile(model, column, state)


@pytest.mark.parametrize("angle,fraction", [(1e-5, 1e-8), (1e-7, 1e-12)])
def test_thin_small_angle_cap_has_finite_correct_first_and_second_inverse_derivatives(
    angle, fraction
):
    g = InclinedCylinderGeometry(0.1, 0.2, inclination_rad=angle)
    volume = torch.tensor(fraction * g.volume_m3, dtype=torch.float64)
    derivative = torch.func.jacfwd(g.level_tensor)
    first = derivative(volume)
    second = torch.func.jacfwd(derivative)(volume)
    assert bool(torch.isfinite(first) & torch.isfinite(second))
    height = g.level_tensor(volume)
    assert float(first * g.cut_tensor(height)[2]) == pytest.approx(1.0, rel=1e-9)
    eps = volume * 1e-3
    fd = (derivative(volume + eps) - derivative(volume - eps)) / (2 * eps)
    torch.testing.assert_close(second, fd, rtol=2e-5, atol=1e-5)
