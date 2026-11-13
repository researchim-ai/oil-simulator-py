"""Independent Stokes-profile integration, dissipation and runtime reuse."""

import numpy as np
import pytest
import torch

from surface.planar_film import mobilities, shear_free_gravity_thickness
from surface.thin_film_dynamics import ThinFilmDynamics


@pytest.mark.parametrize("slip", [0.0, 2e-6])
def test_mobilities_match_integrated_stokes_profile_and_nonnegative_viscous_work(slip):
    h, mu, force, shear = 0.00045, 0.00089, 997 * 9.81, -0.7
    y0, weights = np.polynomial.legendre.leggauss(8)
    y = (y0 + 1) * h / 2
    profile = (force * (h * y - y**2 / 2 + slip * h) + shear * (y + slip)) / mu
    flux = (profile * weights).sum() * h / 2
    surface_velocity = (force * (h**2 / 2 + slip * h) + shear * (h + slip)) / mu
    a, b, c = mobilities(torch.tensor(h, dtype=torch.float64), mu, slip)
    assert float(a * force + b * shear) == pytest.approx(flux, rel=1e-14)
    assert float(b * force + c * shear) == pytest.approx(surface_velocity, rel=1e-14)
    derivative = (force * (h - y) + shear) / mu
    heat = float((mu * derivative**2 * weights).sum() * h / 2)
    if slip:
        wall_velocity = slip * (force * h + shear) / mu
        heat += mu / slip * wall_velocity**2
    assert heat > 0
    assert force * flux + shear * surface_velocity == pytest.approx(heat, rel=1e-14)
    assert float(a * c - b.square()) > 0


def test_gravity_reference_is_actual_runtime_flux_without_artificial_inventory():
    gamma, rho, mu, g = 0.444444444444, 997.0, 0.00089, 9.81
    h = shear_free_gravity_thickness(gamma, rho, mu, g)
    model = ThinFilmDynamics((8,), (0.1,), 0.072, mu, rho, 1e-8, 0.0, 0.0)
    old = model.initialize(torch.full((8,), float(h), dtype=torch.float64))
    new, report = model.step(old, 1e-6, tangential_gravity_m_s2=(g,))
    assert new.thickness_m == old.thickness_m
    np.testing.assert_allclose(report["face_volume_flux_m2_s"][0], gamma / rho, rtol=2e-14)
    assert report["gravity_work_j"] > 0
    assert report["gravity_work_j"] == pytest.approx(report["dissipation_j"], rel=1e-12)
    assert float(shear_free_gravity_thickness(0, rho, mu, g)) == 0
    with pytest.raises(ValueError):
        shear_free_gravity_thickness(gamma, rho, -mu, g)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_reference_and_shared_mobilities_preserve_device_and_gradients():
    gamma = torch.tensor([0.01, 0.1, 0.44], dtype=torch.float64, requires_grad=True)
    cpu = shear_free_gravity_thickness(gamma, 997.0, 0.00089)
    gpu_gamma = gamma.detach().cuda().requires_grad_()
    gpu = shear_free_gravity_thickness(gpu_gamma, 997.0, 0.00089)
    assert gpu.is_cuda
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-13, atol=1e-15)
    gpu.sum().backward()
    torch.testing.assert_close(gpu_gamma.grad.cpu(), cpu.detach() / (3 * gamma.detach()))
    for a, b in zip(mobilities(cpu, 0.00089), mobilities(gpu, 0.00089), strict=True):
        assert b.is_cuda
        torch.testing.assert_close(b.cpu(), a)
