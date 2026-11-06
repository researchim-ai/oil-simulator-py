"""Dedicated tests for the bounded stratified closure; no core integration."""

import math
from dataclasses import asdict

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from surface.stratified_drift import StratifiedGasLiquidClosure


def law(**kw):
    return StratifiedGasLiquidClosure(
        interfacial_friction_factor=kw.pop("interfacial_friction_factor", 0.01),
        wall_roughness_m=kw.pop("wall_roughness_m", 1e-5),
        gas_dynamic_viscosity_pa_s=kw.pop("gas_dynamic_viscosity_pa_s", 1.8e-5),
        liquid_dynamic_viscosity_pa_s=kw.pop("liquid_dynamic_viscosity_pa_s", 0.001),
        **kw,
    )


def evaluate(u=0.2, alpha=0.6, gx=0.0, closure=None, dtype=torch.float64, device="cpu"):
    t = lambda v: torch.tensor(v, dtype=dtype, device=device)
    c = closure or law()
    volumes, rho, velocity = t([alpha, 1 - alpha]), t([10.0, 800.0]), t(u)
    ctx = {
        "diameter_m": 0.1,
        "axial_gravity_m_s2": gx,
        "gravity_m_s2": 9.81,
        "phase_viscosities_pa_s": t([1.8e-5, 0.001]),
    }
    return c, volumes, rho, velocity, ctx, c.parameters(volumes, rho, velocity, **ctx)


def reference(alpha, rho, mu, u, d, gx, roughness, fi):
    """Independent SciPy solve of both pressure gradients in physical units."""
    theta = brentq(
        lambda a: (a - math.sin(a) * math.cos(a)) / math.pi - (1 - alpha), 1e-10, math.pi - 1e-10
    )
    area = math.pi * d * d / 4
    al, ag = area * (1 - alpha), area * alpha
    sl, sg, si = d * theta, d * (math.pi - theta), d * math.sin(theta)
    dl, dg = 4 * al / sl, 4 * ag / (sg + si)
    rg, rl = rho
    mg = alpha * rg / (alpha * rg + (1 - alpha) * rl)

    def wall(v, r, visc, diameter):
        re = max(r * abs(v) * diameter / visc, 2300)
        fd = (-1.8 * math.log10((roughness / (3.7 * diameter)) ** 1.11 + 6.9 / re)) ** -2
        return v * max(8 * visc / diameter, r * fd * abs(v) / 8)

    def gradients(s):
        vg, vl = u + (1 - mg) * s, u - mg * s
        ti = 8 * mu[0] / dg * s + fi * rg * s * abs(s) / 2
        return (
            rg * gx - (wall(vg, rg, mu[0], dg) * sg + ti * si) / ag,
            rl * gx - (wall(vl, rl, mu[1], dl) * sl - ti * si) / al,
        )

    s = brentq(lambda x: gradients(x)[1] - gradients(x)[0], -100, 100, xtol=1e-13)
    return np.array([u + (1 - mg) * s, u - mg * s]), gradients(s)[0]


@pytest.mark.parametrize(
    "alpha,u,gx",
    [(0.1, 0.01, 0), (0.6, 0.2, 0), (0.9, 1, 0), (0.4, -0.3, 0), (0.6, 0, 0.01), (0.6, 0.2, -0.01)],
)
def test_independent_pressure_balance_mass_frame_and_dissipation(alpha, u, gx):
    c, v, rho, velocity, ctx, r = evaluate(u, alpha, gx)
    expected, pressure = reference(
        alpha,
        rho.tolist(),
        ctx["phase_viscosities_pa_s"].tolist(),
        u,
        0.1,
        gx,
        c.wall_roughness_m,
        c.interfacial_friction_factor,
    )
    np.testing.assert_allclose(r["velocities"].numpy(), expected, rtol=1e-11, atol=1e-12)
    assert float(r["pressure_gradient_pa_m"]) == pytest.approx(pressure, rel=1e-11, abs=1e-10)
    assert r["root_valid"]
    torch.testing.assert_close(
        (v * rho * (r["velocities"] - velocity)).sum(), velocity * 0, atol=1e-12, rtol=0
    )
    torch.testing.assert_close(r["gas_pressure_gradient_pa_m"], r["liquid_pressure_gradient_pa_m"])
    assert r["gas_wall_shear_pa"] * r["velocities"][0] >= 0
    assert r["liquid_wall_shear_pa"] * r["velocities"][1] >= 0
    assert r["interfacial_shear_pa"] * r["slip_m_s"] >= 0
    # Mixture balance cancels the interface exactly.
    a = r["gas_area_m2"] + r["liquid_area_m2"]
    force = (
        (rho[0] * r["gas_area_m2"] + rho[1] * r["liquid_area_m2"]) * gx
        - r["gas_wall_shear_pa"] * r["gas_wall_perimeter_m"]
        - r["liquid_wall_shear_pa"] * r["liquid_wall_perimeter_m"]
    ) / a
    torch.testing.assert_close(force, r["pressure_gradient_pa_m"], atol=1e-10, rtol=1e-10)


def test_stop_reversal_axis_and_phase_permutation():
    _, _, _, _, _, stop = evaluate(0)
    assert stop["valid"]
    assert torch.equal(stop["velocities"], torch.zeros(2, dtype=torch.float64))
    c, v, rho, u, ctx, up = evaluate(0.2, gx=0.01)
    down = c.parameters(v, rho, -u, **{**ctx, "axial_gravity_m_s2": -0.01})
    torch.testing.assert_close(up["velocities"], -down["velocities"])
    torch.testing.assert_close(up["pressure_gradient_pa_m"], -down["pressure_gradient_pa_m"])
    perm = c.parameters(
        v.flip(0),
        rho.flip(0),
        u,
        **{**ctx, "phase_viscosities_pa_s": ctx["phase_viscosities_pa_s"].flip(0)},
    )
    torch.testing.assert_close(perm["velocities"], up["velocities"].flip(0))
    _, v, rho, u, _, stopped = evaluate(0, gx=0.001)
    assert stopped["valid"]
    assert stopped["velocities"][0] < 0 < stopped["velocities"][1]
    torch.testing.assert_close((v * rho * stopped["velocities"]).sum(), u, atol=1e-13, rtol=0)


def state_function(z):
    return law().velocities(
        torch.stack((z[0], 1 - z[0])),
        z[1:3],
        z[3],
        diameter_m=z[6],
        axial_gravity_m_s2=z[7],
        gravity_m_s2=9.81,
        phase_viscosities_pa_s=z[4:6],
    )[0]


@pytest.mark.parametrize("u", [0.001, 0.2, -0.2, 0.0])
def test_implicit_derivatives_forward_reverse_and_fd(u):
    z = torch.tensor([0.6, 10.0, 800.0, u, 1.8e-5, 0.001, 0.1, 0.0001], dtype=torch.float64)
    direction = z.new_tensor([0.02, 0.2, 3.0, 0.04, 1e-6, 0.0001, 0.01, 0.00001])
    eps = 1e-4
    jac = torch.func.jacfwd(state_function)(z)
    torch.testing.assert_close(jac, torch.func.jacrev(state_function)(z), rtol=1e-9, atol=1e-9)
    fd = (state_function(z + eps * direction) - state_function(z - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=2e-6, atol=2e-9)
    hess = torch.func.jacfwd(torch.func.jacfwd(state_function))(z)
    jfd = (
        torch.func.jacfwd(state_function)(z + eps * direction)
        - torch.func.jacfwd(state_function)(z - eps * direction)
    ) / (2 * eps)
    torch.testing.assert_close(hess @ direction, jfd, rtol=3e-5, atol=1e-5)


def test_horizontal_stop_tangent_and_batched_line_search():
    z = torch.tensor([0.6, 10.0, 800.0, 0.0, 1.8e-5, 0.001, 0.1, 0.0], dtype=torch.float64)
    jac = torch.func.jacfwd(state_function)(z)
    assert torch.isfinite(jac).all()
    eps = 1e-7
    delta = torch.zeros_like(z)
    delta[3] = eps
    torch.testing.assert_close(
        jac[:, 3],
        (state_function(z + delta) - state_function(z - delta)) / (2 * eps),
        rtol=2e-5,
        atol=1e-8,
    )
    rows = torch.stack((z, z + delta * 1e5, z - delta * 1e5))
    torch.testing.assert_close(
        torch.vmap(state_function)(rows), torch.stack([state_function(row) for row in rows])
    )
    torch.testing.assert_close(
        torch.vmap(torch.func.jacfwd(state_function))(rows),
        torch.stack([torch.func.jacfwd(state_function)(row) for row in rows]),
    )


def test_circular_segment_geometry_and_live_holdup():
    c, v, rho, u, ctx, r = evaluate(alpha=0.5)
    assert float(r["liquid_height_m"]) == pytest.approx(0.05)
    assert float(r["interface_perimeter_m"]) == pytest.approx(0.1)
    assert float(r["liquid_wall_perimeter_m"]) == pytest.approx(math.pi * 0.05)
    assert float(r["liquid_area_m2"]) == pytest.approx(math.pi * 0.1**2 / 8)
    for alpha in [0.02, 0.2, 0.8, 0.98]:
        r = c.parameters(v.new_tensor([alpha, 1 - alpha]), rho, u, **ctx)
        h = float(r["liquid_height_m"])
        theta = math.acos(1 - 2 * h / 0.1)
        assert (theta - math.sin(theta) * math.cos(theta)) / math.pi == pytest.approx(1 - alpha)


def test_rejects_kh_and_bounded_root_without_clipping_to_valid():
    _, _, _, _, _, r = evaluate(5)
    assert r["root_valid"] and not r["long_wave_stable"] and not r["valid"]
    c = law(maximum_slip_m_s=0.01)
    _, _, _, _, _, r = evaluate(0.2, closure=c)
    assert not r["bracketed"] and not r["root_valid"] and not r["valid"]
    assert abs(float(r["slip_m_s"])) <= 0.01
    _, _, _, _, _, r = evaluate(1, closure=law(maximum_phase_speed_m_s=1))
    assert not r["valid"]


@pytest.mark.parametrize(
    "change",
    [
        {"diameter_m": 0},
        {"gravity_m_s2": 0},
        {"axial_gravity_m_s2": 1},
        {"phase_viscosities_pa_s": [0, 0.001]},
        {"phase_viscosities_pa_s": [float("nan"), 0.001]},
    ],
)
def test_invalid_context_rejected(change):
    c, v, rho, u, ctx, _ = evaluate()
    r = c.parameters(v, rho, u, **{**ctx, **change})
    assert not r["valid"]
    assert torch.isfinite(r["velocities"]).all()


@pytest.mark.parametrize(
    "volumes,rho",
    [
        ([0, 1], [10, 800]),
        ([0.999, 0.001], [10, 800]),
        ([0.5, 0.5], [700, 800]),
        ([-0.1, 1.1], [10, 800]),
        ([0.6, 0.4], [float("nan"), 800]),
    ],
)
def test_rejected_phase_states(volumes, rho):
    c, v, r, u, ctx, _ = evaluate()
    values, valid = c.velocities(v.new_tensor(volumes), r.new_tensor(rho), u, **ctx)
    assert not valid
    assert torch.isfinite(values).all()


def test_requires_two_phases_and_real_dynamic_viscosities():
    c, v, rho, u, ctx, _ = evaluate()
    single, valid = c.velocities(v[:1], rho[:1], u)
    assert valid
    torch.testing.assert_close(single, u.reshape(1))
    with pytest.raises(ValueError, match="Exactly two"):
        c.velocities(v.new_ones(3), rho.new_tensor([10, 700, 800]), u, **ctx)
    declared, valid = c.velocities(
        v, rho, u, **{k: val for k, val in ctx.items() if k != "phase_viscosities_pa_s"}
    )
    assert valid
    torch.testing.assert_close(declared, c.velocities(v, rho, u, **ctx)[0])
    with pytest.raises(ValueError, match="viscosities"):
        c.velocities(v, rho, u, **{**ctx, "phase_viscosities_pa_s": [1e-5]})


@pytest.mark.parametrize(
    "kw",
    [
        {"interfacial_friction_factor": 0},
        {"wall_roughness_m": -1},
        {"maximum_inclination_deg": 11},
        {"maximum_slip_m_s": 0},
        {"model": "unknown"},
        {"minimum_gas_volume_fraction": 0},
    ],
)
def test_configuration_guards(kw):
    with pytest.raises(ValueError):
        law(**kw)


def test_constitutive_inputs_affect_solution_and_roughness_envelope():
    _, _, _, _, _, smooth = evaluate(1, closure=law(wall_roughness_m=0))
    _, _, _, _, _, rough = evaluate(1, closure=law(wall_roughness_m=1e-3))
    assert not torch.allclose(smooth["velocities"], rough["velocities"])
    _, _, _, _, _, drag = evaluate(1, closure=law(interfacial_friction_factor=0.1))
    assert abs(drag["slip_m_s"]) < abs(smooth["slip_m_s"])
    _, _, _, _, _, r = evaluate(closure=law(wall_roughness_m=0.1))
    assert not r["state_valid"] and not r["valid"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_runtime_dtype_and_available_cuda(dtype):
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        _, _, _, _, _, r = evaluate(dtype=dtype, device=device)
        assert r["valid"]
        assert r["velocities"].dtype == dtype
        assert r["velocities"].device.type == device


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_stock_scale_does_not_change_holdup_or_mass_frame(scale):
    c, v, rho, u, ctx, expected = evaluate()
    r = c.parameters(v * scale, rho, u, **ctx)
    assert r["valid"]
    torch.testing.assert_close(r["velocities"], expected["velocities"])
    torch.testing.assert_close((v * rho * (r["velocities"] - u)).sum(), u * 0, atol=1e-13, rtol=0)


def test_force_slope_is_positive_through_reversal_and_friction_transitions():
    for roughness in [0.0, 1e-5, 0.001]:
        c = law(wall_roughness_m=roughness)
        speed = torch.cat(
            (-torch.logspace(-6, 2, 100), torch.tensor([0.0]), torch.logspace(-6, 2, 100))
        ).double()
        tau, slope = torch.vmap(
            lambda v, closure=c: closure._wall(
                v, v.new_tensor(800.0), v.new_tensor(0.001), v.new_tensor(0.1)
            )
        )(speed)
        assert (slope > 0).all()
        # The input is not ordered on the negative branch; sort before checking.
        ordered = speed.argsort()
        assert (tau[ordered][1:] - tau[ordered][:-1] > 0).all()
        ad = torch.vmap(
            torch.func.jacfwd(
                lambda v, closure=c: closure._wall(
                    v, v.new_tensor(800.0), v.new_tensor(0.001), v.new_tensor(0.1)
                )[0]
            )
        )(speed)
        torch.testing.assert_close(slope, ad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("bulk,uphill", [(0.0, 0.0), (0.2, 0.001), (-0.2, -0.001), (0.0, 0.001)])
def test_public_uphill_api_and_independent_audit(bulk, uphill):
    from reference_stratified_drift import parameters as audit_parameters
    from reference_stratified_drift import velocities as audit_velocities

    c, stocks, rho, u, ctx, _ = evaluate(bulk)
    ctx = {k: v for k, v in ctx.items() if k != "phase_viscosities_pa_s"}
    ctx["axial_gravity_m_s2"] = uphill
    values, valid = c.velocities(stocks, rho, u, **ctx)
    assert valid
    diagnostics = c.parameters(stocks, rho, u, **{**ctx, "axial_gravity_m_s2": -uphill})
    torch.testing.assert_close(values, diagnostics["velocities"])
    expected = audit_velocities(asdict(c), stocks.numpy(), rho.numpy(), bulk, 0.1, uphill, 9.81)
    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-10, atol=1e-12)
    diag = audit_parameters(asdict(c), stocks.numpy(), rho.numpy(), bulk, 0.1, uphill, 9.81)
    assert diag["valid"]
    assert diag["pressure_gradient_pa_m"] == pytest.approx(
        float(diagnostics["pressure_gradient_pa_m"]), abs=1e-10
    )
    perm, pv = c.velocities(stocks.flip(0), rho.flip(0), u, **ctx)
    assert pv
    torch.testing.assert_close(perm, values.flip(0))
    np.testing.assert_allclose(
        audit_velocities(
            asdict(c), stocks.flip(0).numpy(), rho.flip(0).numpy(), bulk, 0.1, uphill, 9.81
        ),
        expected[::-1],
        rtol=1e-10,
        atol=1e-12,
    )


def test_audit_one_phase_and_rejected_kh():
    from reference_stratified_drift import velocities as audit_velocities

    np.testing.assert_equal(
        audit_velocities(asdict(law()), [1.0], [800.0], 0.2, 0.1, 0.0, 9.81), [0.2]
    )
    with pytest.raises(ValueError, match="validity"):
        audit_velocities(asdict(law()), [0.6, 0.4], [10.0, 800.0], 5.0, 0.1, 0.0, 9.81)


@pytest.mark.parametrize("key", ["gas_dynamic_viscosity_pa_s", "liquid_dynamic_viscosity_pa_s"])
def test_stored_viscosity_must_be_positive_finite(key):
    with pytest.raises(ValueError):
        law(**{key: 0.0})
    with pytest.raises(ValueError):
        law(**{key: float("nan")})
