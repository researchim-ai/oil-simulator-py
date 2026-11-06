"""Published continuous drift, frame inversion and coupled PR transport."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from reference_holmes_drift import parameters as reference

from surface.holmes_drift import HolmesGasLiquidClosure

ROOT = Path(__file__).resolve().parents[1]


def law(**options):
    return HolmesGasLiquidClosure(
        surface_tension_n_m=0.015,
        profile_parameter=1.2,
        profile_flattening_start=0.3,
        velocity_sensitivity=1.0,
        bubble_transition_start=0.2,
        bubble_transition_end=0.4,
        **options,
    )


def evaluate(alpha, velocity, theta=0.0, direction=1.0):
    closure = law()
    volumes = torch.tensor([alpha, 1 - alpha], dtype=torch.float64)
    rho = torch.tensor([12.0, 700.0], dtype=torch.float64)
    u = torch.tensor(velocity, dtype=torch.float64)
    context = {
        "diameter_m": 0.05,
        "axial_gravity_m_s2": direction * 9.80665 * math.cos(math.radians(theta)),
        "gravity_m_s2": 9.80665,
    }
    return closure, volumes, rho, u, context, closure.parameters(volumes, rho, u, **context)


@pytest.mark.parametrize("alpha", [0.01, 0.3, 0.8, 0.99])
@pytest.mark.parametrize("velocity", [-2.0, 0.0, 1.0, 30.0])
@pytest.mark.parametrize("theta", [0.0, 60.0])
def test_volume_frame_inversion_matches_independent_brent_solution(alpha, velocity, theta):
    closure, volumes, rho, u, context, r = evaluate(alpha, velocity, theta)
    if alpha == 0.99 and theta == 60.0:
        assert not r["valid"]
        assert r["mass_flux_slope_lower_bound"] < 0
        with pytest.raises(ValueError, match="monotonicity"):
            reference(
                asdict(closure),
                alpha,
                12.0,
                700.0,
                velocity,
                0.05,
                context["axial_gravity_m_s2"],
                9.80665,
            )
        return
    assert r["valid"]
    c0, vd = reference(
        asdict(closure), alpha, 12.0, 700.0, velocity, 0.05, context["axial_gravity_m_s2"], 9.80665
    )
    v = r["velocities"]
    j = float(volumes @ v)
    assert j == pytest.approx(float(r["volume_flux_m_s"]), abs=2e-11)
    assert float(v[0]) == pytest.approx(c0 * j + vd, rel=1e-10, abs=1e-10)
    assert float((volumes * rho) @ v) == pytest.approx(
        float((volumes @ rho) * u), rel=1e-12, abs=1e-10
    )
    assert float(r["distribution_parameter"]) == pytest.approx(c0, rel=1e-11)


def test_axis_reversal_is_symmetric_and_zero_mass_flow_allows_countercurrent_phases():
    *_, up = evaluate(0.3, 0.0, 45.0, 1.0)
    *_, down = evaluate(0.3, -0.0, 45.0, -1.0)
    torch.testing.assert_close(up["velocities"], -down["velocities"])
    assert up["velocities"][0] > 0 > up["velocities"][1]
    *_, fast = evaluate(0.8, 300.0)
    assert float(fast["distribution_parameter"]) == 1.0


def test_live_jacobian_and_second_derivatives_include_the_implicit_volume_frame():
    closure = law()
    context = {"diameter_m": 0.05, "axial_gravity_m_s2": 8.0, "gravity_m_s2": 9.80665}

    def fun(z):
        volumes = torch.stack((z[0], 1 - z[0]))
        return closure.velocities(volumes, z[1:3], z[3], **context)[0]

    z = torch.tensor([0.6, 12.0, 700.0, 5.0], dtype=torch.float64)
    direction = torch.tensor([0.02, 0.2, 2.0, 0.1], dtype=torch.float64)
    eps = 1e-4
    jac = torch.func.jacfwd(fun)(z)
    fd = (fun(z + eps * direction) - fun(z - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=2e-6, atol=1e-9)
    hess = torch.func.jacfwd(torch.func.jacfwd(fun))(z)
    jfd = (
        torch.func.jacfwd(fun)(z + eps * direction) - torch.func.jacfwd(fun)(z - eps * direction)
    ) / (2 * eps)
    torch.testing.assert_close(hess @ direction, jfd, rtol=2e-5, atol=1e-8)


def test_rejects_horizontal_geometry_and_out_of_range_phase_states():
    closure = law()
    with pytest.raises(ValueError, match="inclination"):
        closure.geometry(0.05, 0.0, 9.80665)
    _, volumes, rho, u, context, _ = evaluate(0.3, 1.0)
    _, valid = closure.velocities(volumes, rho, u, **{**context, "diameter_m": 0.0001})
    assert not valid
    _, valid = closure.velocities(torch.ones(3), torch.tensor([10.0, 600.0, 800.0]), u, **context)
    assert not valid


def test_inclined_pr_graph_control_schedule_restart_and_independent_audit():
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    cfg = json.loads((ROOT / "configs/surface_pr_holmes_inclined.json").read_text())
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    resumed = {
        **cfg,
        "initial": {"state": result["intervals"][-1]["state"]},
        "schedule": cfg["schedule"][:1],
    }
    continued = json.loads(json.dumps(run(resumed)))
    assert audit(resumed, continued)["accepted"]
    for interval in result["intervals"]:
        assert max(map(abs, interval["conservation_residuals"])) < 1e-7
    assert np.isfinite(result["intervals"][-1]["state"]["nodes"]["0"]["total_energy_j"])


def test_constant_profile_fast_path_remains_unique_at_high_gas_content():
    original, volumes, rho, u, context, _ = evaluate(0.99, 1.0, 60.0)
    closure = replace(original, profile_parameter=1.0)
    result = closure.parameters(volumes, rho, u, **context)
    assert result["valid"]
    c0, vd = reference(
        asdict(closure), 0.99, 12.0, 700.0, 1.0, 0.05, context["axial_gravity_m_s2"], 9.80665
    )
    assert c0 == 1.0
    assert float(result["velocities"][0]) == pytest.approx(
        float(result["volume_flux_m_s"]) + vd, abs=1e-12
    )


def test_batched_line_search_and_jacobian_match_individual_evaluations():
    closure = law()
    z = torch.tensor(
        [[0.15, 12.0, 700.0, -1.0], [0.3, 10.0, 680.0, 0.0], [0.6, 15.0, 720.0, 5.0]],
        dtype=torch.float64,
    )

    def fun(row):
        return closure.velocities(
            torch.stack((row[0], 1 - row[0])),
            row[1:3],
            row[3],
            diameter_m=0.05,
            axial_gravity_m_s2=8.0,
            gravity_m_s2=9.80665,
        )[0]

    torch.testing.assert_close(torch.vmap(fun)(z), torch.stack([fun(row) for row in z]))
    torch.testing.assert_close(
        torch.vmap(torch.func.jacfwd(fun))(z),
        torch.stack([torch.func.jacfwd(fun)(row) for row in z]),
    )
