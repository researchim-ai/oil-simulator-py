"""Nonlinear final-state resistance, independent balances and dry limits."""

import pytest
import torch
from test_surface_pr_active_annular_friction import model_state

from surface.annular_film import interfacial_conductance
from surface.pr_active_annular_friction import IMPLICIT_MODEL
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe


@pytest.mark.parametrize("dt", [1e-8, 0.01, 1.0, 1e6])
@pytest.mark.parametrize("ug,ul", [(3.0, -0.1), (-3.0, 0.1), (0.0, 0.0), (2.0, 0.1)])
def test_final_state_momentum_heat_and_independent_scalar_audit(dt, ug, ul):
    from check_pr_active_surface import friction_audit

    pipe, old = model_state(gas_u=ug, liquid_u=ul, single=True, model=IMPLICIT_MODEL)
    new, report = pipe.friction.apply(pipe, old, dt)
    friction_audit(pipe, old, new, report)
    assert min(report["physical_dissipation_j"]) >= -1e-14
    assert max(map(abs, report["conservation_residuals"])) < 1e-8
    assert abs(sum(report["interface_work_j"])) < 1e-12
    assert pipe.to_dict()["wall_friction"]["model"] == IMPLICIT_MODEL
    restored = PRActiveMultifluidPipe.from_dict(pipe.to_dict())
    assert restored.friction.to_dict() == pipe.friction.to_dict()
    diagnostic = report["annular_cells"][0]
    phases = dict(new.cells[0])

    def mass(p):
        return sum(
            n * c.molar_mass_kg_mol for n, c in zip(p.component_moles, pipe.fluid.components)
        )

    gas, liquid = phases["gas"], phases["liquid"]
    ug1, ul1 = gas.momentum_kg_m_s / mass(gas), liquid.momentum_kg_m_s / mass(liquid)
    effective, eta, *_ = interfacial_conductance(
        torch.tensor(diagnostic["alpha_l"], dtype=torch.float64),
        pipe.diameter,
        0.001,
        1e-5,
        mass(gas) / gas.volume_m3,
        torch.tensor(ug1, dtype=torch.float64),
        torch.tensor(ul1, dtype=torch.float64),
        dry_consistent=True,
    )
    assert diagnostic["tau_pa"] == pytest.approx(
        float(effective * (ug1 - eta * ul1)), abs=1e-12, rel=1e-8
    )


@pytest.mark.parametrize("speed", [1e-4, 0.05, 3.0, -3.0])
def test_fully_implicit_dry_limit_at_finite_step(speed):
    pipe, old = model_state(
        alpha=1e-12, gas_u=speed, liquid_u=0.0, single=True, model=IMPLICIT_MODEL
    )
    new, _ = pipe.friction.apply(pipe, old, 0.1)
    wet, dry = dict(new.cells[0])["gas"], dict(new.cells[1])["gas"]
    uw = wet.momentum_kg_m_s / sum(
        n * c.molar_mass_kg_mol for n, c in zip(wet.component_moles, pipe.fluid.components)
    )
    ud = dry.momentum_kg_m_s / sum(
        n * c.molar_mass_kg_mol for n, c in zip(dry.component_moles, pipe.fluid.components)
    )
    assert uw == pytest.approx(ud, abs=1e-12, rel=1e-8)


def test_roughness_outside_declared_domain_is_rejected():
    pipe, state = model_state(model=IMPLICIT_MODEL)
    pipe.friction.roughness = pipe.diameter * 0.1
    with pytest.raises(ValueError, match="roughness"):
        pipe.friction.apply(pipe, state, 0.01)


def test_implicit_time_refinement_against_independent_ode():
    import numpy as np
    from reference_pr_annular_source import final_drive_coefficient, resistance
    from scipy.integrate import solve_ivp

    pipe, initial = model_state(model=IMPLICIT_MODEL)
    f = pipe.fluid
    stocks = dict(initial.cells[0])
    masses = {
        name: sum(n * c.molar_mass_kg_mol for n, c in zip(p.component_moles, f.components))
        for name, p in stocks.items()
    }
    alpha = stocks["liquid"].volume_m3 / pipe.volumes[0]
    eta, beta, wall, perimeter = resistance(alpha, pipe.diameter, 0.001)
    rho = masses["gas"] / stocks["gas"].volume_m3
    length = pipe.lengths[0]
    y0 = [stocks[n].momentum_kg_m_s / masses[n] for n in ("gas", "liquid")] + [
        stocks["gas"].total_energy_j
    ]

    def rhs(_, y):
        ug, ul, _ = y
        drive = ug - eta * ul
        effective = final_drive_coefficient(drive, alpha, pipe.diameter, 1e-5, rho, beta, 0.0)
        tau = effective * drive
        ui = eta * ul + beta * tau
        return [
            -length * perimeter * tau / masses["gas"],
            length * (perimeter * eta * tau - wall * ul) / masses["liquid"],
            -length * perimeter * tau * ui,
        ]

    horizon = 0.02
    ref = solve_ivp(rhs, (0.0, horizon), y0, method="DOP853", rtol=1e-12, atol=1e-13)
    assert ref.success
    errors = []
    for steps in (4, 8, 16):
        state = initial
        for _ in range(steps):
            state, _ = pipe.friction.apply(pipe, state, horizon / steps)
        stocks = dict(state.cells[0])
        actual = [stocks[n].momentum_kg_m_s / masses[n] for n in ("gas", "liquid")] + [
            stocks["gas"].total_energy_j
        ]
        errors.append(np.abs(actual - ref.y[:, -1]))
    for a, b in zip(errors, errors[1:]):
        assert np.all(a / b > 1.7)
        assert np.all(a / b < 2.3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_implicit_source_cuda_matches_cpu():
    results = []
    for device in ("cpu", "cuda"):
        pipe, old = model_state(model=IMPLICIT_MODEL, single=True, device=device)
        new, report = pipe.friction.apply(pipe, old, 0.1)
        assert report["resistance_evaluation"] == "final_state"
        results.append(
            [[p.momentum_kg_m_s, p.total_energy_j] for cell in new.cells for _, p in cell]
        )
    torch.testing.assert_close(
        torch.tensor(results[0]), torch.tensor(results[1]), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("consistent,roughness", [(False, 0.0), (True, 0.0), (True, 0.0002)])
def test_analytic_stress_slope_matches_automatic_differentiation(consistent, roughness):
    alpha = torch.full((9,), 0.02, dtype=torch.float64)
    # Laminar, blend and turbulent branches, in both directions and at zero.
    core = 0.1 * (1 - 0.02) ** 0.5
    drives = torch.tensor(
        [-10000.0, -4000.0, -3000.0, -2000.0, 0.0, 1000.0, 2500.0, 4000.0, 10000.0],
        dtype=torch.float64,
    ) / (1.2 * core / 1.8e-5)
    drives.requires_grad_()
    args = (alpha, 0.1, 0.001, 1.8e-5, 1.2, drives, torch.zeros_like(drives))
    effective, *_ = interfacial_conductance(*args, dry_consistent=consistent, roughness_m=roughness)
    expected = torch.autograd.grad((effective * drives).sum(), drives)[0]
    *_, slope = interfacial_conductance(
        *args, dry_consistent=consistent, roughness_m=roughness, with_slope=True
    )
    torch.testing.assert_close(slope, expected, rtol=1e-9, atol=1e-12)
    assert (slope > 0).all()


def test_mixed_stiff_cells_match_independent_brent_roots():
    import math

    import numpy as np
    from reference_pr_annular_source import final_drive_coefficient, resistance
    from scipy.optimize import brentq

    from surface.annular_implicit_source import implicit_annular_velocities

    rng = np.random.default_rng(4314)
    count, diameter, rho, mu = 48, 0.1, 1.2, 0.001
    alpha = torch.tensor(np.exp(rng.uniform(np.log(1e-4), np.log(0.2), count)), dtype=torch.float64)
    dt_values = np.geomspace(1e-8, 1e6, count)
    dt = torch.tensor(dt_values, dtype=torch.float64)
    ug0 = torch.tensor(rng.uniform(-20, 20, count), dtype=torch.float64)
    ul0 = torch.tensor(rng.uniform(-0.1, 0.1, count), dtype=torch.float64)
    area = math.pi * diameter**2 / 4
    mg, ml = rho * area * (1 - alpha), 998 * area * alpha
    lengths = torch.ones_like(alpha)
    ug, ul, *_ = implicit_annular_velocities(
        alpha, diameter, mu, 1.8e-5, rho, ug0, ul0, mg, ml, lengths, dt, 0.0002
    )
    for j in range(count):
        a, g, l, m_g, m_l, dt_j = map(float, (alpha[j], ug0[j], ul0[j], mg[j], ml[j], dt[j]))
        eta, beta, c, perimeter = resistance(a, diameter, mu)
        den = m_l + dt_j * c
        d0 = g - eta * m_l / den * l
        coupling = dt_j * perimeter * (1 / m_g + eta**2 / den)

        def eff(d, a=a, beta=beta):
            return final_drive_coefficient(d, a, diameter, 1.8e-5, rho, beta, 0.0002)

        root = math.copysign(
            brentq(
                lambda d, coupling=coupling, eff=eff, d0=d0: d + coupling * eff(d) * d - abs(d0),
                0,
                abs(d0),
                xtol=1e-15,
            ),
            d0,
        )
        k = perimeter * eff(root)
        expected = np.linalg.solve(
            [[m_g / dt_j + k, -eta * k], [-eta * k, m_l / dt_j + eta**2 * k + c]],
            [m_g / dt_j * g, m_l / dt_j * l],
        )
        assert float(ug[j]) == pytest.approx(expected[0], rel=2e-8, abs=1e-11)
        assert float(ul[j]) == pytest.approx(expected[1], rel=2e-8, abs=1e-11)
