"""Finite-inertia source terms: analytic limits, EOS energy and entropy."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from surface.equilibrium_properties import EquilibriumFluid
from surface.phase_momentum_exchange import PhaseMomentumExchange
from surface.pr_interphase_relaxation import PRInterphaseRelaxation, PRRelaxationState

ROOT = Path(__file__).resolve().parents[1]


def setup(*, drag=0.05, pressure=1e-10, thermal=0.1, wall=(0.0, 0.0), three=False):
    fluid = EquilibriumFluid.from_dict(
        json.loads((ROOT / "configs/surface_pr_signed_pump_reversal.json").read_text())["model"][
            "fluid"
        ]
    )
    n = 3 if three else 2
    pair = lambda value: tuple(tuple(0.0 if i == j else value for j in range(n)) for i in range(n))
    model = PRInterphaseRelaxation(
        fluid,
        ("vapor", "liquid", "liquid")[:n],
        pair(drag),
        wall_drag_kg_s=wall if not three else (0.0, 0.02, 0.0),
        pressure_conductance_m3_pa_s=pair(pressure),
        heat_conductance_w_k=pair(thermal),
    )
    phases = [
        {
            "pressure_pa": 1.2e6,
            "temperature_k": 340.0,
            "volume_m3": 0.0004,
            "mole_fractions": [1.0, 0.0, 0.0],
            "velocity_m_s": 10.0,
        },
        {
            "pressure_pa": 0.8e6,
            "temperature_k": 330.0,
            "volume_m3": 0.0006,
            "mole_fractions": [0.0, 1.0, 0.0],
            "velocity_m_s": -1.0,
        },
    ]
    if three:
        phases.append(
            {
                "pressure_pa": 1e6,
                "temperature_k": 335.0,
                "volume_m3": 0.0003,
                "mole_fractions": [0.0, 0.0, 1.0],
                "velocity_m_s": 0.5,
            }
        )
    return model, model.initialize(phases), phases


@pytest.mark.parametrize("coefficient", [0.0, 0.1, 1e8, 1e16])
def test_two_phase_analytic_backward_euler_and_stiff_limit(coefficient):
    m = torch.tensor([0.01, 1.0], dtype=torch.float64)
    old = torch.tensor([10.0, -1.0], dtype=torch.float64)
    model = PhaseMomentumExchange(((0, coefficient), (coefficient, 0)), (0, 0))
    v, d = model.solve(m, old, 0.01)
    mean = float((m * old).sum() / m.sum())
    slip = float((old[0] - old[1]) / (1 + 0.01 * coefficient * (1 / m[0] + 1 / m[1])))
    assert float(v[0]) == pytest.approx(mean + float(m[1] / m.sum()) * slip, abs=1e-13)
    assert float(v[1]) == pytest.approx(mean - float(m[0] / m.sum()) * slip, abs=1e-13)
    assert d["valid"]
    assert float(d["momentum_residual"].abs().max()) < 1e-12
    lost = (0.5 * m * (old.square() - v.square())).sum()
    assert float(lost) == pytest.approx(float(d["internal_energy_gain_j"].sum()), abs=1e-12)


def test_batched_three_phase_wall_impulse_and_permutation():
    m = torch.tensor([[0.01, 0.5, 0.2], [0.1, 0.3, 0.9]], dtype=torch.float64)
    v = torch.tensor([[3.0, -1.0, 0.2], [-2.0, 0.4, 1.0]], dtype=torch.float64)
    k = ((0, 0.1, 0.03), (0.1, 0, 0.2), (0.03, 0.2, 0))
    model = PhaseMomentumExchange(k, (0, 0.1, 0.02))
    end, d = model.solve(m, v, 0.03)
    assert d["valid"].all() and float(d["momentum_residual"].abs().max()) < 1e-12
    assert torch.allclose(
        (m * (end - v)).sum(-1), 0.03 * d["wall_force_n"].sum(-1), atol=1e-12, rtol=0
    )
    assert torch.allclose(
        (0.5 * m * (v * v - end * end)).sum(-1),
        d["internal_energy_gain_j"].sum(-1),
        atol=1e-12,
        rtol=0,
    )
    permutation = [2, 0, 1]
    other = PhaseMomentumExchange(
        tuple(tuple(k[i][j] for j in permutation) for i in permutation),
        tuple((0, 0.1, 0.02)[i] for i in permutation),
    )
    actual, _ = other.solve(m[:, permutation], v[:, permutation], 0.03)
    assert torch.allclose(actual, end[:, permutation], atol=1e-12, rtol=0)


def test_local_kernel_derivatives_and_disconnected_phase():
    m = torch.tensor([0.1, 1.0, 0.3], dtype=torch.float64)
    v = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    law = PhaseMomentumExchange(((0, 0.1, 0), (0.1, 0, 0), (0, 0, 0)), (0, 0, 0))
    jac = torch.func.jacrev(lambda x: law.solve(m, x, 0.1)[0])(v)
    for col in range(3):
        delta = torch.zeros_like(v)
        delta[col] = 1e-3
        fd = (law.solve(m, v + delta, 0.1)[0] - law.solve(m, v - delta, 0.1)[0]) / 2e-3
        assert torch.allclose(jac[:, col], fd, atol=1e-10, rtol=1e-8)
    assert float(law.solve(m, v, 0.1)[0][2]) == pytest.approx(float(v[2]), abs=1e-14)


@pytest.mark.parametrize("three", [False, True])
def test_coupled_pressure_heat_and_momentum_keep_closed_stocks(three):
    model, state, _ = setup(three=three)
    before = state.to_dict()
    end, report = model.step(state, 0.001)
    assert state.to_dict() == before
    assert [s.component_moles for s in end.phases] == [s.component_moles for s in state.phases]
    assert abs(report["volume_residual_m3"]) < 1e-12
    assert abs(report["energy_residual_j"]) < 1e-7
    assert abs(report["momentum_residual_kg_m_s"]) < 1e-12
    assert report["entropy_change_j_k"] >= report["entropy_source_lower_bound_j_k"] - 1e-8
    assert report["entropy_change_j_k"] > 0
    assert abs(end.phases[0].pressure_pa - end.phases[1].pressure_pa) < 4e5
    assert all(v >= 0 for v in report["physical_dissipation_j"] + report["temporal_dissipation_j"])


def test_coupled_source_jacobian_has_no_nested_solver(monkeypatch):
    model, state, _ = setup()
    y, _, _, evaluate = model._problem(state, 0.001)

    def forbidden(*args, **kwargs):
        raise AssertionError("Nested equilibrium/relaxation solver in residual")

    monkeypatch.setattr(model, "step", forbidden)
    monkeypatch.setattr(model.fluid, "flash_pt", forbidden)
    jac = torch.func.jacrev(lambda z: evaluate(z)[0])(y)
    for col in range(len(y)):
        h = 1e-6 * max(1.0, abs(float(y[col])))
        d = torch.zeros_like(y)
        d[col] = h
        fd = (evaluate(y + d)[0] - evaluate(y - d)[0]) / (2 * h)
        assert torch.allclose(jac[:, col], fd, atol=3e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", ["zero", "velocity", "pressure", "heat"])
def test_zero_and_stiff_source_limits(kind):
    args = {"drag": 0.0, "pressure": 0.0, "thermal": 0.0}
    if kind == "velocity":
        args["drag"] = 1e16
    if kind == "pressure":
        args["pressure"] = 1e12
    if kind == "heat":
        args["thermal"] = 1e20
    model, state, _ = setup(**args)
    end, r = model.step(state, 0.001)
    mass = [
        sum(n * c.molar_mass_kg_mol for n, c in zip(p.component_moles, model.fluid.components))
        for p in state.phases
    ]
    velocity = [p.momentum_kg_m_s / m for p, m in zip(end.phases, mass)]
    if kind == "zero":
        for a, b in zip(state.phases, end.phases):
            assert a.pressure_pa == pytest.approx(b.pressure_pa, rel=1e-12)
            assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-9)
            assert a.momentum_kg_m_s == pytest.approx(b.momentum_kg_m_s, abs=1e-12)
    elif kind == "velocity":
        assert velocity[0] == pytest.approx(velocity[1], abs=1e-11)
    elif kind == "pressure":
        assert end.phases[0].pressure_pa == pytest.approx(end.phases[1].pressure_pa, abs=1e-4)
    else:
        assert end.phases[0].temperature_k == pytest.approx(end.phases[1].temperature_k, abs=1e-8)
    assert abs(r["energy_residual_j"]) < 1e-7


def test_adaptive_interval_restart_and_invalid_state_are_atomic():
    model, state, _ = setup()
    report = model.advance(state, 1e-5, maximum_attempts=1)
    end = PRRelaxationState.from_dict(json.loads(json.dumps(report["state"])))
    assert end.time_s == 1e-5
    restored = PRInterphaseRelaxation.from_dict(json.loads(json.dumps(model.to_dict())))
    restored.inspect(end)
    next_state, _ = restored.step(end, 1e-5)
    assert next_state.time_s == 2e-5
    bad = replace(
        state,
        phases=(
            replace(state.phases[0], total_energy_j=state.phases[0].total_energy_j + 1.0),
            state.phases[1],
        ),
    )
    with pytest.raises(ValueError):
        model.step(state, 0.0)
    from surface.liquid import PropertyModelError

    with pytest.raises(PropertyModelError):
        model.step(bad, 1e-4)


def test_relaxation_temporal_convergence_to_continuous_drag_solution():
    model, state, _ = setup(pressure=0.0, thermal=0.0)
    masses = [
        sum(n * c.molar_mass_kg_mol for n, c in zip(p.component_moles, model.fluid.components))
        for p in state.phases
    ]
    mean = sum(p.momentum_kg_m_s for p in state.phases) / sum(masses)
    rate = 0.05 * (1 / masses[0] + 1 / masses[1])
    import math

    target = mean + masses[1] / sum(masses) * 11.0 * math.exp(-rate * 0.004)
    errors = []
    for count in (2, 4, 8):
        current = state
        for _ in range(count):
            current, _ = model.step(current, 0.004 / count)
        errors.append(abs(current.phases[0].momentum_kg_m_s / masses[0] - target))
    assert 1.8 < errors[0] / errors[1] < 2.1
    assert 1.8 < errors[1] / errors[2] < 2.1


def test_caloric_reference_does_not_change_relaxation_physics():
    model, state, spec = setup()
    raw = model.to_dict()
    for component in raw["fluid"]["components"]:
        component["reference_temperature_k"] = 300.0
        component["reference_pressure_pa"] = 2e5
    other = PRInterphaseRelaxation.from_dict(raw)
    old = other.initialize(spec)
    a, _ = model.step(state, 0.001)
    b, _ = other.step(old, 0.001)
    for x, y in zip(a.phases, b.phases):
        assert x.pressure_pa == pytest.approx(y.pressure_pa, rel=1e-10)
        assert x.temperature_k == pytest.approx(y.temperature_k, abs=1e-8)
        assert x.momentum_kg_m_s == pytest.approx(y.momentum_kg_m_s, abs=1e-12)


@pytest.mark.parametrize(
    "name", ["interphase_relaxation", "interphase_three_phase", "interphase_stiff"]
)
def test_independent_runner_audit_and_tamper_detection(name, monkeypatch):
    from check_pr_interphase_relaxation import audit
    from run_pr_interphase_relaxation import run

    cfg = json.loads((ROOT / f"configs/surface_pr_{name}.json").read_text())
    report = json.loads(json.dumps(run(cfg)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent source audit invoked runtime solving")

    monkeypatch.setattr(PRInterphaseRelaxation, "initialize", forbidden)
    monkeypatch.setattr(PRInterphaseRelaxation, "inspect", forbidden)
    monkeypatch.setattr(PRInterphaseRelaxation, "step", forbidden)
    monkeypatch.setattr(EquilibriumFluid, "tensor_homogeneous", forbidden)
    assert audit(cfg, report)["accepted"]
    for field in ("pair_impulses_kg_m_s", "physical_dissipation_j", "volume_transfers_m3"):
        damaged = json.loads(json.dumps(report))
        values = damaged["intervals"][0]["steps"][0][field]
        if field == "pair_impulses_kg_m_s":
            values[0][1] += 0.01
        elif values:
            values[0] += 0.01
        else:
            values.append(0.01)
        with pytest.raises(ValueError):
            audit(cfg, damaged)
    changed = json.loads(json.dumps(cfg))
    changed["initial"]["phases"][0]["velocity_m_s"] += 1.0
    damaged = json.loads(json.dumps(report))
    damaged["config"] = changed
    with pytest.raises(ValueError):
        audit(changed, damaged)


def test_full_source_runner_restart_and_budget_failure_are_atomic(monkeypatch):
    from check_pr_interphase_relaxation import audit
    from run_pr_interphase_relaxation import run

    from surface.liquid import HydraulicConvergenceError

    cfg = json.loads((ROOT / "configs/surface_pr_interphase_relaxation.json").read_text())
    report = json.loads(json.dumps(run(cfg)))
    request = {**cfg, "initial": {"state": report["state"]}, "schedule": [{"duration_s": 0.0001}]}
    continued = json.loads(json.dumps(run(request)))
    assert continued["initial"] == report["state"]
    assert audit(request, continued)["accepted"]
    model = PRInterphaseRelaxation.from_dict(cfg["model"])
    state = PRRelaxationState.from_dict(report["initial"])
    before = state.to_dict()

    def rejected(*args, **kwargs):
        raise HydraulicConvergenceError("forced source rejection")

    monkeypatch.setattr(model, "step", rejected)
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        model.advance(state, 0.001, maximum_attempts=1)
    assert state.to_dict() == before


def test_wall_free_drag_is_galilean_invariant_at_large_bulk_speed():
    law = PhaseMomentumExchange(
        ((0.0, 0.1, 0.03), (0.1, 0.0, 0.2), (0.03, 0.2, 0.0)), (0.0, 0.0, 0.0)
    )
    mass = torch.tensor([0.01, 0.5, 0.2], dtype=torch.float64)
    velocity = torch.tensor([3.0, -1.0, 0.5], dtype=torch.float64)
    _, base = law.solve(mass, velocity, 0.03)
    _, boost = law.solve(mass, velocity + 1e8, 0.03)
    for field in ("pair_force_n", "physical_dissipation_j", "temporal_dissipation_j"):
        assert torch.allclose(base[field], boost[field], atol=1e-11, rtol=1e-10)
