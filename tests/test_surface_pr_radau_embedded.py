"""Embedded Radau control: independent ODE, conservative stages, sparse tangent."""

import json
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from check_pr_collocation import IsochoricReference, thermal_fixture
from check_pr_collocation_walls import WallReference, wall_fixture
from check_pr_equilibrium_transport import audit
from scipy.integrate import solve_ivp

from surface import PRCollocation, PREquilibriumTransportNetwork


def temperatures(state):
    return np.array([v.temperature_k for _, v in state.nodes])


def test_checkpoint_energy_is_the_old_conservative_primary():
    model, state = thermal_fixture()
    changed = replace(
        state,
        nodes=(
            (
                state.nodes[0][0],
                replace(state.nodes[0][1], total_energy_j=state.nodes[0][1].total_energy_j + 4e-7),
            ),
            *state.nodes[1:],
        ),
    )
    model.inspect(changed)  # admissible EOS/stock consistency, not a corrupt checkpoint
    rows = []
    for old in (state, changed):
        seeds = [model._seed(i, v) for i, (_, v) in enumerate(old.nodes)]
        y, _, _, evaluate, offsets, problems, _ = model._problem(
            old, seeds, 0.0, model.default_openings, [0.0, 0.0], {}
        )
        index = offsets[0] + problems[0]["energy_row"]
        rows.append(float((evaluate(y)[0][index] * problems[0]["energy_scale"]).cpu()))
    assert rows[1] - rows[0] == pytest.approx(-4e-7, abs=2e-12)


def test_companion_defect_order_and_independent_temperature():
    model, state = thermal_fixture()
    reference = IsochoricReference(model, state)
    exact = solve_ivp(
        reference.rhs,
        (0, 0.2),
        temperatures(state),
        method="DOP853",
        rtol=1e-13,
        atol=1e-12,
        dense_output=True,
    )
    assert exact.success
    solver = PRCollocation(model, scheme="radau3", error_control="embedded", phase_fallback=False)
    defects, errors = [], []
    for dt in (0.2, 0.1, 0.05):
        final, record = solver._step(
            state,
            dt,
            model.default_openings,
            [0.0, 0.0],
            [320.0, 360.0],
            {},
        )
        defects.append(record["embedded_error_ratio"])
        errors.append(abs(temperatures(final) - exact.sol(dt)).max())
        assert max(map(abs, record["conservation_residuals"])) < 1e-7
    assert min(defects[i] / defects[i + 1] for i in (0, 1)) > 6
    assert min(errors[i] / errors[i + 1] for i in (0, 1)) > 12


def test_adaptive_uses_one_nonlinear_solve_and_independent_stage_audit():
    model, state = thermal_fixture()
    solver = PRCollocation(model, scheme="radau3", error_control="embedded", phase_fallback=False)
    result = solver.advance(state, 0.2)
    assert result["fallback_intervals"] == 0
    assert result["collocation_nonlinear_solves"] == result["attempts"]
    assert all(0 <= r["embedded_error_ratio"] <= 1 for r in result["steps"])
    config = {
        "model": model.to_dict(),
        "initial": {"state": state.to_dict()},
        "schedule": [{"duration_s": 0.2}],
    }
    report = {
        "config": config,
        "model": model.to_dict(),
        "initial": state.to_dict(),
        "intervals": [result],
        "conservation_residuals": result["conservation_residuals"],
        "total_heat_j": result["total_heat_j"],
        "total_work_j": result["total_work_j"],
    }
    assert audit(json.loads(json.dumps(config)), json.loads(json.dumps(report)))["accepted"]
    reference = IsochoricReference(model, state)
    exact = solve_ivp(
        reference.rhs, (0, 0.2), temperatures(state), method="DOP853", rtol=1e-13, atol=1e-12
    )
    actual = np.array([v["temperature_k"] for v in result["state"]["nodes"].values()])
    assert abs(actual - exact.y[:, -1]).max() < model.temperature_tolerance_k


def test_sparse_embedded_filter_matches_dense():
    outputs = []
    for limit in (128, 4):
        model, state = thermal_fixture(dense_limit=limit)
        solver = PRCollocation(
            model, scheme="radau3", error_control="embedded", phase_fallback=False
        )
        final, record = solver._step(
            state,
            0.1,
            model.default_openings,
            [0.0, 0.0],
            [320.0, 360.0],
            {},
        )
        outputs.append((temperatures(final), record["embedded_error_ratio"]))
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=0, atol=1e-9)
    assert outputs[0][1] == pytest.approx(outputs[1][1], rel=1e-5, abs=1e-5)


def test_embedded_control_rejects_other_schemes():
    model, _ = thermal_fixture()
    with pytest.raises(ValueError, match="Radau3"):
        PRCollocation(model, error_control="embedded")


def test_fluid_and_wall_temperatures_against_independent_ode():
    model, state = wall_fixture()
    ref = WallReference(model, state)
    initial = np.r_[temperatures(state), [v.wall_temperature_k for _, v in state.nodes], 0.0]
    exact = solve_ivp(ref.rhs, (0, 0.1), initial, method="DOP853", rtol=1e-13, atol=1e-12)
    assert exact.success
    report = PRCollocation(
        model, scheme="radau3", error_control="embedded", phase_fallback=False
    ).advance(state, 0.1)
    final = list(report["state"]["nodes"].values())
    actual = np.array(
        [v["temperature_k"] for v in final] + [v["wall_temperature_k"] for v in final]
    )
    assert abs(actual - exact.y[:4, -1]).max() < model.temperature_tolerance_k
    assert abs(report["total_heat_j"] - exact.y[-1, -1]) < 1e-6
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_large_plenum_roundoff_is_bounded_and_real_energy_error_is_rejected():
    from benchmark_pr_native_execution import build_case

    from surface.native_execution import native_kernel_execution

    with native_kernel_execution() as engine:
        run, check, _ = build_case(
            SimpleNamespace(case="two-phase-pipe", cells=2, duration_s=0.001),
            fused=True,
            integrator="radau3-embedded",
        )
        report = run()
        assert not engine.disabled and engine.stats["fallback_calls"] == 0
        assert len(report["intervals"][0]["steps"]) >= 2
    assert check(report)["accepted"]
    damaged = json.loads(json.dumps(report))
    damaged["intervals"][0]["steps"][0]["stage_states"][0]["inlet"]["total_energy_j"] += 0.01
    with pytest.raises(ValueError):
        check(damaged)


def test_actual_condensation_evaporation_retains_branch_fallback(monkeypatch):
    import test_surface_pr_collocation as existing

    monkeypatch.setattr(
        existing, "PRCollocation", partial(PRCollocation, scheme="radau3", error_control="embedded")
    )
    existing.test_real_condensation_and_evaporation_use_branch_fallback()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires accessible CUDA")
def test_embedded_radau_cuda_matches_cpu():
    model, state = thermal_fixture()
    gpu = PREquilibriumTransportNetwork.from_dict(model.to_dict(), device="cuda")
    results = [
        PRCollocation(m, scheme="radau3", error_control="embedded", phase_fallback=False).advance(
            state, 0.01
        )
        for m in (model, gpu)
    ]
    a, b = [list(r["state"]["nodes"].values()) for r in results]
    for x, y in zip(a, b):
        assert x["temperature_k"] == pytest.approx(y["temperature_k"], abs=1e-8)
        assert x["total_energy_j"] == pytest.approx(y["total_energy_j"], abs=1e-6)
    assert results[1]["fallback_intervals"] == 0
