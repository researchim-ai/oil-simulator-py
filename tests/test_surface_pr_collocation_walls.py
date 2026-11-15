"""Finite wall stocks in common PR stages, conservation and safe restart."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from audit_pr_collocation import audit_gauss_step
from check_pr_collocation_walls import check, wall_fixture

from surface import (
    PRCollocation,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from surface.liquid import HydraulicConvergenceError, PropertyModelError
from surface.vessel_wall import VesselWall


def step(model, state, dt=0.05):
    return PRCollocation(model, phase_fallback=False)._step(
        state,
        dt,
        model.default_openings,
        [0.0] * len(model.nodes),
        [s.temperature_k for _, s in state.nodes],
        {},
    )


def test_independent_pr_wall_ode_and_order():
    summary, _ = check()
    assert min(summary["observed_refinement_ratios"]) > 14
    assert summary["independent_max_step_energy_defect_j"] < 1e-6


@pytest.mark.parametrize("partial", [False, True])
def test_adiabatic_total_fluid_and_wall_energy_and_sparse_match(partial):
    model, state = wall_fixture(insulated=True, partial=partial)
    new, record = step(model, state)
    assert record["ambient_heat_j"] == 0.0
    energy = sum(
        b.total_energy_j
        - a.total_energy_j
        + (
            n.wall.heat_capacity_j_k * (b.wall_temperature_k - a.wall_temperature_k)
            if n.wall
            else 0.0
        )
        for n, (_, a), (_, b) in zip(model.nodes, state.nodes, new.nodes)
    )
    assert abs(energy) < 1e-7
    assert set(record["wall_exchanges"]) == ({"a"} if partial else {"a", "b"})
    audit_gauss_step(
        model.to_dict(), state.to_dict(), record, model.fluid, {"openings": model.default_openings}
    )
    sparse = PREquilibriumTransportNetwork.from_dict({**model.to_dict(), "dense_limit": 4})
    other = replace(state, model_id=sparse.model_id)
    final, _ = step(sparse, other)
    for (_, a), (_, b) in zip(new.nodes, final.nodes):
        assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-9)
        assert a.wall_temperature_k == pytest.approx(b.wall_temperature_k, abs=1e-9)


def test_common_wall_stage_jacobian_and_sparse_support(monkeypatch):
    model, state = wall_fixture(dense_limit=4)
    solver = PRCollocation(model, phase_fallback=False)
    original = solver._solve
    checks = []

    def audit(fn, initial, lo, hi, pattern, update):
        jac = torch.func.jacfwd(fn)(initial)
        direction = torch.linspace(-0.01, 0.01, len(initial), dtype=initial.dtype)
        fd = (fn(initial + 1e-5 * direction) - fn(initial - 1e-5 * direction)) / (2e-5)
        assert float((jac @ direction - fd).abs().max()) < 1e-6
        mask = torch.zeros_like(jac, dtype=torch.bool)
        mask[pattern[0], pattern[1]] = True
        assert float(jac[~mask].abs().max()) < 1e-13
        checks.append(1)
        return original(fn, initial, lo, hi, pattern, update)

    monkeypatch.setattr(solver, "_solve", audit)
    solver._step(state, 0.02, model.default_openings, [0.0, 0.0], [320.0, 360.0], {})
    assert len(checks) == 2


@pytest.mark.parametrize("kind", ["wall_stage", "wall_endpoint", "ambient", "transfer", "registry"])
def test_independent_auditor_rejects_corrupt_wall_state_and_heat(kind, monkeypatch):
    model, state = wall_fixture()
    _, record = step(model, state)
    damaged = copy.deepcopy(record)
    if kind == "wall_stage":
        damaged["stage_states"][0]["a"]["wall_temperature_k"] += 1.0
    elif kind == "wall_endpoint":
        damaged["state"]["nodes"]["a"]["wall_temperature_k"] += 1.0
    elif kind == "ambient":
        damaged["ambient_heat_j"] += 1.0
    elif kind == "transfer":
        damaged["wall_exchanges"]["a"]["fluid_heat_j"] += 1.0
    else:
        del damaged["wall_exchanges"]["a"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Production physics called by auditor")

    monkeypatch.setattr(PREquilibriumTransportNetwork, "_fluxes", forbidden)
    monkeypatch.setattr(PREquilibriumTransportNetwork, "inspect", forbidden)
    monkeypatch.setattr(type(model.fluid), "flash_uv", forbidden)
    monkeypatch.setattr(type(model.fluid), "flash_pt", forbidden)
    assert audit_gauss_step(
        model.to_dict(), state.to_dict(), record, model.fluid, {"openings": model.default_openings}
    )["accepted"]
    with pytest.raises(ValueError):
        audit_gauss_step(
            model.to_dict(),
            state.to_dict(),
            damaged,
            model.fluid,
            {"openings": model.default_openings},
        )


def test_detached_wall_ambient_has_exact_inventory_and_fluid_is_unchanged():
    model, state = wall_fixture()
    nodes = [replace(n, wall=replace(n.wall, fluid_ua_w_k=0.0)) for n in model.nodes]
    model = PREquilibriumTransportNetwork(
        model.fluid,
        nodes,
        initial_step_s=0.02,
        maximum_step_s=0.25,
        relative_tolerance=1e-7,
        temperature_tolerance_k=1e-5,
    )
    state = replace(state, model_id=model.model_id)
    result = PRCollocation(model, phase_fallback=False).advance(state, 0.1)
    final = PREquilibriumTransportState.from_dict(result["state"])
    for node, (_, before), (_, after) in zip(model.nodes, state.nodes, final.nodes):
        w = node.wall
        exact = w.ambient_temperature_k + (
            before.wall_temperature_k - w.ambient_temperature_k
        ) * np.exp(-w.ambient_ua_w_k / w.heat_capacity_j_k * 0.1)
        assert after.wall_temperature_k == pytest.approx(exact, abs=1e-8)
        assert after.temperature_k == pytest.approx(before.temperature_k, abs=1e-9)
    assert abs(result["conservation_residuals"][-1]) < 1e-7


def test_wall_limits_reject_step_without_mutating_state():
    model, state = wall_fixture()
    nodes = [replace(n, wall=replace(n.wall, max_temperature_change_k=1e-5)) for n in model.nodes]
    model = PREquilibriumTransportNetwork(
        model.fluid, nodes, initial_step_s=0.02, maximum_step_s=0.25, maximum_attempts=1
    )
    state = replace(state, model_id=model.model_id)
    before = state.to_dict()
    solver = PRCollocation(model, phase_fallback=False)
    with pytest.raises(PropertyModelError, match="wall temperature"):
        solver._step(state, 0.1, model.default_openings, [0.0, 0.0], [320.0, 360.0], {})
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        solver.advance(state, 0.1)
    assert state.to_dict() == before


def test_wall_restart_split_and_constant_heater_energy():
    model, state = wall_fixture(insulated=True)
    solver = PRCollocation(model, phase_fallback=False)
    commands = {"heat_w": {"a": 10.0}, "utility_temperature_k": {"a": 500.0}}
    full = solver.advance(state, 0.1, **commands)
    first = solver.advance(state, 0.043, **commands)
    resumed = solver.advance(
        PREquilibriumTransportState.from_dict(json.loads(json.dumps(first["state"]))),
        0.057,
        **commands,
    )
    a = PREquilibriumTransportState.from_dict(full["state"])
    b = PREquilibriumTransportState.from_dict(resumed["state"])
    for (_, x), (_, y) in zip(a.nodes, b.nodes):
        assert x.temperature_k == pytest.approx(y.temperature_k, abs=1e-5)
        assert x.wall_temperature_k == pytest.approx(y.wall_temperature_k, abs=1e-5)
    assert full["total_heat_j"] == pytest.approx(1.0, abs=1e-12)
    assert abs(full["conservation_residuals"][-1]) < 1e-7


def test_two_phase_pipe_wall_at_rest_and_uniform_periodic_motion():
    from test_surface_pr_equilibrium_transport import pipe

    for speed in (0.0, 1.0):
        original, state = pipe(196.2, speed, True)
        nodes = [
            replace(n, wall=VesselWall(100.0, 20.0, max_temperature_change_k=20.0))
            for n in original.nodes
        ]
        model = PREquilibriumTransportNetwork(original.fluid, nodes, original.links)
        initial = {
            n.name: {
                "pressure_pa": 1e6,
                "temperature_k": 196.2,
                "mole_fractions": [0.5, 0.5],
                "velocity_m_s": speed,
                "wall_temperature_k": 210.0,
            }
            for n in model.nodes
        }
        state = model.initialize(initial)
        final, record = step(model, state, 1e-5)
        assert all(len(s.phase_roots) == 2 for _, s in final.nodes)
        assert abs(record["conservation_residuals"][-1]) < 1e-7
        audit_gauss_step(
            model.to_dict(),
            state.to_dict(),
            record,
            model.fluid,
            {"openings": model.default_openings},
        )


def test_real_phase_fallback_retains_wall_heat_and_passes_old_auditor():
    from check_pr_collocation import ROOT
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    config = json.loads((ROOT / "configs/surface_pr_collocation_phase_cycle.json").read_text())
    config["model"]["nodes"][0]["wall"] = {
        "heat_capacity_j_k": 1000.0,
        "fluid_ua_w_k": 1.0,
        "ambient_ua_w_k": 0.5,
        "ambient_temperature_k": 200.0,
    }
    config["initial"]["nodes"]["cell"]["wall_temperature_k"] = 210.0
    result = json.loads(json.dumps(run(config, integrator="gauss4")))
    assert audit(config, result)["accepted"]
    assert sum(i["fallback_intervals"] for i in result["intervals"]) >= 1
    assert len(result["intervals"][0]["state"]["nodes"]["cell"]["phase_roots"]) == 2
    assert len(result["intervals"][1]["state"]["nodes"]["cell"]["phase_roots"]) == 1
    assert max(map(abs, result["conservation_residuals"])) < 1e-5


def test_shared_auditor_handles_full_sparse_wall_network_and_tampered_heat():
    from check_pr_collocation import ROOT
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    config = json.loads((ROOT / "configs/surface_pr_collocation_wall_network.json").read_text())
    result = json.loads(json.dumps(run(config, integrator="gauss4")))
    assert audit(config, result)["checked_substeps"] == 14
    assert all(i["fallback_intervals"] == 0 for i in result["intervals"])
    bad = copy.deepcopy(result)
    bad["intervals"][0]["steps"][0]["entropy_production_j_k"] += 1.0
    with pytest.raises(ValueError):
        audit(config, bad)


def test_wall_temperature_error_controls_step_and_precision():
    model, state = wall_fixture()
    nodes = [
        replace(
            n,
            wall=replace(
                n.wall, fluid_ua_w_k=0.0, ambient_ua_w_k=400.0, max_temperature_change_k=100.0
            ),
        )
        for n in model.nodes
    ]
    base = PREquilibriumTransportNetwork(
        model.fluid,
        nodes,
        initial_step_s=0.2,
        maximum_step_s=0.25,
        relative_tolerance=1e-3,
        temperature_tolerance_k=0.1,
    )
    loose_state = replace(state, model_id=base.model_id, next_step_s=0.2)
    strict = PREquilibriumTransportNetwork.from_dict(
        {**base.to_dict(), "temperature_tolerance_k": 1e-6}
    )
    strict_state = replace(loose_state, model_id=strict.model_id)
    weak = PRCollocation(base, phase_fallback=False).advance(loose_state, 0.2)
    strong = PRCollocation(strict, phase_fallback=False).advance(strict_state, 0.2)
    exact = np.array(
        [
            n.wall.ambient_temperature_k
            + (s.wall_temperature_k - n.wall.ambient_temperature_k)
            * np.exp(-n.wall.ambient_ua_w_k / n.wall.heat_capacity_j_k * 0.2)
            for n, (_, s) in zip(base.nodes, state.nodes)
        ]
    )

    def error(result):
        return float(
            abs(
                np.array(
                    [
                        s.wall_temperature_k
                        for _, s in PREquilibriumTransportState.from_dict(result["state"]).nodes
                    ]
                )
                - exact
            ).max()
        )

    assert len(strong["steps"]) > len(weak["steps"])
    assert error(strong) < error(weak)
    assert error(strong) < 1e-6
