"""Conservative PR stages, real phase inventories and independent order audit."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from check_pr_collocation import check, thermal_fixture
from test_surface_pr_equilibrium_transport import pipe

from surface import (
    PRCollocation,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from surface.liquid import HydraulicConvergenceError, PropertyModelError


@pytest.mark.parametrize("temperature", [320.0, 196.2])
def test_moving_periodic_single_and_two_phase_stocks(temperature):
    model, state = pipe(temperature, 2.0, True)
    old = state.to_dict()
    report = PRCollocation(model, phase_fallback=False).advance(state, 1e-5)
    final = PREquilibriumTransportState.from_dict(report["state"])
    for (_, a), (_, b) in zip(state.nodes, final.nodes):
        assert a.component_moles == pytest.approx(b.component_moles, abs=1e-12)
        assert a.total_energy_j == pytest.approx(b.total_energy_j, abs=1e-8)
        assert a.momentum_kg_m_s == pytest.approx(b.momentum_kg_m_s, abs=1e-12)
        assert a.phase_roots == b.phase_roots
    assert report["fallback_intervals"] == 0
    assert report["state"]["time_s"] == 1e-5
    assert max(map(abs, report["conservation_residuals"])) < 1e-8
    assert state.to_dict() == old


def test_independent_isochoric_pr_ode_and_fourth_order():
    report = check()
    assert min(report["observed_refinement_ratios"]) > 14
    assert report["adaptive_temperature_error_k"] < 1e-5


def test_all_stage_ad_jacobians_against_finite_differences(monkeypatch):
    model, state = thermal_fixture()
    solver = PRCollocation(model, phase_fallback=False)
    original = solver._solve
    checked = []

    def audited(fn, initial, *args):
        direction = torch.linspace(-0.01, 0.01, len(initial), dtype=initial.dtype)
        jac = torch.func.jacfwd(fn)(initial)
        eps = 1e-5
        fd = (fn(initial + eps * direction) - fn(initial - eps * direction)) / (2 * eps)
        assert float((jac @ direction - fd).abs().max()) < 2e-7
        checked.append(len(initial))
        return original(fn, initial, *args)

    monkeypatch.setattr(solver, "_solve", audited)
    solver._step(state, 0.02, model.default_openings, [0.0, 0.0], [320.0, 360.0], {})
    assert checked[0] == 2 * checked[1]


def test_independent_heat_link_inventory_updates_and_stages():
    model, state = thermal_fixture()
    solver = PRCollocation(model, phase_fallback=False)
    final, record = solver._step(state, 0.2, model.default_openings, [0.0, 0.0], [320.0, 360.0], {})
    stages = record["stage_states"]
    rates = [5 * (s["a"]["temperature_k"] - s["b"]["temperature_k"]) for s in stages]
    exchanged = 0.2 * sum(rates) / 2
    assert record["heat_links"]["contact"] == pytest.approx(exchanged, abs=1e-12)
    for i, sign in ((0, -1), (1, 1)):
        previous = state.nodes[i][1]
        current = final.nodes[i][1]
        assert current.total_energy_j - previous.total_energy_j == pytest.approx(
            sign * exchanged, abs=1e-10
        )
        for k in range(2):
            expected = previous.total_energy_j + sign * 0.2 * sum(
                solver.a[k][j] * rates[j] for j in range(2)
            )
            assert stages[k][model.names[i]]["total_energy_j"] == pytest.approx(expected, abs=1e-7)


def test_nonuniform_two_phase_transport_shared_face_ledger():
    model, state = pipe(196.2, 0.0, True)
    state = model.initialize(
        {
            str(i): {
                "pressure_pa": 1e6 + (-1) ** i * 20,
                "temperature_k": 196.2,
                "mole_fractions": [0.5, 0.5],
            }
            for i in range(2)
        }
    )
    final, record = PRCollocation(model, phase_fallback=False)._step(
        state, 1e-6, model.default_openings, [0.0, 0.0], [196.2] * 2, {}
    )
    assert all(len(s.phase_roots) == 2 for _, s in final.nodes)
    expected = np.zeros(4)
    for link, row in zip(model.links, record["links"]):
        flux = np.array(row["integrated_flux"])
        expected += flux * (-1 if link.source == "0" else 1)
    before, after = state.nodes[0][1], final.nodes[0][1]
    np.testing.assert_allclose(
        np.array(after.component_moles) - before.component_moles, expected[:-2], atol=1e-12
    )
    assert after.total_energy_j - before.total_energy_j == pytest.approx(expected[-1], abs=1e-7)
    assert max(map(abs, record["conservation_residuals"])) < 1e-7


def test_sparse_stage_solve_matches_dense():
    dense, state = thermal_fixture()
    sparse = PREquilibriumTransportNetwork.from_dict({**dense.to_dict(), "dense_limit": 4})
    other = sparse.initialize(
        {
            name: {
                "pressure_pa": s.pressure_pa,
                "temperature_k": s.temperature_k,
                "mole_fractions": [1.0, 0.0],
            }
            for name, s in state.nodes
        }
    )
    a, _ = PRCollocation(dense, phase_fallback=False)._step(
        state, 0.05, dense.default_openings, [0.0, 0.0], [320.0, 360.0], {}
    )
    b, _ = PRCollocation(sparse, phase_fallback=False)._step(
        other, 0.05, sparse.default_openings, [0.0, 0.0], [320.0, 360.0], {}
    )
    np.testing.assert_allclose(
        [s.temperature_k for _, s in a.nodes],
        [s.temperature_k for _, s in b.nodes],
        rtol=0,
        atol=1e-9,
    )


def test_failed_attempt_is_atomic_and_bounded(monkeypatch):
    model, state = thermal_fixture()
    model = PREquilibriumTransportNetwork.from_dict({**model.to_dict(), "maximum_attempts": 2})
    state = replace(state, model_id=model.model_id)
    before = state.to_dict()
    solver = PRCollocation(model, phase_fallback=False)
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        raise HydraulicConvergenceError("synthetic Newton failure")

    monkeypatch.setattr(solver, "_step", failed)
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        solver.advance(state, 0.5)
    assert len(calls) == 2
    assert state.to_dict() == before


def test_fallback_is_explicit_and_cannot_be_hidden(monkeypatch):
    model, state = thermal_fixture()
    solver = PRCollocation(model)

    def failed(*args, **kwargs):
        raise HydraulicConvergenceError("forced Gauss failure")

    monkeypatch.setattr(solver, "_step", failed)
    report = solver.advance(state, 0.001)
    assert report["fallback_intervals"] > 0
    assert all(
        s["method"] == "backward_euler_fallback" and "forced Gauss failure" in s["fallback_reason"]
        for s in report["steps"]
    )
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_real_condensation_and_evaporation_use_branch_fallback():
    from test_surface_compositional_separator import binary

    from surface.peng_robinson import R

    f = binary()
    c = f.components[1]
    sat = c.saturation_at_temperature(200)
    p = sat["pressure_pa"]
    l, v = sat["liquid"], sat["vapor"]
    mw = c.molar_mass_kg_mol
    vm = (0.4 * l["specific_volume_m3_kg"] + 0.6 * v["specific_volume_m3_kg"]) * mw
    target = (
        10
        * mw
        * (
            0.4 * (l["enthalpy_j_kg"] - p * l["specific_volume_m3_kg"])
            + 0.6 * (v["enthalpy_j_kg"] - p * v["specific_volume_m3_kg"])
        )
    )
    attraction, _ = c._a(220.0)
    p0 = R * 220 / (vm - c.b) - attraction / (vm * vm + 2 * c.b * vm - c.b * c.b)
    model = PREquilibriumTransportNetwork(
        f, [PREquilibriumNode("cell", 10 * vm)], initial_step_s=1.0, maximum_step_s=1.0
    )
    state = model.initialize(
        {"cell": {"pressure_pa": p0, "temperature_k": 220.0, "mole_fractions": [0.0, 1.0]}}
    )
    energy = state.nodes[0][1].total_energy_j
    solver = PRCollocation(model)
    cooled = solver.advance(
        state, 1.0, heat_w={"cell": target - energy}, utility_temperature_k={"cell": 170.0}
    )
    mid = PREquilibriumTransportState.from_dict(cooled["state"])
    assert len(mid.nodes[0][1].phase_roots) == 2
    assert mid.nodes[0][1].temperature_k == pytest.approx(200.0, abs=1e-5)
    assert mid.nodes[0][1].pressure_pa == pytest.approx(p, rel=1e-7)
    assert mid.nodes[0][1].total_energy_j == pytest.approx(target, abs=1e-6)
    assert cooled["fallback_intervals"] > 0
    heated = solver.advance(
        mid, 1.0, heat_w={"cell": energy - target}, utility_temperature_k={"cell": 300.0}
    )
    final = PREquilibriumTransportState.from_dict(heated["state"])
    assert len(final.nodes[0][1].phase_roots) == 1
    assert final.nodes[0][1].temperature_k == pytest.approx(220.0, abs=1e-5)
    assert max(map(abs, heated["conservation_residuals"])) < 1e-6


def test_closed_heat_contact_and_report_split_restart():
    model, state = thermal_fixture()
    solver = PRCollocation(model, phase_fallback=False)
    full = solver.advance(state, 0.1, openings={"contact": False})
    part = solver.advance(state, 0.07, openings={"contact": False})
    restart = PREquilibriumTransportState.from_dict(json.loads(json.dumps(part["state"])))
    final = solver.advance(restart, 0.03, openings={"contact": False})
    assert final["state"]["time_s"] == full["state"]["time_s"]
    for (_, a), (_, b) in zip(
        state.nodes, PREquilibriumTransportState.from_dict(final["state"]).nodes
    ):
        assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-9)
    assert final["state"]["next_step_s"] >= 0.05


@pytest.mark.parametrize(
    "command",
    [
        {"heat_w": {"unknown": 1}},
        {"openings": {"contact": 1}},
        {"heat_w": {"a": 100}},
        {"loss_coefficients": {"a": 1}},
    ],
)
def test_invalid_commands_rejected(command):
    model, state = thermal_fixture()
    with pytest.raises(ValueError):
        PRCollocation(model).advance(state, 0.01, **command)


def test_unsupported_physics_rejected_and_model_mutation_detected():
    model, state = thermal_fixture()
    from surface import PRCollocation

    bad = PREquilibriumTransportNetwork(
        model.fluid, [PREquilibriumNode("x", 0.001)], reservoir_ports="stagnation_single_phase"
    )
    with pytest.raises(ValueError, match="Stagnation"):
        PRCollocation(bad)
    model.relative_tolerance *= 2
    with pytest.raises(PropertyModelError):
        PRCollocation(model).advance(state, 0.1)


def test_cli_integrator_selection_and_grouped_restart():
    from run_pr_equilibrium_transport import run

    model, state = thermal_fixture()
    config = {
        "model": model.to_dict(),
        "initial": {"state": state.to_dict()},
        "schedule": [{"duration_s": 0.01}],
    }
    report = run(config, integrator="gauss4")
    assert report["integrator"] == "gauss4"
    assert report["total_work_j"] == 0
    assert report["intervals"][0]["fallback_intervals"] == 0
    assert max(map(abs, report["conservation_residuals"])) < 1e-6
    with pytest.raises(ValueError, match="Unknown PR integrator"):
        run(config, integrator="other")


@pytest.mark.parametrize("two_phase", [False, True])
def test_independent_serialized_stage_audit_and_corruption_rejection(monkeypatch, two_phase):
    import copy

    from audit_pr_collocation import audit_gauss_step

    if two_phase:
        model, state = pipe(196.2, 0.0, True)
        state = model.initialize(
            {
                str(i): {
                    "pressure_pa": 1e6 + (-1) ** i * 20,
                    "temperature_k": 196.2,
                    "mole_fractions": [0.5, 0.5],
                }
                for i in range(2)
            }
        )
        heat = [0.0, 0.0]
        utility = [196.2] * 2
        dt = 1e-6
    else:
        model, state = thermal_fixture()
        heat = [0.0, 0.0]
        utility = [320.0, 360.0]
        dt = 0.1
    _, record = PRCollocation(model, phase_fallback=False)._step(
        state, dt, model.default_openings, heat, utility, {}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Auditor called production physical operator")

    monkeypatch.setattr(PREquilibriumTransportNetwork, "_step", forbidden)
    monkeypatch.setattr(PREquilibriumTransportNetwork, "_decode", forbidden)
    monkeypatch.setattr(PREquilibriumTransportNetwork, "_fluxes", forbidden)
    monkeypatch.setattr(PREquilibriumTransportNetwork, "inspect", forbidden)
    monkeypatch.setattr(type(model.fluid), "flash_uv", forbidden)
    monkeypatch.setattr(type(model.fluid), "flash_pt", forbidden)
    args = (model.to_dict(), state.to_dict())
    assert audit_gauss_step(*args, record, model.fluid, {"openings": model.default_openings})[
        "accepted"
    ]
    for key in ("stage", "endpoint", "flux"):
        bad = copy.deepcopy(record)
        if key == "stage":
            bad["stage_states"][0][model.names[0]]["temperature_k"] += 1
        elif key == "endpoint":
            bad["state"]["nodes"][model.names[0]]["total_energy_j"] += 1
        elif two_phase:
            bad["links"][0]["integrated_flux"][-1] += 1
        else:
            bad["heat_links"]["contact"] += 1
        with pytest.raises(ValueError):
            audit_gauss_step(*args, bad, model.fluid, {"openings": model.default_openings})


def test_conservative_error_estimator_responds_to_tolerance():
    from check_pr_collocation import IsochoricReference
    from scipy.integrate import solve_ivp

    model, state = thermal_fixture()
    loose = PREquilibriumTransportNetwork.from_dict(
        {
            **model.to_dict(),
            "relative_tolerance": 1e-3,
            "temperature_tolerance_k": 0.1,
            "initial_step_s": 0.25,
        }
    )
    loose_state = replace(state, model_id=loose.model_id, next_step_s=0.25)
    weak = PRCollocation(loose, phase_fallback=False).advance(loose_state, 0.5)
    strict = PRCollocation(model, phase_fallback=False).advance(state, 0.5)
    ref = IsochoricReference(model, state)
    expected = solve_ivp(
        ref.rhs, (0.0, 0.5), [320.0, 360.0], method="DOP853", rtol=1e-13, atol=1e-12
    ).y[:, -1]

    def error(report):
        return max(
            abs(
                np.array(
                    [
                        s.temperature_k
                        for _, s in PREquilibriumTransportState.from_dict(report["state"]).nodes
                    ]
                )
                - expected
            )
        )

    assert error(strict) < error(weak)
    assert len(strict["steps"]) > len(weak["steps"])


def test_phase_cycle_cli_reuses_existing_independent_auditor():
    from check_pr_collocation import ROOT
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    config = json.loads((ROOT / "configs/surface_pr_collocation_phase_cycle.json").read_text())
    report = run(config, integrator="gauss4")
    assert audit(config, json.loads(json.dumps(report)))["accepted"]
    assert sum(i["fallback_intervals"] for i in report["intervals"]) == 2
