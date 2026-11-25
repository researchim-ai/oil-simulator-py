"""Whole PR machine graph: continuous loads and conserved fluid/shaft energy."""

import math
from dataclasses import replace

import pytest
import torch
from test_surface_pr_machine_drive import setup

from surface.pr_collocation import PRCollocation
from surface.pr_equilibrium_transport import PREquilibriumTransportState
from surface.pr_machine_drive import isolated_motion


@pytest.mark.parametrize("kind", ["pump", "compressor"])
@pytest.mark.parametrize("dense_limit", [1, 128])
def test_running_whole_graph_energy_and_common_newton(kind, dense_limit):
    model, old, _ = setup(kind, dense_limit)
    solver = PRCollocation(model, scheme="radau3", error_control="embedded", phase_fallback=False)
    new, record = solver._step(
        old, 1e-5, model.default_openings, [0.0] * len(model.nodes), [350.0] * len(model.nodes), {}
    )
    assert record["nonlinear_solves"] == 1 + record["initialization_nonlinear_solves"]
    assert math.isfinite(record["embedded_error_ratio"])
    assert max(map(abs, record["conservation_residuals"])) < 1e-7
    name = model.drives[0].machine
    drive = record["drives"][name]
    assert drive["shaft_work_j"] > 0
    assert dict(new.rotors)[name] > dict(old.rotors)[name]
    fluid_change = sum(
        b.total_energy_j - a.total_energy_j for (_, a), (_, b) in zip(old.nodes, new.nodes)
    )
    assert fluid_change + drive["kinetic_energy_change_j"] == pytest.approx(
        drive["electrical_energy_j"] - drive["heat_to_environment_j"], abs=1e-7
    )
    assert fluid_change == pytest.approx(
        drive["shaft_work_j"] - drive["kinetic_quadrature_difference_j"], abs=1e-7
    )
    assert record["entropy_production_j_k"] >= -1e-7
    model.inspect(PREquilibriumTransportState.from_dict(new.to_dict()))


@pytest.mark.parametrize("kind", ["pump", "compressor"])
def test_continuous_stage_jacobian_and_declared_sparse_support(kind):
    model, state, _ = setup(kind)
    solver = PRCollocation(model, scheme="radau3", phase_fallback=False)
    initial, _, _, evaluate, rates, *_ = solver._problem(
        state, model.default_openings, [0.0] * len(model.nodes), {}
    )
    dt = 1e-4

    def residual(y):
        storage, physical = evaluate(y)
        return storage - dt * rates(physical)

    jac = torch.func.jacfwd(residual)(initial)
    direction = torch.linspace(-1e-3, 1e-3, len(initial), dtype=initial.dtype)
    eps = 1e-4
    fd = (residual(initial + eps * direction) - residual(initial - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=2e-5, atol=2e-7)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    mask[evaluate.pattern] = True
    assert not bool((jac[~mask].abs() > 1e-11).any())
    assert bool((jac[-1, :-1].abs() > 1e-12).any())
    assert bool((jac[:-1, -1].abs() > 1e-12).any())
    assert all(
        bool(torch.isfinite(v).all())
        for v in evaluate(initial)[1]["drives"].values()
        for v in v.values()
    )


def test_isolated_radau_third_order_against_exact_rotor_without_extra_nonlinear_solves():
    base, _, initial = setup()
    config = base.to_dict()
    config["drives"][0]["drive"]["viscous_drag_nm_s"] = 0.2
    model = type(base).from_dict(config)
    old = model.initialize(initial)
    commands = dict.fromkeys(model.default_openings, False)
    drive = model.drives[0]
    exact = isolated_motion(
        model.fluid.tensor(dict(old.rotors)[drive.machine]),
        model.fluid.tensor(drive.motor_torque_nm),
        0.1,
        drive.drive,
    )[0]
    errors = []
    for count in (1, 2, 4):
        solver = PRCollocation(model, scheme="radau3", phase_fallback=False)
        state = old
        for _ in range(count):
            state, record = solver._step(state, 0.1 / count, commands, [0.0] * 3, [350.0] * 3, {})
            assert record["nonlinear_solves"] == 1
            assert max(map(abs, record["conservation_residuals"])) < 1e-7
        errors.append(abs(dict(state.rotors)[drive.machine] - float(exact)))
    assert errors[0] / errors[1] > 7
    assert errors[1] / errors[2] > 7


def test_public_embedded_advance_preserves_rotor_restart_and_counts_work():
    model, old, _ = setup()
    old = replace(old, next_step_s=1e-5)
    solver = PRCollocation(model, scheme="radau3", error_control="embedded", phase_fallback=False)
    report = solver.advance(old, 2e-5)
    assert report["fallback_intervals"] == 0
    assert report["total_work_j"] > 0
    assert report["collocation_nonlinear_solves"] == sum(
        s["nonlinear_solves"] for s in report["steps"]
    )
    new = PREquilibriumTransportState.from_dict(report["state"])
    assert new.rotors
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


@pytest.mark.parametrize("backpressure", [1e6, 1.1e6])
@pytest.mark.parametrize("dense_limit", [1, 128])
def test_true_flow_pump_connection_open_and_closed(backpressure, dense_limit):
    from test_surface_pr_pump_connection import setup as connection_setup

    model, old, _ = connection_setup(
        backpressure=backpressure, driven=True, dense_limit=dense_limit
    )
    solver = PRCollocation(model, scheme="radau3", error_control="embedded", phase_fallback=False)
    new, record = solver._step(old, 1e-5, model.default_openings, [0.0] * 2, [350.0] * 2, {})
    assert math.isfinite(record["embedded_error_ratio"])
    assert max(map(abs, record["conservation_residuals"])) < 1e-7
    shaft_work = record["drives"]["booster"]["shaft_work_j"]
    assert shaft_work > 0 if backpressure == 1e6 else shaft_work == 0
    if backpressure > 1e6:
        for (_, a), (_, b) in zip(old.nodes, new.nodes):
            assert b.component_moles == pytest.approx(a.component_moles, abs=1e-10)


def test_finite_coolant_electrical_losses_and_independent_stage_torque_balance():
    model, _, initial = setup()
    config = model.to_dict()
    config["drives"][0]["heat_sink_node"] = "feed"
    model = type(model).from_dict(config)
    old = model.initialize(initial)
    solver = PRCollocation(model, scheme="radau3", error_control="embedded", phase_fallback=False)
    # Isolate all fluid connections; the motor still accelerates the shaft.
    record = solver.advance(
        replace(old, next_step_s=1e-3),
        1e-3,
        openings=dict.fromkeys(model.default_openings, False),
        drive_torque_nm={"booster": 4.0},
    )
    drive = model.drives[0]
    before_speed = dict(old.rotors)["booster"]
    for step in record["steps"]:
        speeds = [s["booster"] for s in step["stage_rotors"]]
        rates = [
            (4.0 - drive.drive.viscous_drag_nm_s * speed) / drive.drive.inertia_kg_m2
            for speed in speeds
        ]
        for i, speed in enumerate(speeds):
            assert speed - before_speed == pytest.approx(
                step["dt_s"] * sum(solver.a[i][j] * rates[j] for j in range(2)),
                abs=1e-9,
            )
        d = step["drives"]["booster"]
        assert d["heat_to_node_j"] > 0
        assert d["heat_to_environment_j"] == 0
        assert step["ambient_heat_j"] == 0
        before_speed = speeds[-1]
    assert max(map(abs, record["conservation_residuals"])) < 1e-7


def test_cli_radau_machine_commands_and_fallback_remain_explicit():
    from run_pr_equilibrium_transport import run

    model, _, initial = setup()
    config = {
        "model": model.to_dict(),
        "initial": {"nodes": initial},
        "schedule": [{"duration_s": 1e-5, "drive_torque_nm": {"booster": 4.0}}],
    }
    report = run(config, integrator="radau3-embedded")
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    assert report["intervals"][0]["fallback_intervals"] == 0
    assert report["total_work_j"] > 0


@pytest.mark.parametrize("kind", ["pump", "compressor", "connection", "restriction"])
def test_independent_scalar_eos_cards_and_stage_energy_audit(kind):
    import copy
    import json

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    if kind == "connection":
        from test_surface_pr_pump_connection import setup as factory

        model, _, initial = factory(driven=True)
    elif kind == "restriction":
        from test_surface_pr_hem_restriction import setup as factory

        model, _, initial = factory(backpressure=8e5)
    else:
        model, _, initial = setup(kind)
    request = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 1e-5}],
            }
        )
    )
    report = json.loads(json.dumps(run(request, integrator="radau3-embedded")))
    assert audit(request, report)["accepted"]
    broken = copy.deepcopy(report)
    stage = broken["intervals"][0]["steps"][0]
    family = {
        "pump": "pumps",
        "connection": "pump_connections",
        "compressor": "compressors",
        "restriction": "restrictions",
    }[kind]
    row = next(iter(stage["stage_equipment"][0][family].values()))
    field = "molar_rate_mol_s" if kind == "restriction" else "shaft_power_w"
    row[field] += 1.0
    with pytest.raises(ValueError):
        audit(request, broken)


def test_embedded_initial_map_projection_is_reused_only_for_same_physical_endpoint():
    model, old, _ = setup("compressor")
    solver = PRCollocation(model, scheme="radau3", error_control="embedded", phase_fallback=False)
    result = solver.advance(replace(old, next_step_s=1e-5), 3e-5)
    counts = [r["initialization_nonlinear_solves"] for r in result["steps"]]
    assert counts[0] == 1
    assert sum(counts[1:]) == 0
    # Rewinding to the original checkpoint must discard the endpoint cache.
    _, repeated = solver._step(old, 1e-5, model.default_openings, [0.0] * 2, [350.0] * 2, {})
    assert repeated["initialization_nonlinear_solves"] == 1


def test_wrong_closed_pump_branch_is_corrected_before_stage_acceptance():
    from test_surface_pr_pump_connection import setup as factory

    model, old, _ = factory(driven=True)
    solver = PRCollocation(model, scheme="radau3", error_control="embedded", phase_fallback=False)
    _new, report = solver._step(
        old,
        1e-5,
        model.default_openings,
        [0.0] * 2,
        [350.0] * 2,
        {},
        _pump_valve_states={"booster": True},
    )
    assert report["pump_check_valve_retries"] >= 1
    assert report["drives"]["booster"]["shaft_work_j"] > 0
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


@pytest.mark.parametrize("temperature,reverse", [(320.0, False), (320.0, True), (196.2, False)])
def test_critical_and_reversed_restriction_in_monolithic_radau(temperature, reverse):
    from test_surface_pr_hem_restriction import setup as factory

    model, old, _ = factory(
        backpressure=2e5, temperature=temperature, reverse=reverse, dense_limit=1
    )
    solver = PRCollocation(model, scheme="radau3", error_control="embedded", phase_fallback=False)
    result = solver.advance(
        replace(old, next_step_s=1e-6), 1e-6, restriction_openings={"valve": 0.5}
    )
    assert result["fallback_intervals"] == 0
    assert max(map(abs, result["conservation_residuals"])) < 1e-7
