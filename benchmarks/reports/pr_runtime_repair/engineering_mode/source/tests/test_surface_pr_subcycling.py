"""Conservative split boundary exchange and unsupported-partition guards."""

import copy
import json
from pathlib import Path

import pytest
from check_pr_controlled_surface import audit
from run_pr_controlled_surface import run

from surface.pr_subcycled_surface import PRSubcycledSurface
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState


@pytest.fixture(scope="module")
def subcycle_config():
    path = Path(__file__).resolve().parents[1] / "configs/surface_pr_integrated_facility.json"
    config = json.loads(path.read_text())
    original = PRControlledSurface.from_dict(config["model"])
    model = PRSubcycledSurface(original, ["gas_recycle_0"])
    state = model.adopt(PRControlledSurfaceState.from_dict(config["initial"]["state"]))
    config.update(
        model=model.to_dict(), initial={"state": state.to_dict()}, schedule=[{"duration_s": 1e-8}]
    )
    return json.loads(json.dumps(config))


@pytest.fixture(scope="module")
def result(subcycle_config):
    return json.loads(json.dumps(run(subcycle_config)))


def test_split_retains_all_stocks_and_shared_eos(subcycle_config):
    m = PRControlledSurface.from_dict(subcycle_config["model"])
    assert isinstance(m, PRSubcycledSurface)
    assert m.fast.fluid is m.plant.transport.fluid is m.slow.plant.transport.fluid
    s = PRControlledSurfaceState.from_dict(subcycle_config["initial"]["state"])
    m.inspect(s)
    assert set(m.fast.names) | set(m.slow.plant.transport.names) == set(m.plant.transport.names)
    assert set(m.fast.names) & set(m.slow.plant.transport.names) == set(m.boundaries)
    restored = PRControlledSurface.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(s)
    assert restored.to_dict() == m.to_dict()


def test_split_audit_exact_interval_and_restart(subcycle_config, result):
    checked = audit(subcycle_config, result)
    assert checked["accepted"] and checked["checked_split_steps"] == 2
    assert checked["checked_local_steps"] >= 4
    state = result["intervals"][-1]["state"]
    assert state["plant"]["transport"]["time_s"] == 1e-8
    m = PRControlledSurface.from_dict(subcycle_config["model"])
    m.inspect(PRControlledSurfaceState.from_dict(state))
    assert max(abs(x) for x in checked["conservation_residuals"][:-1]) < 1e-8
    assert abs(checked["conservation_residuals"][-1]) < 1e-5


@pytest.mark.parametrize("corruption", ["transfer", "pipe", "stage", "initial", "energy", "hint"])
def test_independent_audit_rejects_corrupt_split(subcycle_config, result, corruption):
    damaged = copy.deepcopy(result)
    step = damaged["intervals"][0]["steps"][0]
    stages = step["local_subcycling"]["stages"]
    if corruption == "transfer":
        transfer = next(iter(stages[0]["boundary_transfers"].values()))
        transfer["component_moles"][0] += 1
    elif corruption == "pipe":
        step["state"]["plant"]["transport"]["nodes"]["gas_recycle_0"]["total_energy_j"] += 1
    elif corruption == "stage":
        stages.pop()
    elif corruption == "initial":
        stages[1]["initial"]["plant"]["transport"]["nodes"]["gas_export"]["total_energy_j"] += 1
    elif corruption == "hint":
        damaged["intervals"][0]["state"]["next_step_s"] = -1.0
    else:
        step["total_heat_j"] += 1
    with pytest.raises((ValueError, AssertionError)):
        audit(subcycle_config, damaged)


@pytest.mark.parametrize(
    "selection", [[], ["missing"], ["gas_export"], ["gas_recycle_0", "gas_recycle_0"]]
)
def test_invalid_partition_rejected(subcycle_config, selection):
    m = PRControlledSurface.from_dict(subcycle_config["model"])
    with pytest.raises(ValueError):
        PRSubcycledSurface(m.reference_surface, selection)


def test_commands_partition_and_rejection_are_atomic(subcycle_config):
    m = PRControlledSurface.from_dict(subcycle_config["model"])
    s = PRControlledSurfaceState.from_dict(subcycle_config["initial"]["state"])
    before = s.to_dict()
    fast, slow = m._commands(
        {"heat_w": {"gas_recycle_0": 1.0}, "utility_temperature_k": {"gas_recycle_0": 400.0}}
    )
    assert fast["heat_w"] == {"gas_recycle_0": 1.0}
    assert slow["heat_w"] == {}
    with pytest.raises(ValueError):
        m.advance(s, 1e-8, openings={"absent": False})
    assert s.to_dict() == before


def test_failed_local_stage_preserves_input(subcycle_config, monkeypatch):
    model = PRControlledSurface.from_dict(subcycle_config["model"])
    state = PRControlledSurfaceState.from_dict(subcycle_config["initial"]["state"])
    before = state.to_dict()

    def fail(*args, **kwargs):
        raise RuntimeError("local solve failed")

    monkeypatch.setattr(model.fast, "advance", fail)
    with pytest.raises(RuntimeError, match="local solve failed"):
        model._step(state, 1e-8, {}, {}, {})
    assert state.to_dict() == before


def test_preparation_preserves_physics_and_schedule(subcycle_config):
    from enable_pr_subcycling import enable

    model = PRControlledSurface.from_dict(subcycle_config["model"])
    state = copy.deepcopy(subcycle_config["initial"]["state"])
    state["model_id"] = model.reference_surface.model_id
    original = {
        "model": model.reference_surface.to_dict(),
        "initial": {"state": state},
        "schedule": subcycle_config["schedule"],
    }
    prepared = enable(original, ["gas_recycle_0"])
    assert prepared["model"]["plant"] == original["model"]["plant"]
    assert json.loads(json.dumps(prepared["initial"]["state"]["plant"])) == state["plant"]
    assert prepared["schedule"] == original["schedule"]
    assert original["initial"]["state"]["model_id"] == model.reference_surface.model_id


def small_network(reverse=False, darcy=False):
    import math

    from check_pr_collocation import thermal_fixture

    from surface.actuator import ConnectionActuator
    from surface.pr_equilibrium_transport import (
        PREquilibriumLink,
        PREquilibriumNode,
        PREquilibriumTransportNetwork,
    )
    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.pr_surface_control import PRControlValve

    base, _ = thermal_fixture()
    nodes = [PREquilibriumNode(n, 0.1) for n in ("a", "b", "c")]
    nodes += [
        PREquilibriumNode(
            n,
            math.pi * d * d / 4 * 0.2,
            "pipe_cell",
            length_m=0.2,
            diameter_m=d,
            dynamic_viscosity_pa_s=1e-4,
        )
        for n, d in (("fast", 0.001), ("slow", 0.01))
    ]
    links = [
        PREquilibriumLink("af", "a", "fast", 0.001),
        PREquilibriumLink("fb", "fast", "b", 0.001),
        PREquilibriumLink("bs", "b", "slow", 0.01),
        PREquilibriumLink("sc", "slow", "c", 0.01),
    ]
    transport = PREquilibriumTransportNetwork(
        base.fluid,
        nodes,
        links,
        reservoir_ports="stagnation_single_phase",
        initial_step_s=1e-5,
        maximum_step_s=1e-3,
    )
    ts = transport.initialize(
        {
            n.name: {
                "pressure_pa": 1e6
                + (-1 if reverse else 1)
                * (100 if n.name == "a" else 50 if n.name == "fast" else 0),
                "temperature_k": 350.0,
                "mole_fractions": [1.0, 0.0],
            }
            for n in nodes
        }
    )
    from surface.pr_darcy_connection import PRDarcyConnection

    plant = PRGravityTransportNetwork(
        transport,
        {},
        [],
        darcy_connections=(
            [PRDarcyConnection("bypass", "b", "c", 1.0, 0.001, 1e-4)] if darcy else []
        ),
    )
    original = PRControlledSurface(
        plant,
        [PRControlValve("v", "sc", 1.0, ConnectionActuator(1.0, 1.0, 0.0))],
        initial_step_s=1e-5,
        maximum_step_s=1e-3,
    )
    state = original.initialize(plant.initialize(ts, {}), {"v": 1.0})
    model = PRSubcycledSurface(original, ["fast"])
    return original, state, model, model.adopt(state)


@pytest.mark.parametrize("reverse", [False, True])
def test_small_forward_reverse_matches_monolithic(reverse):
    original, initial, model, state = small_network(reverse)
    dt = 1e-5
    split = model.advance(state, dt, targets={"v": 1.0})
    reference = original.advance(initial, dt, targets={"v": 1.0})
    a = PRControlledSurfaceState.from_dict(split["state"])
    b = PRControlledSurfaceState.from_dict(reference["state"])
    # Compare physical coordinates in the ORIGINAL solver's error norm.
    assert original._error(a, b) < 1
    change = (
        dict(a.plant.transport.nodes)["a"].component_moles[0]
        - dict(initial.plant.transport.nodes)["a"].component_moles[0]
    )
    assert change > 0 if reverse else change < 0


def test_isolated_fast_region_has_no_boundary_exchange():
    _, _, model, state = small_network()
    _, report = model._fast_stage(state, 0.0, 1e-5, {"openings": {"af": False, "fb": False}})
    for exchange in report["boundary_transfers"].values():
        assert exchange["component_moles"] == [0.0, 0.0]
        assert exchange["total_energy_j"] == 0.0


def test_small_restart_advances_and_fast_controls_are_rejected():
    original, _, model, state = small_network()
    with pytest.raises(ValueError, match="Actuated"):
        PRSubcycledSurface(original, ["slow"])
    first = model.advance(state, 1e-5, targets={"v": 1.0})
    restored = PRControlledSurface.from_dict(json.loads(json.dumps(model.to_dict())))
    resumed = restored.advance(
        PRControlledSurfaceState.from_dict(first["state"]), 1e-5, targets={"v": 1.0}
    )
    terminal = PRControlledSurfaceState.from_dict(resumed["state"])
    restored.inspect(terminal)
    assert terminal.time_s == 2e-5
    assert abs(resumed["conservation_residuals"][-1]) < 1e-5


def test_fast_heating_is_applied_once_with_closed_boundaries():
    _, _, model, state = small_network()
    result, report = model._fast_stage(
        state,
        0.0,
        1e-5,
        {
            "openings": {"af": False, "fb": False},
            "heat_w": {"fast": 1.0},
            "utility_temperature_k": {"fast": 400.0},
        },
    )
    before = dict(state.plant.transport.nodes)["fast"].total_energy_j
    after = dict(result.plant.transport.nodes)["fast"].total_energy_j
    assert report["report"]["total_heat_j"] == pytest.approx(1e-5, rel=1e-10)
    assert after - before == pytest.approx(1e-5, abs=1e-9)


@pytest.mark.parametrize("opened", [False, True])
def test_slow_darcy_error_control_receives_actual_isolation(opened):
    _, _, model, state = small_network(darcy=True)
    report = model.advance(state, 1e-5, targets={"v": 1.0}, openings={"bypass": opened})
    assert report["state"]["plant"]["transport"]["time_s"] == 1e-5
    assert model._error_openings(report["steps"][-1])["bypass"] is opened


def test_audit_accepts_outer_hint_changes_between_macrosteps():
    _, _, model, state = small_network()
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [{"duration_s": 4e-5, "targets": {"v": 1.0}}],
            }
        )
    )
    result = json.loads(json.dumps(run(config)))
    assert result["intervals"][0]["attempts"] > 1
    assert audit(config, result)["accepted"]


def test_explicit_local_boundary_model_is_serialized_and_audited():
    original, initial, unchanged, _ = small_network()
    changed = PRSubcycledSurface(original, ["fast"], fast_reservoir_ports="riemann")
    assert changed.fast.reservoir_ports == "riemann"
    assert changed.slow.plant.transport.reservoir_ports == original.plant.transport.reservoir_ports
    assert changed.model_id != unchanged.model_id
    assert "fast_reservoir_ports" not in unchanged.to_dict()
    state = changed.adopt(initial)
    config = json.loads(
        json.dumps(
            {
                "model": changed.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [{"duration_s": 1e-5, "targets": {"v": 1.0}}],
            }
        )
    )
    restored = PRControlledSurface.from_dict(config["model"])
    assert restored.fast.reservoir_ports == "riemann"
    assert audit(config, json.loads(json.dumps(run(config))))["accepted"]


def test_invalid_local_boundary_model_is_rejected():
    original, _, _, _ = small_network()
    with pytest.raises(ValueError, match="reservoir-port"):
        PRSubcycledSurface(original, ["fast"], fast_reservoir_ports="automatic")
