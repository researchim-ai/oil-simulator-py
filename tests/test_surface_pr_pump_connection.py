"""True-flow pump connection: shutoff, PS/PH work, rotor and sparse graph."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from test_surface_pr_pump_unit import setup as legacy_setup

from surface.pr_equilibrium_transport import (
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from surface.pr_machine_drive import PRMachineDrive
from surface.pr_pump_connection import PRPumpConnection
from surface.rotating_drive import RotatingDrive


@pytest.fixture(autouse=True)
def script_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))


def setup(backpressure=1e6, wave=2000.0, dense_limit=128, driven=False):
    base, _, initial = legacy_setup()
    data = base.to_dict()
    data["nodes"] = [n for n in data["nodes"] if n["name"] != "body"]
    data["links"] = []
    data.pop("pumps")
    data["pump_connections"] = [
        asdict(
            PRPumpConnection(
                replace(base.pumps[0].card, check_valve=True), outlet_loss_pa_s2_m6=1e13
            )
        )
    ]
    data["wave_speed_bound_m_s"] = wave
    data["dense_limit"] = dense_limit
    if driven:
        data["drives"] = [
            asdict(
                PRMachineDrive(
                    "booster",
                    RotatingDrive(0.1, 300.0, 30.0, 30.0, viscous_drag_nm_s=0.001),
                    100.0,
                    2.0,
                )
            )
        ]
    model = PREquilibriumTransportNetwork.from_dict(data)
    initial.pop("body")
    initial["return"]["pressure_pa"] = backpressure
    return model, model.initialize(initial), initial


@pytest.mark.parametrize("dense_limit", [1, 128])
@pytest.mark.parametrize("driven", [False, True])
def test_true_flow_work_and_restart(dense_limit, driven):
    model, old, _ = setup(dense_limit=dense_limit, driven=driven)
    new, report = model._step(old, 1e-5, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    row = report["pump_connections"]["booster"]
    assert not row["valve_closed"]
    assert row["flow_m3_s"] > 0 and row["shaft_power_w"] > 0
    i = model.names.index("feed")
    rho = float(model._state_data(i, dict(new.nodes)["feed"])["rho"])
    delta = dict(old.nodes)["feed"].component_moles[1] - dict(new.nodes)["feed"].component_moles[1]
    mw = float(model.fluid.molar_masses[1])
    assert delta * mw == pytest.approx(1e-5 * row["flow_m3_s"] * rho, rel=1e-6)
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    restored = PREquilibriumTransportState.from_dict(json.loads(json.dumps(new.to_dict())))
    type(model).from_dict(json.loads(json.dumps(model.to_dict()))).inspect(restored)


@pytest.mark.parametrize("wave", [2000.0, 4000.0])
@pytest.mark.parametrize("driven", [False, True])
def test_closed_check_valve_has_exactly_zero_mass_and_power(wave, driven):
    model, old, _ = setup(backpressure=1.1e6, wave=wave, driven=driven)
    new, report = model._step(old, 1e-4, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    row = report["pump_connections"]["booster"]
    assert row["valve_closed"]
    assert row["flow_m3_s"] == row["shaft_power_w"] == 0
    for name, value in old.nodes:
        assert dict(new.nodes)[name].component_moles == pytest.approx(
            value.component_moles, rel=1e-12
        )
        assert dict(new.nodes)[name].total_energy_j == pytest.approx(value.total_energy_j, abs=1e-7)
    if driven:
        assert report["drives"]["booster"]["shaft_work_j"] == 0


def test_connection_jacobian_support():
    model, state, _ = setup(dense_limit=1, driven=True)
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, offsets, *_ = model._problem(
        state, seeds, 1e-5, model.default_openings, [0.0, 0.0]
    )
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    support = (
        set(zip(*model._sparsity(offsets, model.default_openings))) | evaluate.additional_sparsity
    )
    for row, col in support:
        mask[row, col] = True
    assert not bool((jac[~mask].abs() > 1e-12).any())
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=2e-4, atol=1e-7)


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("driven", [False, True])
def test_independent_cli_pump_connection_audit(closed, driven):
    import copy

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    model, _, initial = setup(backpressure=1.1e6 if closed else 1e6, driven=driven)
    cfg = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 1e-5}],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    corrupted = copy.deepcopy(result)
    corrupted["intervals"][0]["steps"][0]["pump_connections"]["booster"]["shaft_power_w"] += 1.0
    with pytest.raises(ValueError):
        audit(cfg, corrupted)


def test_closed_rotor_brakes_to_rest_and_starts_without_fictitious_load():
    from dataclasses import replace

    from surface.pr_machine_drive import isolated_motion

    model, state, _ = setup(backpressure=1.1e6, driven=True)
    # Finite rotating stock, exact zero-flow trajectory including unilateral rest.
    state = replace(state, rotors=(("booster", 0.01),))
    new, report = model._step(
        state,
        0.1,
        model.default_openings,
        [0.0, 0.0],
        [350.0, 350.0],
        drive_torque_nm={"booster": -1.0},
    )
    assert dict(new.rotors)["booster"] == 0.0
    assert report["pump_connections"]["booster"]["shaft_power_w"] == 0.0
    started, r = model._step(
        new,
        0.1,
        model.default_openings,
        [0.0, 0.0],
        [350.0, 350.0],
        drive_torque_nm={"booster": 1.0},
    )
    target, _, _ = isolated_motion(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        0.1,
        model.drives[0].drive,
    )
    assert dict(started.rotors)["booster"] == pytest.approx(float(target), rel=1e-12)
    assert r["pump_connections"]["booster"]["flow_m3_s"] == 0.0


def test_invalid_drive_command_leaves_original_checkpoint_intact():
    model, state, _ = setup(driven=True)
    before = state.to_dict()
    with pytest.raises(ValueError, match="torque"):
        model.advance(state, 1e-5, drive_torque_nm={"booster": 100.0})
    assert state.to_dict() == before


def test_material_column_pipe_pump_and_rotor_independent_audit():
    from pathlib import Path

    from check_pr_gravity_transport import audit
    from run_pr_gravity_transport import run

    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_driven_pump_column.json"
        ).read_text()
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    steps = [s for interval in result["intervals"] for s in interval["steps"]]
    assert any(abs(s["columns"]["ports"]["liquid_draw"]["flux"][1]) > 0 for s in steps)
    assert any(s["transport"]["pump_connections"]["booster"]["flow_m3_s"] > 0 for s in steps)
    assert any(s["transport"]["drives"]["booster"]["heat_to_node_j"] > 0 for s in steps)


def test_cli_restart_retains_rotor_and_binds_requested_checkpoint():
    import copy

    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    model, _, initial = setup(driven=True)
    cfg = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 1e-5}],
            }
        )
    )
    first = run(cfg)["intervals"][-1]["state"]
    restart = json.loads(json.dumps({**cfg, "initial": {"state": first}}))
    result = json.loads(json.dumps(run(restart)))
    assert audit(restart, result)["accepted"]
    assert result["initial"] == restart["initial"]["state"]
    assert result["intervals"][-1]["state"]["time_s"] == 2e-5
    altered = copy.deepcopy(result)
    altered["initial"]["rotors"][0][1] += 0.01
    with pytest.raises(ValueError, match="restart"):
        audit(restart, altered)
    mismatch = copy.deepcopy(restart)
    mismatch["initial"]["state"]["model_id"] = "incompatible"
    with pytest.raises(ValueError):
        run(mismatch)


@pytest.mark.parametrize("family", ["gravity", "controlled"])
def test_full_surface_cli_restart_preserves_entire_graph(family):
    import copy

    if family == "gravity":
        from check_pr_gravity_transport import audit
        from run_pr_gravity_transport import run
        from test_surface_pr_gravity_transport import setup as plant_setup

        model, state, _ = plant_setup()
    else:
        from check_pr_controlled_surface import audit
        from run_pr_controlled_surface import run
        from test_surface_pr_surface_control import controlled

        model, state = controlled()
    cfg = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [{"duration_s": 1e-7}],
            }
        )
    )
    if family == "controlled":
        cfg["schedule"][0]["targets"] = {"v": 0.5}
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    next_cfg = {**cfg, "initial": {"state": result["intervals"][-1]["state"]}}
    resumed = json.loads(json.dumps(run(next_cfg)))
    assert audit(next_cfg, resumed)["accepted"]
    assert resumed["initial"] == next_cfg["initial"]["state"]
    bad = copy.deepcopy(resumed)
    bad["initial"]["model_id"] = "wrong"
    with pytest.raises(ValueError, match="restart"):
        audit(next_cfg, bad)


def test_parallel_pumps_share_finite_stocks_with_independent_check_valves():
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    base, _, initial = setup(backpressure=1.02e6, driven=True)
    data = json.loads(json.dumps(base.to_dict()))
    second = json.loads(json.dumps(data["pump_connections"][0]))
    second["card"].update(name="standby", speed_ratio=0.5)
    data["pump_connections"].append(second)
    drive = json.loads(json.dumps(data["drives"][0]))
    drive["machine"] = "standby"
    data["drives"].append(drive)
    cfg = json.loads(
        json.dumps(
            {"model": data, "initial": {"nodes": initial}, "schedule": [{"duration_s": 1e-5}]}
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    row = result["intervals"][0]["steps"][0]
    assert row["pump_connections"]["booster"]["flow_m3_s"] > 0
    assert row["pump_connections"]["standby"]["valve_closed"]
    assert row["pump_connections"]["standby"]["flow_m3_s"] == 0.0
    assert row["drives"]["standby"]["shaft_work_j"] == 0.0
    assert row["drives"]["booster"]["shaft_work_j"] > 0
    assert max(map(abs, result["conservation_residuals"])) < 1e-7


def test_multiliquid_suction_is_not_misreported_as_a_timestep_failure(monkeypatch):
    from surface.liquid import EquipmentOperatingError
    from surface.pr_pump_connection import PumpConnectionBundle
    from surface.torch_nonlinear import TorchNewton

    model, _, initial = setup()
    initial["feed"]["mole_fractions"] = [0.0, 0.5, 0.5]
    old = model.initialize(initial)
    assert len(dict(old.nodes)["feed"].phase_roots) == 2
    checkpoint = old.to_dict()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported initial pump state reached Newton/UV enumeration")

    monkeypatch.setattr(TorchNewton, "solve", forbidden)
    with pytest.raises(EquipmentOperatingError, match="booster.*single liquid inlet.*feed"):
        model.advance(old, 1.0)
    assert old.to_dict() == checkpoint
    # An isolated machine must not prevent conserving a multiphase network.
    PumpConnectionBundle(model, old, {"booster": False}, 0)
