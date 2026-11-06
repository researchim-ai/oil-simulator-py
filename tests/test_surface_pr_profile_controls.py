"""Persistent pipe isolation and shared PI/ESD must survive full restart."""

import json
from pathlib import Path

import pytest

from surface.actuator import ConnectionActuator
from surface.pr_control_channels import PRMotorControl, PRSurfaceTrip
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
from surface.pr_layered_transport import PRLayeredTransportNetwork
from surface.pr_profile_graph import PRProfileGraphNetwork
from surface.pr_surface_control import PRControlledSurface, PRSurfacePI
from surface.stratified_pr import StratifiedPRColumn

ROOT = Path(__file__).resolve().parents[1]


def controlled_request():
    cfg = json.loads((ROOT / "configs/surface_pr_signed_pump_reversal.json").read_text())
    raw = cfg["model"]
    raw["drives"][0]["initial_speed_rad_s"] = -0.1
    physics = raw["drives"][0]["drive"]
    physics["max_electrical_draw_w"] = physics["max_electrical_export_w"] = 1000.0
    transport = PREquilibriumTransportNetwork.from_dict(raw)
    column = StratifiedPRColumn.from_dict(
        {
            **json.loads((ROOT / "configs/surface_stratified_pr.json").read_text())["system"],
            "fluid": transport.fluid.to_dict(),
            "gravity_m_s2": transport.gravity_m_s2,
            "elevation_m": -0.01,
            "geometry": {
                "orientation": "inclined",
                "diameter_m": 0.02,
                "length_m": 0.04,
                "inclination_rad": 0.0,
            },
        }
    )
    pipe = PRLayeredTransportNetwork([column], wave_speed_bound_m_s=10000)
    plant = PRProfileGraphNetwork(transport, {"bypass": (pipe, ("feed", "return"))})
    motor = PRMotorControl(
        "motor", "booster", ConnectionActuator(100, 100, 0.5, max_position_change=1), -2, 2
    )
    trip = PRSurfaceTrip(
        "esd",
        "motor",
        "pressure",
        "feed",
        0.999e6,
        0.99e6,
        isolate_switches=(
            "profile:bypass:left",
            "profile:bypass:right",
            "booster",
            "inlet",
            "outlet",
        ),
    )
    control = PRControlledSurface(
        plant,
        [],
        [PRSurfacePI("motor", -0.2, 0.01, 0.01, measurement="speed", action=-1)],
        motors=[motor],
        trips=[trip],
        initial_step_s=1e-6,
    )
    return {
        "model": control.to_dict(),
        "initial": {
            "plant": {
                "transport": cfg["initial"]["nodes"],
                "profiles": {
                    "bypass": {
                        "pressure_pa": 1e6,
                        "temperature_k": 350.0,
                        "mole_fractions": [0, 1, 0],
                    }
                },
            },
            "positions": {"motor": 0.5},
            "integrals": {},
        },
        "schedule": [
            {"duration_s": 1e-6, "profile_openings": {"bypass": [True, True]}},
            {"duration_s": 1e-6},
        ],
    }


@pytest.fixture(scope="module")
def tripped():
    from run_pr_controlled_surface import run

    request = json.loads(json.dumps(controlled_request()))
    return request, json.loads(json.dumps(run(request)))


def test_shared_pi_and_esd_override_opening_and_latch(tripped):
    from check_pr_controlled_surface import audit

    cfg, report = tripped
    for interval in report["intervals"]:
        assert interval["state"]["trips"]["esd"]["latched"]
        assert interval["state"]["plant"]["profile_openings"]["bypass"] == [False, False]
        for step in interval["steps"]:
            assert step["plant"]["openings"]["profile:bypass:left"] is False
            assert step["plant"]["openings"]["profile:bypass:right"] is False
    assert audit(cfg, report)["accepted"]


def test_controlled_restart_preserves_protection_and_isolation(tripped):
    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    cfg, report = tripped
    last = report["intervals"][-1]["state"]
    cfg = {**cfg, "initial": {"state": last}, "schedule": [{"duration_s": 1e-7}]}
    continued = json.loads(json.dumps(run(cfg)))
    assert continued["initial"] == last
    assert continued["intervals"][-1]["state"]["plant"]["profile_openings"]["bypass"] == [
        False,
        False,
    ]
    assert audit(cfg, continued)["accepted"]
    cfg["schedule"] = [{"duration_s": 1e-7, "reset_trips": ["esd"]}]
    with pytest.raises(ValueError, match="hysteresis"):
        run(cfg)


def test_independent_audit_rejects_reopened_native_endpoint(tripped, monkeypatch):
    from check_pr_controlled_surface import audit

    cfg, report = tripped

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent auditor called the runtime solver")

    monkeypatch.setattr(PRControlledSurface, "advance", forbidden)
    monkeypatch.setattr(PRProfileGraphNetwork, "advance", forbidden)
    monkeypatch.setattr(StratifiedPRColumn, "initialize", forbidden)
    monkeypatch.setattr(StratifiedPRColumn, "equilibrate", forbidden)
    assert audit(cfg, report)["accepted"]
    damaged = json.loads(json.dumps(report))
    damaged["intervals"][0]["steps"][0]["plant"]["openings"]["profile:bypass:left"] = True
    with pytest.raises(ValueError):
        audit(cfg, damaged)


def test_manual_isolation_persists_without_protection_and_reopens_explicitly():
    from test_surface_pr_profile_graph import setup

    from surface.pr_profile_graph import ProfileGraphState

    model, initial = setup()
    closed, _ = model.step(initial, 1e-7, profile_openings={"line_1": [False, False]})
    request = json.loads(json.dumps(model.to_dict()))
    restored = PRProfileGraphNetwork.from_dict(request)
    checkpoint = ProfileGraphState.from_dict(json.loads(json.dumps(closed.to_dict())))
    end = restored.advance(checkpoint, 1e-7, maximum_attempts=1)
    kept = ProfileGraphState.from_dict(end["state"])
    assert dict(kept.profile_openings)["line_1"] == (False, False)
    assert end["openings"]["profile:line_1:left"] is False
    a, b = dict(checkpoint.profiles)["line_1"].cells[0], dict(kept.profiles)["line_1"].cells[0]
    assert b.profile.component_moles == pytest.approx(a.profile.component_moles, abs=1e-10)
    assert b.total_energy_j == pytest.approx(a.total_energy_j, abs=1e-7)
    opened, _ = restored.step(
        kept, 1e-7, openings={"profile:line_1:left": True, "profile:line_1:right": True}
    )
    assert dict(opened.profile_openings)["line_1"] == (True, True)
    assert dict(opened.profiles)["line_1"].cells[0].momentum_kg_m_s > 0
    before = kept.to_dict()
    with pytest.raises(ValueError, match="Conflicting"):
        restored.step(
            kept,
            1e-7,
            profile_openings={"line_1": [True, True]},
            openings={"profile:line_1:left": False},
        )
    assert kept.to_dict() == before


def test_endpoint_flags_are_strict_checkpoint_coordinates():
    from surface.liquid import PropertyModelError
    from surface.pr_profile_graph import ProfileGraphState

    cfg = controlled_request()
    model = PRControlledSurface.from_dict(cfg["model"]).plant
    # Cold start and malformed checkpoint rejection are exercised by the full
    # fixture; schema rejection itself must not require a hydraulic solve.
    from run_pr_controlled_surface import run

    result = json.loads(json.dumps(run({**cfg, "schedule": [{"duration_s": 1e-8}]})))
    state = result["intervals"][-1]["state"]["plant"]
    state["profile_openings"]["bypass"][0] = 1
    with pytest.raises(PropertyModelError, match="flags"):
        model.inspect(ProfileGraphState.from_dict(state))


def test_remote_profile_level_feedback_is_independently_reconstructed():
    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    cfg = json.loads(json.dumps(controlled_request()))
    cfg["model"].pop("trips")
    pi = cfg["model"]["controllers"][0]
    pi.update(measurement="level", sensor="profile:bypass:0", component="hexane", setpoint=0.01)
    cfg["schedule"] = [{"duration_s": 1e-7}]
    report = json.loads(json.dumps(run(cfg)))
    first = report["intervals"][0]["steps"][0]
    assert first["measurements"]["motor"] == pytest.approx(0.02, abs=1e-9)
    assert audit(cfg, report)["accepted"]
    first["measurements"]["motor"] = 0.0
    with pytest.raises(ValueError):
        audit(cfg, report)
