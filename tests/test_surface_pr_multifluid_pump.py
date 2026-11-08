"""Phase-selective liquid pump: force and shaft heat share one Newton solve."""

from dataclasses import asdict

import pytest
from test_surface_pr_multifluid_gravity import inclined
from test_surface_pr_multifluid_pipe import initial
from test_surface_pr_pump_four_quadrant import mapping

from surface.pr_multifluid_network import PRMultifluidNetwork


def pumped(speed=100.0, dense=128):
    raw = inclined(slope=0.0, dense=dense).to_dict()
    raw.update(
        model="pr_multifluid_network_v4",
        pumps=[
            {
                "name": "booster",
                "cell": "p0",
                "phase": 1,
                "minimum_density_kg_m3": 200.0,
                "map": asdict(mapping()),
                "speed_rad_s": speed,
            }
        ],
    )
    return PRMultifluidNetwork.from_dict(raw)


@pytest.mark.parametrize("speed", [-100.0, 0.0, 100.0])
def test_signed_phase_pump_force_work_and_restart(speed):
    model = pumped(speed)
    old = initial(model, velocity=0.0)
    end, report = model.step(old, 1e-6)
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    if speed:
        assert end.cells[1].phases[1].momentum_kg_m_s * speed > 0
        assert report["external_work_j"] > 0
    else:
        assert report["external_work_j"] == 0
    assert model._stored_energy(end) - model._stored_energy(old) == pytest.approx(
        report["external_work_j"], abs=1e-7
    )
    assert PRMultifluidNetwork.from_dict(model.to_dict()).model_id == model.model_id


def test_pump_work_independent_report_audit_and_tamper_rejection(monkeypatch):
    import copy
    import json

    from check_pr_multifluid_pipe import audit
    from run_pr_multifluid_pipe import run

    model = pumped()
    old = initial(model, velocity=0.0)
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": old.to_dict()},
                "schedule": [{"duration_s": 2e-7, "maximum_step_s": 2e-7}],
            }
        )
    )
    result = json.loads(json.dumps(run(config)))
    from surface.pr_pump_four_quadrant import SignedPumpMap

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference called runtime pump or transport")

    monkeypatch.setattr(SignedPumpMap, "evaluate", forbidden)
    monkeypatch.setattr(PRMultifluidNetwork, "_network_operator", forbidden)
    assert audit(config, result)["accepted"]
    tampered = copy.deepcopy(result)
    tampered["intervals"][0]["steps"][0]["external_work_j"] += 0.01
    with pytest.raises(ValueError):
        audit(config, tampered)


def test_pump_and_gravity_share_vector_impulse_and_energy_in_sparse_newton():
    outputs = []
    for dense in [128, 1]:
        raw = inclined(slope=0.6, dense=dense).to_dict()
        raw.update(model="pr_multifluid_network_v4", pumps=pumped().to_dict()["pumps"])
        model = PRMultifluidNetwork.from_dict(raw)
        old = initial(model, velocity=0.0)
        end, report = model.step(old, 1e-6)
        assert max(map(abs, report["conservation_residuals"])) < 1e-7
        assert report["external_work_j"] > 0
        outputs.append([p.pressure_pa for c in end.cells for p in c.phases])
    assert outputs[0] == pytest.approx(outputs[1], abs=1e-5)


def test_outside_liquid_density_envelope_rejects_without_partial_state():
    from surface.liquid import HydraulicConvergenceError

    raw = pumped().to_dict()
    raw["pumps"][0]["minimum_density_kg_m3"] = 1e6
    model = PRMultifluidNetwork.from_dict(raw)
    old = initial(model, velocity=0.0)
    before = old.to_dict()
    with pytest.raises(HydraulicConvergenceError):
        model.step(old, 1e-6)
    assert old.to_dict() == before
