"""Finite inventory ports: rest, acceleration, isolation, reversal and restart."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_surface_pr_momentum_pipe import model

from surface.liquid import ConservationError
from surface.pr_momentum_thermal import PRThermalMomentumPipe
from surface.pr_momentum_vessels import (
    PRMomentumVesselNetwork,
    PRMomentumVesselNetworkState,
    hllc_port,
)
from surface.vessel_wall import VesselWall


def setup(
    *,
    pressure=(1000100.0, 999900.0),
    compositions=((0.7, 0.3), (0.2, 0.8)),
    device="cpu",
    **options,
):
    options = {"boundary_mode": "ends", "initial_step_s": 1e-5, **options}
    p = model(device, **options)
    thermal = PRThermalMomentumPipe(p, VesselWall(100.0, 2.0, 1.0, 300.0, 320.0))
    n = PRMomentumVesselNetwork(
        thermal,
        vessel_volumes_m3=(0.01, 0.01),
        vessel_walls=(VesselWall(100.0, 2.0), VesselWall(100.0, 2.0)),
    )
    s = n.initialize(
        thermal.initialize(p.initialize(1e6, 320.0, [0.5, 0.5]), 320.0),
        pressure_pa=pressure,
        temperature_k=(320.0, 320.0),
        mole_fractions=compositions,
        wall_temperature_k=(320.0, 320.0),
    )
    return n, s


def test_rest_and_contact_donor_without_diffusion():
    n, s = setup(pressure=(1e6, 1e6), compositions=((0.5, 0.5), (0.5, 0.5)))
    _out, report = n._step(s, 1e-4, (True, True))
    assert report["port_contact_velocity_m_s"] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert max(map(abs, report["conservation_residuals"])) < 1e-6
    p = n.pipe.pipe
    d = p._decode(
        p.fluid.tensor(s.pipe.pipe.solution_variables), p._support(s.pipe.pipe.component_moles)
    )
    a = (d["u"][0], d["flux"][0], d["p"][0], d["v"][0], d["q"]["density_kg_m3"][0], d["entropy"][0])
    # Equal p and v, unequal density/composition is a stationary contact.
    n, s = setup(pressure=(1e6, 1e6))
    rd = n._decode(
        p.fluid.tensor([v.solution_variables for v in s.vessels]).flatten(),
        p._support(s.pipe.pipe.component_moles),
    )
    b = (
        rd["u"][0],
        rd["flux"][0],
        rd["p"][0],
        d["v"][0],
        rd["q"]["density_kg_m3"][0],
        rd["entropy"][0],
    )
    flux, _, contact, _, valid = hllc_port(a, b, 500.0)
    assert bool(valid)
    assert float(contact) == pytest.approx(0.0, abs=1e-10)
    assert flux[:-2].cpu().tolist() == pytest.approx([0.0, 0.0], abs=1e-9)


def test_acceleration_conservation_actual_donor_and_isolation():
    n, s = setup()
    out, report = n._step(s, 1e-4, (True, True))
    assert all(v > 0 for v in report["port_contact_velocity_m_s"])
    assert out.vessels[0].component_moles[0] < s.vessels[0].component_moles[0]
    assert out.vessels[1].component_moles[0] > s.vessels[1].component_moles[0]
    flux = np.array(report["pipe"]["left_flux"][:-2])
    assert flux / flux.sum() == pytest.approx([0.7, 0.3], abs=1e-8)
    assert max(map(abs, report["conservation_residuals"])) < 1e-6
    isolated, closed = n._step(out, 1e-4, (False, False))
    assert closed["pipe"]["left_flux"][:-2] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert closed["pipe"]["right_flux"][:-2] == pytest.approx([0.0, 0.0], abs=1e-12)
    for a, b in zip(out.vessels, isolated.vessels):
        assert a.component_moles == b.component_moles
    assert closed["entropy_production_j_k"] >= -1e-7


def test_reverse_uses_right_donor_and_restart():
    n, s = setup(pressure=(999900.0, 1000100.0))
    out, r = n._step(s, 1e-4, (True, True))
    assert all(v < 0 for v in r["port_contact_velocity_m_s"])
    flux = np.array(r["pipe"]["right_flux"][:-2])
    assert flux / flux.sum() == pytest.approx([0.2, 0.8], abs=1e-8)
    restored = PRMomentumVesselNetwork.from_dict(json.loads(json.dumps(n.to_dict())))
    checkpoint = PRMomentumVesselNetworkState.from_dict(json.loads(json.dumps(out.to_dict())))
    restored.inspect(checkpoint)
    a = n.advance(out, 1e-5)
    b = restored.advance(checkpoint, 1e-5)
    assert a["state"] == b["state"]
    assert abs(a["conservation_residuals"][-1]) < 1e-6
    with pytest.raises(ConservationError):
        n.inspect(
            replace(
                out,
                vessels=(
                    replace(
                        out.vessels[0], internal_energy_j=out.vessels[0].internal_energy_j + 1.0
                    ),
                    out.vessels[1],
                ),
            )
        )


def test_whole_coupled_jacobian_includes_vessel_pipe_cross_terms():
    n, s = setup()
    y, _, _, evaluate, k = n._problem(s, 1e-4, (True, True))
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-0.1, 0.1, len(y), dtype=y.dtype, device=y.device)
    eps = 1e-6
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-6
    assert float(jac[:k, k:].abs().max()) > 0
    assert float(jac[k:, :k].abs().max()) > 0


def test_independent_numpy_audit_and_corruption_rejection(monkeypatch):
    from copy import deepcopy
    from pathlib import Path

    from check_pr_momentum_vessels import audit
    from run_pr_momentum_vessels import run

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_momentum_vessels.json"
        ).read_text()
    )
    config["schedule"] = [{"duration_s": 1e-5}]
    before = deepcopy(config)
    report = json.loads(json.dumps(run(config)))
    assert config == before

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent audit invoked runtime dynamics")

    monkeypatch.setattr(PRMomentumVesselNetwork, "_decode", forbidden)
    monkeypatch.setattr(PRMomentumVesselNetwork, "_step", forbidden)
    monkeypatch.setattr(PRMomentumVesselNetwork, "inspect", forbidden)
    assert audit(config, report)["accepted"]
    for key in ("internal_energy_j", "wall_temperature_k"):
        bad = deepcopy(report)
        bad["intervals"][0]["steps"][0]["state"]["vessels"][0][key] += 1.0
        with pytest.raises((ValueError, AssertionError)):
            audit(config, bad)
    bad = deepcopy(report)
    bad["intervals"][0]["steps"][0]["pipe"]["left_flux"][0] *= 2
    with pytest.raises((ValueError, AssertionError)):
        audit(config, bad)


def test_temporal_refinement_against_finer_coupled_solution():
    n, s = setup(compositions=((0.5, 0.5), (0.5, 0.5)))
    answers = []
    for subdivisions in (1, 2, 8):
        current = s
        for _ in range(subdivisions):
            current, _ = n._step(current, 0.001 / subdivisions, (True, True))
        answers.append(np.array(current.pipe.pipe.momentum_kg_m_s))
    assert np.linalg.norm(answers[1] - answers[2]) < 0.7 * np.linalg.norm(answers[0] - answers[2])


def test_spatial_refinement_coupled_pressure_wave():
    answers = []
    for cells in (4, 8, 16):
        n, s = setup(cells=cells, compositions=((0.5, 0.5), (0.5, 0.5)))
        for _ in range(4):
            s, _ = n._step(s, 0.00025, (True, True))
        # Compare cell averages on the shared coarse grid.
        a = np.array(n.pipe.pipe.describe(s.pipe.pipe)["velocity_m_s"])
        answers.append(a.reshape(4, -1).mean(axis=1))
    assert np.linalg.norm(answers[1] - answers[2]) < 0.8 * np.linalg.norm(answers[0] - answers[2])


def test_heated_receiver_causes_reversal_then_isolation_and_restart():
    base, old = setup(compositions=((0.5, 0.5), (0.2, 0.8)))
    n = PRMomentumVesselNetwork(
        base.pipe,
        vessel_volumes_m3=base.volumes,
        vessel_walls=(VesselWall(1000.0, 2000.0), VesselWall(1000.0, 2000.0)),
    )
    s = n.initialize(
        old.pipe,
        pressure_pa=(1000100.0, 999900.0),
        temperature_k=(320.0, 320.0),
        mole_fractions=((0.5, 0.5), (0.2, 0.8)),
        wall_temperature_k=(320.0, 350.0),
    )
    initial = s.to_dict()
    s, r = n._step(s, 1e-5, (True, True))
    assert r["port_contact_velocity_m_s"][1] > 0
    reversed_flow = False
    for _ in range(12):
        s, r = n._step(s, 5e-5, (True, True))
        if r["port_contact_velocity_m_s"][1] < 0:
            flux = np.array(r["pipe"]["right_flux"][:-2])
            actual = np.array(s.vessels[1].component_moles)
            assert flux / flux.sum() == pytest.approx(actual / actual.sum(), abs=1e-8)
            reversed_flow = True
    assert reversed_flow
    shut, _closed = n._step(s, 5e-5, (False, False))
    for a, b in zip(s.vessels, shut.vessels):
        assert a.component_moles == b.component_moles
    _reopened, report = n._step(shut, 5e-5, (True, True))
    assert report["port_contact_velocity_m_s"][1] < 0
    assert max(map(abs, report["conservation_residuals"])) < 1e-5
    assert initial["pipe"]["pipe"]["time_s"] == 0.0


def test_newton_residual_does_not_rebuild_static_inventory(monkeypatch):
    n, s = setup()
    y, _, _, evaluate, _ = n._problem(s, 1e-4, (True, True))

    def forbidden(*args, **kwargs):
        raise AssertionError("Newton residual rebuilt support or host inventory")

    monkeypatch.setattr(n.pipe.pipe, "_support", forbidden)
    residual, _ = evaluate(y)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    assert bool(torch.isfinite(residual).all())
    assert bool(torch.isfinite(jac).all())


def test_failed_coupled_interval_is_atomic_and_isolation_is_binary():
    from surface.liquid import HydraulicConvergenceError

    n, s = setup(maximum_attempts=1)
    before = s.to_dict()
    with pytest.raises(ValueError):
        n.advance(s, 1e-5, openings=[0.5, True])
    with pytest.raises(HydraulicConvergenceError):
        n.advance(s, 0.001)
    assert s.to_dict() == before
