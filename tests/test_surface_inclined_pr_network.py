"""Inclined pipe coupled to gravity columns at unequal absolute elevations."""

import json
from pathlib import Path

import pytest
import torch
from test_surface_pr_inclined_pipe import inclined

from surface.pr_momentum_thermal import PRThermalMomentumPipe
from surface.stratified_pr import StratifiedPRColumn
from surface.stratified_pr_momentum import StratifiedMomentumState, StratifiedPRMomentumNetwork
from surface.vessel_wall import VesselWall


def network(elevation=1000.0, change=1.0):
    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_stratified_pr_gas.json").read_text()
    )["system"]
    cfg["geometry"].update(diameter_m=0.1, length_m=2.0)
    columns = [
        StratifiedPRColumn.from_dict({**cfg, "elevation_m": elevation + change * i})
        for i in range(2)
    ]
    p = inclined(
        cells=2,
        length_m=max(2.0, 2 * abs(change)),
        elevation_change_m=change,
        reference_elevation_m=elevation + 1.0,
    )
    thermal = PRThermalMomentumPipe(p, VesselWall(100.0, 2.0))
    return StratifiedPRMomentumNetwork(
        thermal,
        columns=columns,
        vessel_walls=[VesselWall(100.0, 2.0)] * 2,
        port_heights_m=[1.0, 1.0],
    )


def test_entire_inclined_system_hydrostatic_rest_and_datum_invariance():
    n = network()
    s = n.initialize_equilibrium(1e6, 320.0, [0.5, 0.5])
    original = s.to_dict()
    out, r = n._step(s, 1e-4, (True, True))
    assert r["port_contact_velocity_m_s"] == pytest.approx([0.0, 0.0], abs=1e-8)
    assert n.pipe.pipe.describe(out.pipe.pipe)["velocity_m_s"] == pytest.approx(
        [0.0, 0.0], abs=1e-8
    )
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    shifted = network(elevation=11000.0)
    t = shifted.initialize_equilibrium(1e6, 320.0, [0.5, 0.5])
    b, _rb = shifted._step(t, 1e-4, (True, True))
    assert out.pipe.pipe.solution_variables == pytest.approx(
        b.pipe.pipe.solution_variables, abs=1e-8
    )
    assert s.to_dict() == original


def test_inclined_joint_jacobian_and_reverse_slope():
    n = network(change=-1.0)
    s = n.initialize_equilibrium(1e6, 320.0, [0.5, 0.5], vessel_wall_temperature_k=[320.0, 350.0])
    y, _, _, evaluate, offsets, _, _ = n._problem(s, 1e-4, (True, True), 4)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-0.01, 0.01, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-6
    assert float(jac[: offsets[0], offsets[0] :].abs().max()) > 0
    a, r = n._step(s, 1e-4, (True, True))
    assert r["port_contact_velocity_m_s"][1] < 0
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    b, closed = n._step(a, 1e-4, (False, False))
    for old, new in zip(a.vessels, b.vessels):
        assert old.column.component_moles == new.column.component_moles
    assert closed["pipe"]["left_flux"][:-2] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert closed["pipe"]["right_flux"][:-2] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_inclined_adaptive_restart():
    n = network()
    s = n.initialize_equilibrium(1e6, 320.0, [0.5, 0.5])
    a = n.advance(s, 1e-5)
    checkpoint = StratifiedMomentumState.from_dict(json.loads(json.dumps(a["state"])))
    restored = StratifiedPRMomentumNetwork.from_dict(json.loads(json.dumps(n.to_dict())))
    restored.inspect(checkpoint)
    b = restored.advance(checkpoint, 1e-5, openings=(False, False))
    assert b["state"]["pipe"]["pipe"]["time_s"] == 2e-5
    assert abs(b["conservation_residuals"][-1]) < 1e-6


def test_large_elevation_energy_and_independent_coupled_audit(monkeypatch):
    from copy import deepcopy

    from check_stratified_pr_momentum import audit
    from run_stratified_pr_momentum import run

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_inclined_equilibrium.json"
        ).read_text()
    )
    config["model"]["pipe"]["pipe"].update(elevation_change_m=100.0, length_m=200.0)
    config["model"]["columns"][1]["elevation_m"] = 1100.0
    config["model"]["vessel_walls"] = [{"heat_capacity_j_k": 1000.0, "fluid_ua_w_k": 2000.0}] * 2
    config["initial"]["equilibrium"]["vessel_wall_temperature_k"] = [320.0, 350.0]
    config["schedule"] = [{"duration_s": 1e-4}]
    original = deepcopy(config)
    report = json.loads(json.dumps(run(config)))
    assert config == original
    assert report["intervals"][0]["steps"][-1]["port_contact_velocity_m_s"][1] < 0

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference invoked runtime gravity/pipe dynamics")

    monkeypatch.setattr(StratifiedPRColumn, "sample", forbidden)
    monkeypatch.setattr(StratifiedPRColumn, "_problem", forbidden)
    monkeypatch.setattr(StratifiedPRMomentumNetwork, "_step", forbidden)
    assert audit(config, report)["accepted"]
    corrupt = deepcopy(report)
    corrupt["intervals"][0]["steps"][0]["pipe"]["body_force_impulse_kg_m_s"] += 1.0
    with pytest.raises(ValueError):
        audit(config, corrupt)
    corrupt = deepcopy(report)
    corrupt["intervals"][0]["steps"][0]["port_energy_rates_with_pe_w"][1] += 100.0
    with pytest.raises(ValueError):
        audit(config, corrupt)
