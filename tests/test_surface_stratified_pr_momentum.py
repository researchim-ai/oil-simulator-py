"""Gravity profile / inertial pipe: conservation, datum invariance and donors."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_surface_pr_momentum_pipe import model

from surface.pr_momentum_thermal import PRThermalMomentumPipe
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
from surface.stratified_pr_momentum import StratifiedMomentumState, StratifiedPRMomentumNetwork
from surface.vessel_wall import VesselWall


def setup(
    *,
    elevation=1000.0,
    delta=100.0,
    temperature=(320.0, 320.0),
    compositions=((0.7, 0.3), (0.2, 0.8)),
    g=9.80665,
    heights=(1.0, 1.0),
    bases=None,
    column_height_m=2.0,
):
    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_stratified_pr_gas.json").read_text()
    )["system"]
    cfg["geometry"].update(diameter_m=0.1, length_m=column_height_m)
    cfg.update(elevation_m=elevation, gravity_m_s2=g)
    p = model(boundary_mode="ends", cells=2, initial_step_s=1e-5)
    thermal = PRThermalMomentumPipe(p, VesselWall(100.0, 2.0))
    columns = [
        StratifiedPRColumn.from_dict(
            {**cfg, "elevation_m": elevation if bases is None else bases[i]}
        )
        for i in range(2)
    ]
    network = StratifiedPRMomentumNetwork(
        thermal, columns=columns, vessel_walls=[VesselWall(100.0, 2.0)] * 2, port_heights_m=heights
    )
    states = [
        StratifiedPRState.from_dict(
            c.initialize(1e6 + delta * (1 if i == 0 else -1), temperature[i], compositions[i])[
                "state"
            ]
        )
        for i, c in enumerate(columns)
    ]
    if delta == 0 and compositions[0] == compositions[1]:
        sample = columns[0].sample(states[0], heights[0])
        pp, tt, xx = sample["pressure_pa"], sample["temperature_k"], sample["mole_fractions"]
    else:
        pp, tt, xx = 1e6, 320.0, [0.5, 0.5]
    pipe_state = thermal.initialize(p.initialize(pp, tt, xx), 320.0)
    return network, network.initialize(
        pipe_state, [network.initial_vessel_state(i, s, 320.0) for i, s in enumerate(states)]
    )


def test_gravitational_rest_and_component_energy_ledger():
    n, s = setup(delta=0.0, compositions=((0.5, 0.5), (0.5, 0.5)))
    final, r = n._step(s, 1e-4, (True, True))
    assert r["port_contact_velocity_m_s"] == pytest.approx([0.0, 0.0], abs=2e-8)
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    assert final.vessels[0].column.component_moles == pytest.approx(
        s.vessels[0].column.component_moles, abs=1e-10
    )
    assert r["quadrature_order"] >= 8
    assert r["quadrature_error"] <= 1


def test_gravitational_transfer_has_same_port_flux_in_both_equations():
    n, s = setup()
    out, r = n._step(s, 1e-4, (True, True))
    assert all(v > 0 for v in r["port_contact_velocity_m_s"])
    for i, side in enumerate(["left_flux", "right_flux"]):
        sign = -1 if i == 0 else 1
        delta = (
            np.array(out.vessels[i].column.component_moles) - s.vessels[i].column.component_moles
        )
        assert delta == pytest.approx(sign * 1e-4 * np.array(r["pipe"][side][:-2]), abs=1e-10)
    flux = np.array(r["pipe"]["left_flux"][:-2])
    assert flux / flux.sum() == pytest.approx(r["port_mole_fractions"][0], abs=1e-8)
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    # The pressure actually comes from the local profile, not the mean column P.
    assert abs(r["port_pressure_pa"][0] - r["columns"][0]["layers"][0]["pressure_pa"][0]) > 0.1


def test_elevation_datum_shift_preserves_thermal_and_flow_solution():
    n, s = setup(elevation=0.0)
    shifted, t = setup(elevation=10000.0)
    a, ra = n._step(s, 1e-4, (True, True))
    b, rb = shifted._step(t, 1e-4, (True, True))
    assert a.pipe.pipe.solution_variables == pytest.approx(b.pipe.pipe.solution_variables, abs=1e-8)
    assert ra["port_contact_velocity_m_s"] == pytest.approx(
        rb["port_contact_velocity_m_s"], abs=1e-8
    )
    for x, y in zip(a.vessels, b.vessels):
        assert x.column.temperature_k == pytest.approx(y.column.temperature_k, abs=1e-7)
        assert x.column.component_moles == pytest.approx(y.column.component_moles, abs=1e-10)
    assert max(map(abs, rb["conservation_residuals"])) < 1e-6


def test_cross_jacobian_and_static_preparation_outside_residual(monkeypatch):
    n, s = setup()
    y, _, _, evaluate, offsets, _, _ = n._problem(s, 1e-4, (True, True), 4)

    def forbidden(*args, **kwargs):
        raise AssertionError("Static host preparation inside Newton")

    monkeypatch.setattr(n.pipe.pipe, "_support", forbidden)
    for c in n.columns:
        monkeypatch.setattr(c, "sample", forbidden)
        monkeypatch.setattr(c, "_problem", forbidden)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-0.01, 0.01, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-6
    assert float(jac[: offsets[0], offsets[0] :].abs().max()) > 0
    assert float(jac[offsets[0] :, : offsets[0]].abs().max()) > 0


def test_restart_exact_interval_and_isolation():
    n, s = setup()
    original = s.to_dict()
    report = n.advance(s, 1e-5, openings=(False, False))
    out = StratifiedMomentumState.from_dict(json.loads(json.dumps(report["state"])))
    restored = StratifiedPRMomentumNetwork.from_dict(json.loads(json.dumps(n.to_dict())))
    restored.inspect(out)
    assert out.time_s == 1e-5
    assert all(v.time_s == out.time_s for v in out.vessels)
    assert all(v.next_step_s == out.pipe.pipe.next_step_s for v in out.vessels)
    assert s.to_dict() == original
    for a, b in zip(s.vessels, out.vessels):
        assert a.column.component_moles == b.column.component_moles
    a = n.advance(out, 1e-5)
    b = restored.advance(out, 1e-5)
    assert a["state"] == b["state"]


def test_zero_gravity_reduces_to_uniform_finite_vessel_model():
    from surface.pr_momentum_vessels import PRMomentumVesselNetwork

    n, s = setup(g=0.0)
    uniform = PRMomentumVesselNetwork(
        n.pipe, vessel_volumes_m3=[c.geometry.volume_m3 for c in n.columns], vessel_walls=n.walls
    )
    old = uniform.initialize(
        s.pipe,
        pressure_pa=[np.exp(v.column.solution_variables[0]) for v in s.vessels],
        temperature_k=[v.column.temperature_k for v in s.vessels],
        mole_fractions=[v.column.phase_seeds[0] for v in s.vessels],
        wall_temperature_k=[v.wall_temperature_k for v in s.vessels],
    )
    a, ra = n._step(s, 1e-4, (True, True))
    b, rb = uniform._step(old, 1e-4, (True, True))
    assert a.pipe.pipe.solution_variables == pytest.approx(b.pipe.pipe.solution_variables, abs=1e-8)
    assert ra["port_contact_velocity_m_s"] == pytest.approx(
        rb["port_contact_velocity_m_s"], abs=1e-8
    )
    for x, y in zip(a.vessels, b.vessels):
        assert x.column.component_moles == pytest.approx(y.component_moles, abs=1e-9)
        assert x.column.total_energy_j == pytest.approx(y.internal_energy_j, abs=1e-6)


def test_different_base_elevations_use_local_head_at_shared_absolute_port():
    n, s = setup(
        delta=0.0, bases=(1000.0, 999.0), heights=(0.5, 1.5), compositions=((0.5, 0.5), (0.5, 0.5))
    )
    _, r = n._step(s, 1e-4, (True, True))
    assert r["port_pressure_pa"][0] > r["port_pressure_pa"][1]
    assert r["port_contact_velocity_m_s"][1] > 0
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_reverse_pressure_uses_sampled_right_donor_and_closed_ports_block_mass():
    n, s = setup(delta=-100.0)
    out, r = n._step(s, 1e-4, (True, True))
    assert all(v < 0 for v in r["port_contact_velocity_m_s"])
    flux = np.array(r["pipe"]["right_flux"][:-2])
    assert flux / flux.sum() == pytest.approx(r["port_mole_fractions"][1], abs=1e-8)
    closed, report = n._step(out, 1e-4, (False, False))
    for old, new in zip(out.vessels, closed.vessels):
        assert new.column.component_moles == old.column.component_moles
    assert report["pipe"]["left_flux"][:-2] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert report["pipe"]["right_flux"][:-2] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_port_approximation_and_checkpoint_tampering_are_rejected():
    from surface.liquid import PropertyModelError

    n, s = setup()
    limited = StratifiedPRMomentumNetwork(
        n.pipe,
        columns=n.columns,
        vessel_walls=n.walls,
        port_heights_m=n.heights,
        maximum_transverse_head_fraction=1e-12,
    )
    state = limited.initialize(
        s.pipe,
        [
            limited.initial_vessel_state(i, v.column, v.wall_temperature_k)
            for i, v in enumerate(s.vessels)
        ],
    )
    with pytest.raises(PropertyModelError, match="transverse"):
        limited._step(state, 1e-5, (True, True))
    with pytest.raises(PropertyModelError, match="clock"):
        n.inspect(replace(s, vessels=(replace(s.vessels[0], time_s=1.0), s.vessels[1])))
    with pytest.raises(ValueError, match="equal absolute"):
        StratifiedPRMomentumNetwork(
            n.pipe, columns=n.columns, vessel_walls=n.walls, port_heights_m=(0.5, 1.5)
        )


def test_independent_finer_profile_audit_and_corruption_rejection(monkeypatch):
    from copy import deepcopy

    from check_stratified_pr_momentum import audit
    from run_stratified_pr_momentum import run

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_stratified_pr_momentum.json"
        ).read_text()
    )
    config["schedule"] = [{"duration_s": 1e-5}]
    before = deepcopy(config)
    report = json.loads(json.dumps(run(config)))
    assert config == before

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference audit invoked Torch column dynamics")

    monkeypatch.setattr(StratifiedPRColumn, "_problem", forbidden)
    monkeypatch.setattr(StratifiedPRColumn, "sample", forbidden)
    monkeypatch.setattr(StratifiedPRColumn, "inspect", forbidden)
    monkeypatch.setattr(StratifiedPRMomentumNetwork, "_step", forbidden)
    assert audit(config, report)["accepted"]
    for field in ("total_energy_j", "temperature_k"):
        corrupt = deepcopy(report)
        corrupt["intervals"][0]["steps"][0]["state"]["vessels"][0]["column"][field] += 1.0
        with pytest.raises(ValueError):
            audit(config, corrupt)
    corrupt = deepcopy(report)
    corrupt["intervals"][0]["steps"][0]["port_energy_rates_with_pe_w"][0] *= 2
    with pytest.raises(ValueError):
        audit(config, corrupt)


def test_tall_columns_keep_resolved_gravity_and_component_segregation():
    n, s = setup(column_height_m=100.0, heights=(50.0, 50.0))
    out, r = n._step(s, 1e-4, (True, True))
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    layer = r["columns"][0]["layers"][0]
    assert layer["pressure_pa"][0] - layer["pressure_pa"][-1] > 1000.0
    assert abs(layer["mole_fractions"][0][0] - layer["mole_fractions"][-1][0]) > 1e-5
    assert out.vessels[0].column.quadrature_order >= 8


def test_actual_two_phase_column_is_rejected_instead_of_flattened():
    from surface.liquid import PropertyModelError
    from surface.pr_momentum_pipe import PRMomentumPipe

    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_stratified_pr.json").read_text()
    )
    c = StratifiedPRColumn.from_dict(cfg["system"])
    cs = StratifiedPRState.from_dict(c.initialize(**cfg["initial"])["state"])
    assert len(cs.phase_roots) == 2
    p = PRMomentumPipe(
        c.fluid,
        length_m=1.0,
        diameter_m=0.05,
        cells=2,
        wave_speed_bound_m_s=500.0,
        maximum_velocity_m_s=50.0,
    )
    thermal = PRThermalMomentumPipe(p, VesselWall(100.0, 0.0))
    n = StratifiedPRMomentumNetwork(
        thermal,
        columns=[c, c],
        vessel_walls=[VesselWall(100.0, 0.0)] * 2,
        port_heights_m=[1.0, 1.0],
    )
    pipe = thermal.initialize(p.initialize(1e6, 320.0, cfg["initial"]["mole_fractions"]), 320.0)
    with pytest.raises(PropertyModelError, match="one stable phase"):
        n.initialize(pipe, [n.initial_vessel_state(i, cs, cs.temperature_k) for i in range(2)])


def test_temporal_refinement_of_gravitational_coupled_wave():
    n, s = setup(compositions=((0.5, 0.5), (0.5, 0.5)))
    results = []
    for pieces in (1, 2, 4):
        current = s
        for _ in range(pieces):
            current, _ = n._step(current, 0.001 / pieces, (True, True))
        results.append(np.asarray(current.pipe.pipe.momentum_kg_m_s))
    assert np.linalg.norm(results[1] - results[2]) < 0.8 * np.linalg.norm(results[0] - results[2])
