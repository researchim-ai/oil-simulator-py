"""Independent limits, conservative inventories, reversal and shutdown."""

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import HydraulicConvergenceError, PropertyModelError
from surface.transient_pipe import DriftFluxPipe, PipeInventory


def pipe(cells=4, **kw):
    return DriftFluxPipe(
        **{
            "length_m": 10,
            "diameter_m": 0.1,
            "cells": cells,
            "liquid_density_kg_m3": 850,
            "gas_constant_j_kg_k": 500,
            "temperature_k": 300,
            "distribution_parameter": 1,
            "drift_velocity_m_s": 0,
            "darcy_factor": 0.02,
            **kw,
        }
    )


def boundary(p, j=0):
    return {"pressure_pa": p, "gas_void_fraction": 0.2, "mixture_velocity_m_s": j}


def test_primitive_closure_and_homogeneous_limit():
    p = pipe()
    s = p.initialize(1e6, 0.2, 0.3)
    q = p.inspect(s)
    assert q["liquid_holdup"] == pytest.approx([0.8] * 4)
    assert q["liquid_velocity_m_s"] == pytest.approx([0.3] * 4)
    assert q["gas_velocity_m_s"] == pytest.approx([0.3] * 4)
    assert sum(s.liquid_mass_kg) == pytest.approx(p.area_m2 * 10 * 0.8 * 850)
    assert sum(s.gas_mass_kg) == pytest.approx(p.area_m2 * 10 * 0.2 * 1e6 / (500 * 300))
    # Wood's mixture sound speed for incompressible liquid + isothermal gas.
    u = p.primitive(1e6, 0.2, 0)
    rho = u[0] + u[1]
    expected = np.sqrt(1e6 / (0.2 * rho))
    assert p._wave_speed(u) == pytest.approx(expected, rel=2e-6)


def test_slip_distribution_and_countercurrent_velocity():
    p = pipe(distribution_parameter=1.2, drift_velocity_m_s=0.2)
    s = p.initialize(1e6, 0.2, 0.1)
    q = p.inspect(s)
    assert q["gas_velocity_m_s"] == pytest.approx([1.2 * 0.1 + 0.2] * 4)
    assert q["liquid_velocity_m_s"] == pytest.approx([(0.1 - 0.2 * 0.32) / 0.8] * 4)
    stopped = p.initialize(1e6, 0.2, 0)
    q = p.inspect(stopped)
    assert min(q["gas_velocity_m_s"]) > 0 and max(q["liquid_velocity_m_s"]) < 0


def test_closed_stationary_line_keeps_inventory_and_pressure():
    p = pipe()
    s = p.initialize(1e6, 0.2)
    r = p.advance(s, 0.05, initial_dt_s=0.01)
    assert r["properties"]["pressure_pa"] == pytest.approx([1e6] * 4, abs=1e-6)
    end = PipeInventory(**r["state"])
    assert end.liquid_mass_kg == pytest.approx(s.liquid_mass_kg, abs=1e-10)
    assert end.gas_mass_kg == pytest.approx(s.gas_mass_kg, abs=1e-10)
    assert end.time_s == 0.05


def test_open_linepack_boundary_component_and_momentum_balance():
    p = pipe(distribution_parameter=1.1, drift_velocity_m_s=0.1)
    s = p.initialize(1e6, 0.2)
    r = p.advance(s, 0.03, left=boundary(1.01e6), right=boundary(1e6), initial_dt_s=0.01)
    end = PipeInventory(**r["state"])
    old = np.array([sum(s.liquid_mass_kg), sum(s.gas_mass_kg), sum(s.momentum_kg_m_s)])
    new = np.array([sum(end.liquid_mass_kg), sum(end.gas_mass_kg), sum(end.momentum_kg_m_s)])
    flux = sum(
        np.array(x["left_boundary_transfer"]) - x["right_boundary_transfer"] for x in r["steps"]
    )
    body = sum(x["body_impulse_kg_m_s"] for x in r["steps"])
    assert new - old == pytest.approx(flux + [0, 0, body], abs=1e-7)
    assert not np.allclose(r["properties"]["pressure_pa"], [1e6] * 4)
    assert s.time_s == 0


def test_reversal_symmetry_and_shutdown_preserves_stored_material():
    p = pipe()
    s = p.initialize(1e6, 0.2)
    a = p.advance(s, 0.02, left=boundary(1.01e6), right=boundary(1e6))
    b = p.advance(s, 0.02, left=boundary(1e6), right=boundary(1.01e6))
    assert a["properties"]["pressure_pa"] == pytest.approx(
        b["properties"]["pressure_pa"][::-1], abs=1e-6
    )
    assert a["properties"]["mixture_velocity_m_s"] == pytest.approx(
        -np.array(b["properties"]["mixture_velocity_m_s"][::-1]), abs=1e-8
    )
    current = PipeInventory(**a["state"])
    stopped = p.advance(current, 0.02)
    end = PipeInventory(**stopped["state"])
    assert sum(end.liquid_mass_kg) == pytest.approx(sum(current.liquid_mass_kg), abs=1e-8)
    assert sum(end.gas_mass_kg) == pytest.approx(sum(current.gas_mass_kg), abs=1e-8)
    assert end.time_s == 0.04


def test_frictionless_uniform_open_transport_and_restart():
    p = pipe(darcy_factor=0)
    s = p.initialize(1e6, 0.2, 0.1)
    kw = {"left": boundary(1e6, 0.1), "right": boundary(1e6, 0.1), "initial_dt_s": 0.01}
    a = p.advance(s, 0.02, **kw)
    saved = copy.deepcopy(a)
    end = PipeInventory(**a["state"])
    first = p.advance(end, 0.02, **kw)
    second = p.advance(PipeInventory(**saved["state"]), 0.02, **kw)
    assert first == second and a == saved
    assert first["properties"]["mixture_velocity_m_s"] == pytest.approx([0.1] * 4, abs=1e-9)


def test_nonuniform_pressures_refine_in_time():
    p = pipe()
    s = p.initialize(1e6, 0.2)
    results = []
    for dt in (0.02, 0.01, 0.005):
        r = p.advance(s, 0.04, left=boundary(1.01e6), right=boundary(1e6), initial_dt_s=dt)
        results.append(np.array(r["properties"]["pressure_pa"]))
    assert np.linalg.norm(results[2] - results[1]) < 0.8 * np.linalg.norm(results[1] - results[0])


def test_bad_domains_and_incompatible_states_fail():
    p = pipe()
    s = p.initialize(1e6, 0.2)
    with pytest.raises(PropertyModelError):
        p.initialize(1e6, 0)
    with pytest.raises(PropertyModelError):
        replace(p, diameter_m=0.2).inspect(s)
    with pytest.raises(ValueError):
        p.advance(s, 0.1, left={"pressure_pa": 1e6})
    singular = pipe(distribution_parameter=2)
    with pytest.raises(PropertyModelError, match="singular"):
        singular.initialize(1e6, 0.8)
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        p.advance(s, 0.1, initial_dt_s=0.01, max_substeps=1)


def test_schedule_shutdown_reversal_exact_json_restart_and_physical_audit():
    import json

    from surface import ConservationError
    from surface.operation import _digest
    from surface.pipe_schedule import run_pipe_schedule

    p = pipe()
    initial = p.initialize(1e6, 0.2)
    periods = [
        {"duration_s": 0.02, "left": boundary(1.01e6), "right": boundary(1e6)},
        {"duration_s": 0.02},
        {"duration_s": 0.02, "left": boundary(1e6), "right": boundary(1.01e6)},
    ]
    full = run_pipe_schedule(p, initial, periods)
    part = run_pipe_schedule(p, initial, periods, stop_after=1)
    resumed = run_pipe_schedule(p, initial, periods, checkpoint=json.loads(json.dumps(part)))
    assert full == resumed
    assert full["state"]["time_s"] == 0.06
    bad = copy.deepcopy(part)
    bad["state"]["liquid_mass_kg"][0] *= 1.001
    bad.pop("checksum")
    bad["checksum"] = _digest(bad)
    with pytest.raises(ConservationError):
        run_pipe_schedule(p, initial, periods, checkpoint=bad)
    with pytest.raises(ValueError, match="fingerprint"):
        run_pipe_schedule(
            replace(p, darcy_factor=0.03),
            replace(initial, model_id=replace(p, darcy_factor=0.03).model_id),
            periods,
            checkpoint=part,
        )
