"""Active phase utility heat, independent balances and connected equipment."""

from copy import deepcopy

import pytest
import torch
from test_surface_pr_active_multifluid_pipe import phase, setup

from surface.liquid import PropertyModelError


def test_independent_exact_comparison_cannot_hide_error_behind_zero_over_zero():
    from check_stratified_pr_flow import close

    close([0.0, 1.0], [0.0, 1.0], atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="audit mismatch"):
        close([0.0, 2.0], [0.0, 1.0], atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="audit mismatch"):
        close([0.0, 1.0], [0.0, 2.0], atol=0.0, rtol=0.0)


def gas_pipe(device="cpu"):
    from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe

    pipe, v = setup(mode="periodic", count=2)
    pipe = PRActiveMultifluidPipe.from_dict(pipe.to_dict(), device=device)
    cell = {"gas": phase(pipe.fluid, 1e6, 320.0, "vapor", v, x=(0.5, 0.5))}
    return pipe, pipe.initialize([cell, cell])


@pytest.mark.parametrize("rate,bath", [(1e5, 500.0), (-1e5, 200.0), (0.0, None)])
def test_single_phase_heat_matches_energy_and_scalar_pr_auditor(rate, bath):
    from check_pr_active_surface import pipe_audit

    pipe, old = gas_pipe()
    dt = 0.25 * pipe.maximum_step_s
    new, r = pipe.step(old, dt, heat_rates_w=[rate] * 2, heat_temperatures_k=[bath] * 2)
    pipe_audit(pipe, old, new, r)
    assert r["external_heat_j"] == pytest.approx(2 * dt * rate, abs=1e-14)
    assert max(map(abs, r["conservation_residuals"])) < 1e-8
    for a, b in zip(old.cells, new.cells):
        assert len(b) == 1
        assert dict(b)["gas"].total_energy_j - dict(a)["gas"].total_energy_j == pytest.approx(
            dt * rate, abs=1e-12
        )
    assert r["entropy_production_j_k"] >= -1e-7


def test_multiphase_heat_is_named_and_keeps_own_thermodynamics():
    from check_pr_active_surface import pipe_audit

    pipe, v = setup(mode="periodic", count=2)
    f = pipe.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    cell = {
        "liquid": phase(f, p, 200.0, "liquid", v * 0.2),
        "gas": phase(f, p, 200.0, "vapor", v * 0.8),
    }
    old = pipe.initialize([cell, cell])
    dt = 0.25 * pipe.maximum_step_s
    rates = [{"gas": 1000.0, "liquid": -500.0}] * 2
    baths = [{"gas": 400.0, "liquid": 170.0}] * 2
    new, r = pipe.step(old, dt, heat_rates_w=rates, heat_temperatures_k=baths)
    pipe_audit(pipe, old, new, r)
    assert dict(new.cells[0])["gas"].temperature_k > 200.0
    assert dict(new.cells[0])["liquid"].temperature_k < 200.0
    with pytest.raises(ValueError, match="named phase"):
        pipe.step(old, dt, heat_rates_w=[1000.0] * 2, heat_temperatures_k=[400.0] * 2)


@pytest.mark.parametrize(
    "rates,baths,error",
    [
        ([1.0, 1.0], None, "reservoir"),
        ([{"liquid": 1.0}] * 2, [400.0] * 2, "absent"),
        ([{"typo": 1.0}] * 2, [400.0] * 2, "Unknown"),
        ([1.0], [400.0] * 2, "physical pipe cell"),
    ],
)
def test_invalid_heat_boundaries_reject_atomically(rates, baths, error):
    pipe, old = gas_pipe()
    snapshot = old.to_dict()
    with pytest.raises(ValueError, match=error):
        pipe.step(old, 0.1 * pipe.maximum_step_s, heat_rates_w=rates, heat_temperatures_k=baths)
    assert old.to_dict() == snapshot


@pytest.mark.parametrize("rate,bath", [(1e5, 300.0), (-1e5, 400.0), (1e12, 320.001)])
def test_heat_cannot_flow_uphill_or_overshoot_reservoir(rate, bath):
    pipe, old = gas_pipe()
    with pytest.raises((PropertyModelError, RuntimeError)):
        pipe.step(
            old, 0.25 * pipe.maximum_step_s, heat_rates_w=[rate] * 2, heat_temperatures_k=[bath] * 2
        )


def test_independent_heat_audit_detects_changed_work_and_utility_boundary():
    from check_pr_active_surface import pipe_audit
    from reference_pr_active_heat import check_commands

    pipe, old = gas_pipe()
    new, r = pipe.step(
        old, 0.25 * pipe.maximum_step_s, heat_rates_w=[1e5] * 2, heat_temperatures_k=[500.0] * 2
    )
    bad = deepcopy(r)
    bad["thermal"]["phase_heat_j"][0] *= 2
    with pytest.raises(ValueError, match="audit differs"):
        pipe_audit(pipe, old, new, bad)
    with pytest.raises(ValueError, match="requested boundary"):
        check_commands(r, [2e5] * 2, [500.0] * 2)


def test_heated_active_pipe_couples_equipment_and_exact_reporting_interval():
    from check_pr_equilibrium_euler_plant import audit_step
    from test_surface_pr_active_multifluid_plant import plant

    model, old = plant(1e6)
    duration = 1.25 * model.maximum_step_s
    result = model.advance(
        old,
        duration,
        pipe_heat_w={"line": [100.0] * 2},
        pipe_heat_temperatures_k={"line": [500.0] * 2},
    )
    assert result["state"]["transport"]["time_s"] == duration
    heat = sum(r["pipes"]["line"]["external_heat_j"] for r in result["steps"])
    assert heat == pytest.approx(duration * 200.0, abs=1e-12)
    assert max(map(abs, result["conservation_residuals"])) < 1e-5
    from surface.pr_equilibrium_euler_plant import PREquilibriumEulerPlantState

    state = old
    for r in result["steps"]:
        assert audit_step(
            model,
            state,
            r,
            pipe_heat_w={"line": [100.0] * 2},
            pipe_heat_temperatures_k={"line": [500.0] * 2},
        )["accepted"]
        state = PREquilibriumEulerPlantState.from_dict(r["state"])


def test_standalone_thermal_interval_and_independent_runner_balance():
    import json

    from check_pr_active_multifluid_pipe import audit
    from run_pr_active_multifluid_pipe import run

    pipe, old = gas_pipe()
    duration = 1.25 * pipe.maximum_step_s
    config = {
        "model": pipe.to_dict(),
        "initial": {"state": old.to_dict()},
        "schedule": [
            {
                "duration_s": duration,
                "heat_rates_w": [100.0] * 2,
                "heat_temperatures_k": [500.0] * 2,
            }
        ],
    }
    result = run(config)
    config, result = json.loads(json.dumps([config, result]))
    assert audit(config, result)["accepted"]
    assert max(map(abs, result["conservation_residuals"])) < 1e-7


def test_heat_precedes_thermochemical_phase_birth_and_audits_both_sources():
    from check_pr_active_surface import pipe_audit

    pipe, v = setup(mode="periodic", chemical=0.0, count=2)
    f = pipe.fluid
    p = f.components[1].saturation_at_temperature(250.0)["pressure_pa"] * 0.9
    cell = {"liquid": phase(f, p, 250.0, "liquid", v)}
    old = pipe.initialize([cell, cell])
    new, r = pipe.step(
        old,
        0.1 * pipe.maximum_step_s,
        heat_rates_w=[{"liquid": 100.0}] * 2,
        heat_temperatures_k=[400.0] * 2,
    )
    assert all(len(c) == 2 for c in new.cells)
    pipe_audit(pipe, old, new, r)


def test_finite_aperture_heating_isolation_restart_and_full_audit():
    import json
    from pathlib import Path

    from check_pr_active_surface import audit
    from run_pr_active_surface import run

    from surface.pr_active_surface import PRActiveSurfaceNetwork, PRActiveSurfaceState

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/surface_pr_active_surface_pipe_heat.json"
        ).read_text()
    )
    result = run(config)
    model = PRActiveSurfaceNetwork.from_dict(config["system"])
    old = PRActiveSurfaceState.from_dict(result["initial"])
    heat = 0.0
    for r in result["result"]["steps"]:
        new = PRActiveSurfaceState.from_dict(r["state"])
        audit(model, old, new, r)
        heat += r["pipes"]["line"]["external_heat_j"]
        old = new
    assert heat == pytest.approx(25 * config["schedule"][0]["duration_s"], abs=1e-12)
    assert max(map(abs, result["result"]["conservation_residuals"])) < 1e-7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_active_phase_heat_cuda_matches_cpu():
    cpu, old = gas_pipe()
    gpu, gold = gas_pipe("cuda")
    dt = 0.25 * cpu.maximum_step_s
    a, ra = cpu.step(old, dt, heat_rates_w=[1e5] * 2, heat_temperatures_k=[500.0] * 2)
    b, rb = gpu.step(gold, dt, heat_rates_w=[1e5] * 2, heat_temperatures_k=[500.0] * 2)
    for ca, cb in zip(a.cells, b.cells):
        assert dict(ca)["gas"].temperature_k == pytest.approx(
            dict(cb)["gas"].temperature_k, abs=1e-10
        )
    assert ra["external_heat_j"] == rb["external_heat_j"]
