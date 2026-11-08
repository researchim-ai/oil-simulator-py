"""Spatial HEM phase events with conservative bulk stocks, not phase floors."""

import json
import math

import pytest
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_pr_phase_equilibration import stock

from surface.pr_equilibrium_euler_pipe import PREquilibriumEulerPipe, PREquilibriumEulerPipeState


def liquid_pipe(count=3, mode="periodic", device="cpu"):
    f = fluid(device)
    parcel = stock(f, 2e6, 250.0, "liquid")
    area = math.pi * 0.1**2 / 4
    model = PREquilibriumEulerPipe(f, [parcel.volume_m3 / area] * count, 0.1, boundary_mode=mode)
    specs = [
        {
            "component_moles": parcel.component_moles,
            "total_energy_j": parcel.total_energy_j,
            "momentum_kg_m_s": 0.0,
            "pressure_guess_pa": 2e6,
            "temperature_guess_k": 250.0,
        }
        for _ in range(count)
    ]
    return model, model.initialize(specs)


def cooling_energy(model, old):
    f = model.fluid
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    liquid = f.homogeneous(p, 200.0, [0.0, 1.0], "liquid")
    vapor = f.homogeneous(p, 200.0, [0.0, 1.0], "vapor")
    beta = (model.volumes[0] - float(liquid["molar_volume_m3_mol"].cpu())) / float(
        (vapor["molar_volume_m3_mol"] - liquid["molar_volume_m3_mol"]).cpu()
    )
    return (
        float(
            (
                (1 - beta) * liquid["molar_internal_energy_j_mol"]
                + beta * vapor["molar_internal_energy_j_mol"]
            ).cpu()
        )
        - old.bulk_stocks[0][-1]
    )


def test_uniform_spatial_cooling_birth_and_heating_death_with_exact_bulk_balances():
    model, old = liquid_pipe()
    dt = model.maximum_step_s * 0.5
    heat = cooling_energy(model, old)
    split, r = model.step(old, dt, heat_rates_w=[heat / dt] * 3, heat_temperatures_k=[100.0] * 3)
    assert [len(c.phases) for c in split.cells] == [2] * 3
    assert max(map(abs, r["conservation_residuals"])) < 1e-8
    end, r = model.step(split, dt, heat_rates_w=[-heat / dt] * 3, heat_temperatures_k=[500.0] * 3)
    assert [len(c.phases) for c in end.cells] == [1] * 3
    for a, b in zip(old.bulk_stocks, end.bulk_stocks):
        assert b == pytest.approx(a, abs=1e-8)
    assert end.time_s == 2 * dt


def test_local_phase_birth_propagates_flow_and_component_stocks_between_cells():
    model, old = liquid_pipe()
    dt = model.maximum_step_s * 0.1
    heat = cooling_energy(model, old)
    split, _r = model.step(
        old, dt, heat_rates_w=[heat / dt, 0.0, 0.0], heat_temperatures_k=[100.0, None, None]
    )
    assert [len(c.phases) for c in split.cells] == [2, 1, 1]
    new, _r2 = model.step(split, dt)
    end, _r3 = model.step(new, dt)
    assert any(abs(v[-2]) > 1e-10 for v in end.bulk_stocks)
    assert any(abs(a[1] - b[1]) > 1e-12 for a, b in zip(old.bulk_stocks, end.bulk_stocks))
    assert max(map(abs, model.ledger(old, end, 0.0, heat))) < 1e-8
    assert all(sum(p.component_moles) > 0 for c in end.cells for p in c.phases)


def test_restart_cfl_guard_and_complete_interval():
    model, old = liquid_pipe(mode="walls")
    with pytest.raises(ValueError, match="CFL"):
        model.step(old, 2 * model.maximum_step_s)
    copied = PREquilibriumEulerPipe.from_dict(json.loads(json.dumps(model.to_dict())))
    state = PREquilibriumEulerPipeState.from_dict(json.loads(json.dumps(old.to_dict())))
    duration = 2.5 * model.maximum_step_s
    result = copied.advance(state, duration)
    assert result["state"]["time_s"] == duration
    assert len(result["steps"]) >= 3
    assert max(map(abs, result["conservation_residuals"])) < 1e-8


def test_independent_full_report_audit_and_restart(monkeypatch):
    from check_pr_multifluid_pipe import audit
    from run_pr_multifluid_pipe import run

    from surface.pr_phase_equilibration import PRPhaseEquilibration

    model, state = liquid_pipe()
    dt = model.maximum_step_s * 0.1
    heat = cooling_energy(model, state)
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [
                    {
                        "duration_s": dt,
                        "maximum_step_s": dt,
                        "heat_rates_w": [heat / dt, 0.0, 0.0],
                        "heat_temperatures_k": [100.0, None, None],
                    }
                ],
            }
        )
    )
    report = json.loads(json.dumps(run(config)))
    assert [len(c["phases"]) for c in report["state"]["cells"]] == [2, 1, 1]

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent Euler audit called runtime physics")

    monkeypatch.setattr(PRPhaseEquilibration, "recover", forbidden)
    monkeypatch.setattr(PREquilibriumEulerPipe, "_packets", forbidden)
    assert audit(config, report)["accepted"]
    import copy

    changed = copy.deepcopy(report)
    changed["intervals"][0]["steps"][0]["external_heat_j"] += 1.0
    with pytest.raises(ValueError):
        audit(config, changed)


def test_transport_without_external_heat_can_remove_a_previously_present_phase():
    model, old = liquid_pipe()
    f = model.fluid
    low, high = 200.0, 250.0
    for _ in range(40):
        t = 0.5 * (low + high)
        p = f.components[1].saturation_at_temperature(t)["pressure_pa"]
        q = f.homogeneous(p, t, [0.0, 1.0], "liquid")
        if float(q["molar_volume_m3_mol"].cpu()) > model.volumes[0]:
            high = t
        else:
            low = t
    t = 0.5 * (low + high) - 0.05
    p = f.components[1].saturation_at_temperature(t)["pressure_pa"]
    l = f.homogeneous(p, t, [0.0, 1.0], "liquid")
    g = f.homogeneous(p, t, [0.0, 1.0], "vapor")
    beta = (model.volumes[0] - float(l["molar_volume_m3_mol"].cpu())) / float(
        (g["molar_volume_m3_mol"] - l["molar_volume_m3_mol"]).cpu()
    )
    assert 0 < beta < 1
    energy = float(
        (
            (1 - beta) * l["molar_internal_energy_j_mol"] + beta * g["molar_internal_energy_j_mol"]
        ).cpu()
    )
    dt = model.maximum_step_s
    heat = energy - old.bulk_stocks[0][-1]
    mixed, _ = model.step(
        old, dt, heat_rates_w=[heat / dt, 0.0, 0.0], heat_temperatures_k=[100.0, None, None]
    )
    assert len(mixed.cells[0].phases) == 2
    end, r = model.step(mixed, dt)
    assert r["external_heat_j"] == 0.0
    assert len(end.cells[0].phases) == 1
    assert max(map(abs, r["conservation_residuals"])) < 1e-8


def test_epoch_clock_uses_representable_cfl_steps_and_actual_heat_duration():
    model, old = liquid_pipe()
    raw = old.to_dict()
    epoch = 1e9
    raw["time_s"] = epoch
    for cell in raw["cells"]:
        cell["time_s"] = epoch
    old = PREquilibriumEulerPipeState.from_dict(raw)
    duration = 2.5 * model.maximum_step_s
    result = model.advance(old, duration, heat_rates_w=[1.0] * 3, heat_temperatures_k=[500.0] * 3)
    assert result["state"]["time_s"] == epoch + duration
    assert all(r["dt_s"] <= model.maximum_step_s for r in result["steps"])
    assert math.fsum(r["dt_s"] for r in result["steps"]) == result["state"]["time_s"] - epoch
    heat = 3 * (result["state"]["time_s"] - epoch)
    new = PREquilibriumEulerPipeState.from_dict(result["state"])
    assert max(map(abs, model.ledger(old, new, 0.0, heat))) < 1e-8


def test_exactly_absent_component_arrives_without_a_seed_inventory():
    f = fluid()
    area = math.pi * 0.1**2 / 4
    volume = 0.001
    model = PREquilibriumEulerPipe(f, [volume / area] * 3, 0.1)
    specs = []
    for x in [[1.0, 0.0], [0.8, 0.2], [0.8, 0.2]]:
        q = f.homogeneous(1e6, 350.0, x, "stable")
        amount = volume / float(q["molar_volume_m3_mol"].cpu())
        n = [amount * v for v in x]
        mass = sum(v * c.molar_mass_kg_mol for v, c in zip(n, f.components))
        specs.append(
            {
                "component_moles": n,
                "total_energy_j": amount * float(q["molar_internal_energy_j_mol"].cpu())
                + 0.5 * mass * 0.02**2,
                "momentum_kg_m_s": mass * 0.02,
                "pressure_guess_pa": 1e6,
                "temperature_guess_k": 350.0,
            }
        )
    old = model.initialize(specs)
    assert old.bulk_stocks[0][1] == 0.0
    end, r = model.step(old, 0.1 * model.maximum_step_s)
    assert end.bulk_stocks[0][1] > 0.0
    assert max(map(abs, r["conservation_residuals"])) < 1e-8


def test_last_cell_uv_failure_does_not_publish_partial_transport(monkeypatch):
    from surface.liquid import HydraulicConvergenceError

    model, old = liquid_pipe()
    saved = old.to_dict()
    recover = model.equilibration.recover
    calls = [0]

    def fail_last(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 3:
            raise HydraulicConvergenceError("Injected last-cell UV failure")
        return recover(*args, **kwargs)

    monkeypatch.setattr(model.equilibration, "recover", fail_last)
    dt = model.maximum_step_s * 0.1
    heat = cooling_energy(model, old)
    with pytest.raises(HydraulicConvergenceError, match="last-cell"):
        model.step(old, dt, heat_rates_w=[heat / dt] * 3, heat_temperatures_k=[100.0] * 3)
    assert old.to_dict() == saved
    model.inspect(old)
