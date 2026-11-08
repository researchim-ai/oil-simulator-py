"""Conservative finite-rate mass exchange and actual phase birth, not floors."""

import math

import pytest
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_pr_phase_equilibration import stock

from surface.pr_homogeneous_uv import caloric_tv, recover_homogeneous_phases
from surface.pr_thermochemical_relaxation import PRThermochemicalRelaxation


@pytest.mark.parametrize(
    "root,p,t,n",
    [
        ("liquid", 2e6, 250.0, (0.0, 1.0)),
        ("vapor", 1e6, 320.0, (0.5, 0.5)),
        ("liquid", 1e6, 196.2, (0.0, 1.0)),
        ("vapor", 1e5, 200.0, (1.0, 0.0)),
    ],
)
def test_homogeneous_uv_retains_declared_branch_and_canonical_stocks(root, p, t, n):
    f = fluid()
    old = stock(f, p, t, root, n, velocity=7.0)
    (new,) = recover_homogeneous_phases(
        f,
        [old.component_moles],
        [old.volume_m3],
        [old.momentum_kg_m_s],
        [old.total_energy_j],
        [root],
        [t + 10.0],
    )
    assert new.pressure_pa == pytest.approx(p, rel=1e-8)
    assert new.temperature_k == pytest.approx(t, abs=1e-8)
    assert new.component_moles == old.component_moles
    assert new.total_energy_j == old.total_energy_j
    assert new.momentum_kg_m_s == old.momentum_kg_m_s
    q = caloric_tv(f, f.tensor(t), f.tensor(old.volume_m3 / sum(n)), f.tensor(n) / sum(n))
    assert float(q["p"]) == pytest.approx(p, rel=1e-10)


def test_finite_source_preserves_independent_velocities_and_exact_exponential_decay():
    f = fluid()
    model = PRThermochemicalRelaxation(f, 0.2)
    phases = [
        stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), 10.0),
        stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), -10.0),
    ]
    old = model.initialize(phases, ["vapor", "vapor"])
    end, r = model.step(old, 0.05)
    assert len(end.phases) == 2
    for before, after in zip(old.phases, end.phases):
        assert after.component_moles == pytest.approx(before.component_moles, abs=1e-12)
        assert after.momentum_kg_m_s == pytest.approx(
            before.momentum_kg_m_s * math.exp(-0.25), abs=1e-12
        )
        assert after.temperature_k > before.temperature_k
    assert r["entropy_production_j_k"] > 0.0
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_metastable_liquid_evaporates_at_finite_rate_without_forcing_equal_pt():
    f = fluid()
    p = f.components[1].saturation_at_temperature(250.0)["pressure_pa"]
    phase = stock(f, 0.9 * p, 250.0, "liquid")
    model = PRThermochemicalRelaxation(f, 0.1)
    old = model.initialize([phase], ["liquid"])
    end, r = model.step(old, 0.02)
    assert len(old.phases) == 1 and len(end.phases) == 2
    assert all(sum(q.component_moles) > 0.0 for q in end.phases)
    assert abs(end.phases[0].pressure_pa - end.phases[1].pressure_pa) > 1.0
    assert abs(end.phases[0].temperature_k - end.phases[1].temperature_k) > 1e-5
    assert r["entropy_production_j_k"] >= r["entropy_lower_bound_j_k"] - 1e-7
    assert max(map(abs, r["conservation_residuals"])) < 1e-6


def test_instantaneous_limit_removes_an_absent_phase_without_a_seed():
    f = fluid()
    model = PRThermochemicalRelaxation(f, 0.0)
    old = model.initialize(
        [
            stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), 10.0),
            stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), -10.0),
        ],
        ["vapor", "vapor"],
    )
    end, r = model.step(old, 0.01)
    assert len(end.phases) == 1
    assert sum(end.phases[0].component_moles) == pytest.approx(2.0)
    assert r["entropy_production_j_k"] > 0.0


def test_condensing_gas_decays_without_a_floor_and_instantaneous_limit_removes_it():
    f = fluid()
    phases = [stock(f, 2e6, 180.0, "liquid"), stock(f, 1e5, 220.0, "vapor", (0.0, 1e-7))]
    model = PRThermochemicalRelaxation(f, 0.1)
    old = model.initialize(phases, ["liquid", "vapor"])
    end, r = model.step(old, 0.03)
    assert r["target_families"] == ("liquid",)
    assert len(end.phases) == 2
    assert end.phases[1].component_moles[1] == pytest.approx(1e-7 * math.exp(-0.3), rel=1e-10)
    assert end.phases[1].temperature_k == pytest.approx(220.0, abs=1e-8)
    instant = PRThermochemicalRelaxation(f, 0.0)
    gone, report = instant.step(instant.initialize(phases, ["liquid", "vapor"]), 0.03)
    assert len(gone.phases) == 1
    assert report["entropy_production_j_k"] >= 0.0


def test_component_exchange_changes_compositions_without_losing_species():
    f = fluid()
    model = PRThermochemicalRelaxation(f, 0.1)
    phases = [
        stock(f, 1e6, 250.0, "liquid", (0.01, 0.99)),
        stock(f, 1e6, 250.0, "vapor", (0.045, 0.005)),
    ]
    old = model.initialize(phases, ["liquid", "vapor"])
    end, r = model.step(old, 0.005)
    assert len(end.phases) == 2
    assert any(
        abs(a - b) > 1e-8
        for a, b in zip(old.phases[0].component_moles, end.phases[0].component_moles)
    )
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    assert r["entropy_production_j_k"] > 0.0


def test_duplicate_populations_do_not_depend_on_input_phase_order():
    f = fluid()
    model = PRThermochemicalRelaxation(f, 0.2)
    phases = [
        stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), 10.0),
        stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), -10.0),
    ]
    a, _ = model.step(model.initialize(phases, ["vapor", "vapor"]), 0.05)
    b, _ = model.step(model.initialize(phases[::-1], ["vapor", "vapor"]), 0.05)
    for aa, bb in zip(a.phases, reversed(b.phases)):
        assert aa.momentum_kg_m_s == pytest.approx(bb.momentum_kg_m_s, abs=1e-12)
        assert aa.total_energy_j == pytest.approx(bb.total_energy_j, abs=1e-8)


def test_independent_scalar_source_audit_and_corruption_detection(monkeypatch):
    import copy

    from check_pr_thermochemical_relaxation import audit

    f = fluid()
    model = PRThermochemicalRelaxation(f, 0.1)
    p = f.components[1].saturation_at_temperature(250.0)["pressure_pa"]
    old = model.initialize([stock(f, 0.9 * p, 250.0, "liquid")], ["liquid"])
    new, report = model.step(old, 0.02)

    def forbidden(*args, **kwargs):
        raise AssertionError("Audit called runtime physics")

    monkeypatch.setattr(f, "flash_uv", forbidden)
    monkeypatch.setattr(PRThermochemicalRelaxation, "step", forbidden)
    assert audit(model.to_dict(), old.to_dict(), new.to_dict(), report)["accepted"]
    changed = copy.deepcopy(report)
    changed["old_weight"] += 0.01
    with pytest.raises(ValueError):
        audit(model.to_dict(), old.to_dict(), new.to_dict(), changed)


def test_target_cache_is_used_only_for_the_accepted_closed_source_state(monkeypatch):
    f = fluid()
    model = PRThermochemicalRelaxation(f, 0.1)
    old = model.initialize(
        [
            stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), 10.0),
            stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), -10.0),
        ],
        ["vapor", "vapor"],
    )
    first, _ = model.step(old, 0.01)

    def forbidden(*args, **kwargs):
        raise AssertionError("Repeated equilibrium flash")

    monkeypatch.setattr(f, "flash_uv", forbidden)
    end, _ = model.step(first, 0.01)
    assert end.time_s == pytest.approx(0.02)
    copied = model.initialize(first.phases, first.phase_roots, time_s=first.time_s)
    with pytest.raises(AssertionError, match="Repeated equilibrium flash"):
        model.step(copied, 0.01)


def test_source_subdivision_restart_and_exact_report_interval():
    import json

    from surface.pr_phase_equilibration import PRPhaseEquilibriumState

    f = fluid()
    model = PRThermochemicalRelaxation(f, 0.1)
    p = f.components[1].saturation_at_temperature(250.0)["pressure_pa"]
    old = model.initialize([stock(f, 0.9 * p, 250.0, "liquid")], ["liquid"])
    full, _ = model.step(old, 0.02)
    half, _ = model.step(old, 0.01)
    split, _ = model.step(half, 0.01)
    for a, b in zip(full.phases, split.phases):
        assert a.component_moles == pytest.approx(b.component_moles, abs=1e-11)
        assert a.total_energy_j == pytest.approx(b.total_energy_j, abs=1e-8)
        assert a.volume_m3 == pytest.approx(b.volume_m3, abs=1e-14)
    restored = PRThermochemicalRelaxation.from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PRPhaseEquilibriumState.from_dict(json.loads(json.dumps(half.to_dict())))
    end, r = restored.step(checkpoint, 0.01)
    assert end.time_s == split.time_s
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    interval = restored.advance(checkpoint, 0.025, maximum_step_s=0.01)
    assert interval["state"]["time_s"] == checkpoint.time_s + 0.025
    assert len(interval["steps"]) == 3


def test_phase_budget_and_unresolved_time_refuse_atomically():
    f = fluid()
    model = PRThermochemicalRelaxation(f, 0.1, maximum_active_phases=1)
    p = f.components[1].saturation_at_temperature(250.0)["pressure_pa"]
    old = model.initialize([stock(f, 0.9 * p, 250.0, "liquid")], ["liquid"])
    saved = old.to_dict()
    with pytest.raises(ValueError, match="phase budget"):
        model.advance(old, 0.01)
    assert old.to_dict() == saved
    from dataclasses import replace

    old = replace(old, time_s=1e20)
    with pytest.raises(ValueError, match="clock resolution"):
        model.step(old, 1e-10)


def test_large_inventory_uses_roundoff_budget_without_relaxing_the_eos():
    from dataclasses import replace

    from check_pr_thermochemical_relaxation import audit

    f = fluid()
    scale = 1e7
    original = [stock(f, 2e6, 180.0, "liquid"), stock(f, 1e5, 220.0, "vapor", (0.0, 1e-7))]
    phases = [
        replace(
            p,
            component_moles=tuple(scale * n for n in p.component_moles),
            volume_m3=scale * p.volume_m3,
            momentum_kg_m_s=scale * p.momentum_kg_m_s,
            total_energy_j=scale * p.total_energy_j,
        )
        for p in original
    ]
    model = PRThermochemicalRelaxation(f, 0.1)
    old = model.initialize(phases, ["liquid", "vapor"])
    new, report = model.step(old, 0.03)
    assert all(
        abs(r) <= tol
        for r, tol in zip(report["conservation_residuals"], report["conservation_tolerances"])
    )
    assert audit(model.to_dict(), old.to_dict(), new.to_dict(), report)["accepted"]
