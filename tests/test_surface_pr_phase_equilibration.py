"""Actual birth/death of active phases without keeping a residual gas seed."""

import pytest
from test_surface_equilibrium_thermodynamics import fluid

from surface.pr_interphase_relaxation import PRRelaxationPhase
from surface.pr_phase_equilibration import PRPhaseEquilibration, PRPhaseEquilibriumState


def stock(f, p, t, root, n=(0.0, 1.0), velocity=0.0):
    amount = sum(n)
    q = f.homogeneous(p, t, [v / amount for v in n], root)
    mass = sum(v * c.molar_mass_kg_mol for v, c in zip(n, f.components))
    return PRRelaxationPhase(
        n,
        amount * float(q["molar_volume_m3_mol"]),
        p,
        t,
        mass * velocity,
        amount * float(q["molar_internal_energy_j_mol"]) + 0.5 * mass * velocity**2,
    )


def test_cooling_birth_and_heating_death_in_rigid_cell_with_restart():
    f = fluid()
    model = PRPhaseEquilibration(f)
    old = model.initialize([stock(f, 2e6, 250.0, "liquid")], ["liquid"])
    sat = f.components[1].saturation_at_temperature(200.0)
    p = sat["pressure_pa"]
    liquid = f.homogeneous(p, 200.0, [0.0, 1.0], "liquid")
    vapor = f.homogeneous(p, 200.0, [0.0, 1.0], "vapor")
    volume = old.phases[0].volume_m3
    beta = (volume - float(liquid["molar_volume_m3_mol"])) / (
        float(vapor["molar_volume_m3_mol"]) - float(liquid["molar_volume_m3_mol"])
    )
    assert 0 < beta < 1
    target = (1 - beta) * float(liquid["molar_internal_energy_j_mol"]) + beta * float(
        vapor["molar_internal_energy_j_mol"]
    )
    heat = target - old.phases[0].total_energy_j
    split, r = model.project(old, heat_j=heat, heat_temperature_k=100.0)
    from check_pr_phase_equilibration import audit

    assert audit(model.to_dict(), old.to_dict(), split.to_dict(), r)["accepted"]
    assert len(split.phases) == 2
    assert max(map(abs, r["conservation_residuals"])) < 1e-6
    assert all(sum(p.component_moles) > 0 for p in split.phases)
    checkpoint = PRPhaseEquilibriumState.from_dict(split.to_dict())
    restored = PRPhaseEquilibration.from_dict(model.to_dict())
    end, r = restored.project(checkpoint, heat_j=-heat, heat_temperature_k=500.0)
    assert audit(restored.to_dict(), checkpoint.to_dict(), end.to_dict(), r)["accepted"]
    assert len(end.phases) == 1
    assert end.phases[0].temperature_k == pytest.approx(250.0, abs=1e-5)
    assert end.phases[0].pressure_pa == pytest.approx(2e6, rel=1e-6)


def test_counterflow_kinetic_energy_is_thermalized_not_deleted():
    f = fluid()
    model = PRPhaseEquilibration(f)
    phases = [
        stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), 10.0),
        stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), -10.0),
    ]
    old = model.initialize(phases, ["vapor", "vapor"])
    end, r = model.project(old)
    assert abs(sum(p.momentum_kg_m_s for p in end.phases)) < 1e-10
    assert sum(p.total_energy_j for p in end.phases) == pytest.approx(
        sum(p.total_energy_j for p in old.phases), abs=1e-6
    )
    assert end.phases[0].temperature_k > 320.0
    assert r["entropy_production_j_k"] > 0


def test_independent_phase_audit_and_no_runtime_scipy(monkeypatch):
    import copy

    import scipy.optimize
    from check_pr_phase_equilibration import audit

    f = fluid()
    model = PRPhaseEquilibration(f)
    old = model.initialize(
        [
            stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), 10.0),
            stock(f, 1e6, 320.0, "vapor", (0.5, 0.5), -10.0),
        ],
        ["vapor", "vapor"],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime must not use the SciPy reference")

    for name in ["root", "brentq", "minimize", "least_squares"]:
        monkeypatch.setattr(scipy.optimize, name, forbidden)
    end, r = model.project(old)
    monkeypatch.undo()
    monkeypatch.setattr(f, "flash_uv", forbidden)
    assert audit(model.to_dict(), old.to_dict(), end.to_dict(), r)["accepted"]
    changed = copy.deepcopy(r)
    changed["heat_j"] = 1.0
    with pytest.raises(ValueError):
        audit(model.to_dict(), old.to_dict(), end.to_dict(), changed)


def test_three_liquid_phases_remain_real_active_stocks():
    from test_surface_equilibrium_thermodynamics import symmetric

    f = symmetric()
    q = f.flash_pt(1e6, 300.0, [1 / 3] * 3)
    assert q["phase_count"] == 3
    phases = []
    for phase in q["phases"]:
        n = tuple(float(phase["molar_fraction"]) * float(x) for x in phase["mole_fractions"])
        phases.append(stock(f, 1e6, 300.0, "liquid", n))
    model = PRPhaseEquilibration(f)
    old = model.initialize(phases, ["liquid"] * 3)
    end, r = model.project(old)
    assert len(end.phases) == 3
    assert [max(range(3), key=p.component_moles.__getitem__) for p in end.phases] == [0, 1, 2]
    from check_pr_phase_equilibration import audit

    assert audit(model.to_dict(), old.to_dict(), end.to_dict(), r)["accepted"]


def test_heat_boundary_is_required_and_failure_leaves_state_unchanged():
    model = PRPhaseEquilibration(fluid())
    old = model.initialize([stock(model.fluid, 1e6, 320.0, "vapor", (0.5, 0.5))], ["vapor"])
    saved = old.to_dict()
    with pytest.raises(ValueError, match="reservoir temperature"):
        model.project(old, heat_j=1.0)
    assert old.to_dict() == saved


@pytest.mark.parametrize("beta", [1e-6, 0.5, 1.0 - 1e-6])
def test_direct_pure_uv_near_latent_plateau_has_no_nested_ph_accuracy_floor(beta):
    f = fluid()
    saturation = f.components[1].saturation_at_temperature(200.0)
    p = saturation["pressure_pa"]
    l = f.homogeneous(p, 200.0, [0.0, 1.0], "liquid")
    g = f.homogeneous(p, 200.0, [0.0, 1.0], "vapor")
    energy = float(
        (1 - beta) * l["molar_internal_energy_j_mol"] + beta * g["molar_internal_energy_j_mol"]
    )
    volume = float((1 - beta) * l["molar_volume_m3_mol"] + beta * g["molar_volume_m3_mol"])
    q = f.flash_uv([0.0, 1.0], energy, volume, pressure_guess_pa=2e6, temperature_guess_k=250.0)
    assert q["phase_count"] == 2
    assert float(q["temperature_k"]) == pytest.approx(200.0, abs=1e-7)
    assert float(q["phases"][1]["molar_fraction"]) == pytest.approx(beta, abs=1e-9)
    assert abs(float(q["volume_residual_m3"])) < 1e-12
    assert abs(float(q["energy_residual_j"])) < 1e-6


@pytest.mark.parametrize("root", ["liquid", "vapor"])
def test_exact_saturation_endpoints_do_not_keep_machine_roundoff_phase(root):
    f = fluid()
    p = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    q = f.homogeneous(p, 200.0, [0.0, 1.0], root)
    result = f.flash_uv(
        [0.0, 1.0],
        float(q["molar_internal_energy_j_mol"]),
        float(q["molar_volume_m3_mol"]),
        pressure_guess_pa=2e6,
        temperature_guess_k=250.0,
    )
    assert result["phase_count"] == 1
    assert float(result["phases"][0]["molar_fraction"]) == 1.0
