"""Published coupled states must come from the same profile/plant Newton."""

import json
from pathlib import Path

import pytest
from test_surface_pr_layered_transport import model as profile_model

from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork
from surface.pr_layered_plant import LayeredPlantState, PRLayeredPlantNetwork


def setup(reverse=False, dense_limit=768):
    native, initial = profile_model(cells=1)
    profile = native.initialize_equilibrium(**initial)
    center = native._disk_bottom(0, -1) + native.diameter * native.cosine / 2
    transport = PREquilibriumTransportNetwork(
        native.fluid,
        [PREquilibriumNode("junction", 0.001, elevation_m=center)],
        gravity_m_s2=native.gravity,
        wave_speed_bound_m_s=native.alpha,
        dense_limit=dense_limit,
        initial_step_s=1e-6,
        maximum_step_s=1e-4,
    )
    ordinary = transport.initialize(
        {
            "junction": {
                "pressure_pa": 0.995e6 if reverse else 1.005e6,
                "temperature_k": 196.2,
                "mole_fractions": [0.19, 0.81],
            }
        }
    )
    plant = PRLayeredPlantNetwork(transport, native, ("junction", None))
    return plant, plant.initialize(ordinary, profile)


@pytest.mark.parametrize("reverse", [False, True])
def test_public_common_step_publishes_conservative_state_and_support_reaction(reverse):
    m, s = setup(reverse)
    old = s.to_dict()
    end, report = m.step(s, 1e-6)
    assert s.to_dict() == old
    assert (end.layered.cells[0].momentum_kg_m_s < 0) == reverse
    assert max(map(abs, report["conservation_residuals"])) < 1e-5
    assert abs(report["vessel_support_force_n"][0]) > 1.0
    assert report["entropy_production_j_k"] >= -1e-6
    restored = PRLayeredPlantNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(LayeredPlantState.from_dict(json.loads(json.dumps(end.to_dict()))))
    _again, r = restored.step(end, 1e-6)
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_common_adaptive_interval_and_checkpoint_restore():
    m, s = setup()
    report = m.advance(s, 1e-6, maximum_attempts=1)
    assert report["attempts"] == 1
    end = LayeredPlantState.from_dict(report["state"])
    assert end.time_s == 1e-6 and end.layered.time_s == end.time_s
    assert max(map(abs, report["conservation_residuals"])) < 1e-5
    assert sum(r["dt_s"] for r in report["steps"]) == pytest.approx(1e-6, abs=1e-20)


def test_common_sparse_solve_matches_dense():
    a, s = setup()
    b, t = setup(dense_limit=1)
    x, rx = a.step(s, 1e-6)
    y, ry = b.step(t, 1e-6)
    assert x.layered.cells[0].momentum_kg_m_s == pytest.approx(
        y.layered.cells[0].momentum_kg_m_s, rel=1e-5, abs=1e-10
    )
    assert max(map(abs, rx["conservation_residuals"])) < 1e-5
    assert max(map(abs, ry["conservation_residuals"])) < 1e-5


@pytest.fixture(scope="module")
def coupled_fixture():
    from run_pr_layered_plant import run

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/surface_pr_layered_plant.json").read_text())
    return cfg, json.loads(json.dumps(run(cfg)))


def test_independent_combined_audit_and_tamper_rejection(coupled_fixture):
    from check_pr_layered_plant import audit

    cfg, report = coupled_fixture
    assert audit(cfg, report)["accepted"]
    damaged = json.loads(json.dumps(report))
    damaged["intervals"][0]["steps"][0]["vessel_support_force_n"][0] = 0.0
    with pytest.raises(ValueError):
        audit(cfg, damaged)
    damaged = json.loads(json.dumps(report))
    damaged["intervals"][0]["steps"][0]["state"]["layered"]["cells"][0]["total_energy_j"] += 1.0
    with pytest.raises(ValueError):
        audit(cfg, damaged)


def test_full_runner_restart_retains_both_component_stocks(coupled_fixture):
    from check_pr_layered_plant import audit
    from run_pr_layered_plant import run

    cfg, report = coupled_fixture
    request = {**cfg, "initial": {"state": report["state"]}, "schedule": [{"duration_s": 1e-7}]}
    continued = json.loads(json.dumps(run(request)))
    assert audit(request, continued)["accepted"]
    assert continued["initial"] == report["state"]


@pytest.mark.parametrize("part", ["transport", "layered_equilibrium"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("pressure_pa", 1.1e6),
        ("temperature_k", 200.0),
        ("mole_fractions", [0.2, 0.8]),
    ],
)
def test_independent_audit_binds_requested_initialization(coupled_fixture, part, field, value):
    from check_pr_layered_plant import audit

    cfg, report = coupled_fixture
    changed = json.loads(json.dumps(cfg))
    spec = changed["initial"][part]
    if part == "transport":
        spec = spec["junction"]
    spec[field] = value
    damaged = json.loads(json.dumps(report))
    damaged["config"] = changed
    with pytest.raises(ValueError):
        audit(changed, damaged)


def test_initial_audit_uses_scalar_references_without_runtime_initializers(
    coupled_fixture, monkeypatch
):
    from check_pr_layered_plant import audit

    from surface.equilibrium_properties import EquilibriumFluid
    from surface.pr_layered_transport import PRLayeredTransportNetwork
    from surface.stratified_pr import StratifiedPRColumn

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime initialization/equilibrium called by independent audit")

    monkeypatch.setattr(PREquilibriumTransportNetwork, "initialize", forbidden)
    monkeypatch.setattr(PRLayeredTransportNetwork, "initialize_equilibrium", forbidden)
    monkeypatch.setattr(StratifiedPRColumn, "initialize", forbidden)
    monkeypatch.setattr(StratifiedPRColumn, "equilibrate", forbidden)
    monkeypatch.setattr(EquilibriumFluid, "flash_pt", forbidden)
    cfg, report = coupled_fixture
    assert audit(cfg, report)["accepted"]


def test_common_attempt_budget_still_rejects_failed_attempt(coupled_fixture, monkeypatch):
    from surface.liquid import HydraulicConvergenceError

    cfg, report = coupled_fixture
    model = PRLayeredPlantNetwork.from_dict(cfg["model"])
    state = LayeredPlantState.from_dict(report["initial"])
    calls = []

    def rejected(*args, **kwargs):
        calls.append(1)
        raise HydraulicConvergenceError("Forced rejected trial")

    monkeypatch.setattr(model, "step", rejected)
    with pytest.raises(HydraulicConvergenceError, match="interval budget exhausted"):
        model.advance(state, 1e-6, maximum_attempts=1)
    assert len(calls) == 1
