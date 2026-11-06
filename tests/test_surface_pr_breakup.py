"""Powered fragmentation: count kinetics, interface work and common PR ledger."""

import json
from dataclasses import replace

import pytest
from test_surface_pr_droplet_population import closed

from surface.pr_droplet_population import PRPoweredDropletPopulation
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork


def setup(rate=0.5, power=1.0, kernel=0.0):
    base, _, initial = closed(kernel)
    spec = PRPoweredDropletPopulation(
        "oil_drops", kernel, breakup_rate_s=rate, agitation_power_w=power
    )
    node = replace(base.nodes[0], droplet_population=spec)
    model = PREquilibriumTransportNetwork(base.fluid, [node], initial_step_s=0.001)
    return model, model.initialize({"emulsion": initial}), initial


@pytest.fixture(autouse=True)
def script_path(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))


def test_powered_breakup_changes_count_and_size_with_exact_electromechanical_energy():
    m, old, _ = setup()
    new, r = m._step(old, 0.01, m.default_openings, [0.0], [350.0])
    n0 = old.nodes[0][1].droplet_number
    assert new.nodes[0][1].droplet_number == pytest.approx(n0 / (1 - 0.005), rel=1e-10)
    assert new.nodes[0][1].total_energy_j - old.nodes[0][1].total_energy_j == pytest.approx(
        0.01, abs=1e-8
    )
    pop = r["droplet_populations"]["emulsion"]
    prior = m._state_data(0, old.nodes[0][1])["population"]
    assert pop["diameter_m"] < float(prior["diameter_m"])
    assert pop["surface_energy_j"] > float(prior["surface_energy_j"])
    assert 0 < pop["breakup_surface_power_w"] < 1.0
    assert r["external_work_j"] == 0.01 and r["external_heat_j"] == 0.0
    assert r["agitation_work_j"] == {"emulsion": 0.01}
    assert max(map(abs, r["conservation_residuals"])) < 1e-7


def test_underpowered_breakup_is_rejected_without_altering_stocks():
    m, old, _ = setup(rate=10.0, power=1e-12)
    before = old.to_dict()
    with pytest.raises(ValueError, match="mechanical"):
        m.advance(old, 0.001)
    assert old.to_dict() == before
    with pytest.raises(ValueError):
        setup(power=0.0)


def test_independent_breakup_and_work_audit_and_restart():
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    m, _, initial = setup()
    cfg = json.loads(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": {"nodes": {"emulsion": initial}},
                "schedule": [{"duration_s": 0.001}],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    restart = {**cfg, "initial": {"state": result["intervals"][-1]["state"]}}
    assert audit(restart, json.loads(json.dumps(run(restart))))["accepted"]


def test_coalescence_and_breakup_share_the_same_number_balance():
    base, start, _ = setup()
    n0 = start.nodes[0][1].droplet_number
    coefficient = 1 / n0
    kernel = 2 * base.nodes[0].volume_m3 * coefficient
    m, old, _ = setup(rate=0.5, power=1.0, kernel=kernel)
    new, _ = m._step(old, 0.02, m.default_openings, [0.0], [350.0])
    b = 1 - 0.5 * 0.02
    expected = 2 * n0 / (b + (b * b + 4 * 0.02 * coefficient * n0) ** 0.5)
    assert new.nodes[0][1].droplet_number == pytest.approx(expected, rel=1e-10)


def test_fragmentation_count_converges_to_continuous_kinetics():
    import math

    m, start, _ = setup()
    exact = start.nodes[0][1].droplet_number * math.exp(0.5 * 0.02)
    errors = []
    for count in (2, 4):
        state = start
        for _ in range(count):
            state, _ = m._step(state, 0.02 / count, m.default_openings, [0.0], [350.0])
        errors.append(abs(state.nodes[0][1].droplet_number - exact))
    assert errors[1] < 0.51 * errors[0]
