"""Stage material boundaries and stiffly accurate PR Radau integration."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from audit_pr_collocation import audit_gauss_step
from check_pr_collocation_walls import WallReference, wall_fixture
from check_pr_open_transport import audit
from scipy.integrate import solve_ivp
from test_surface_pr_open_transport import build

from surface import PRCollocation, PREquilibriumTransportNetwork, PREquilibriumTransportState
from surface.component_energy import ComponentEnergyStream
from surface.liquid import HydraulicConvergenceError
from surface.pr_open_transport import PROpenTransportNetwork, PRTransportFeed


@pytest.mark.parametrize("scheme", ["gauss4", "radau3"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("temperature", [320.0, 196.2])
def test_actual_open_pipe_phase_composition_energy_and_reverse(scheme, reverse, temperature):
    model, state = build(reverse=reverse, temperature=temperature)
    before = state.to_dict()
    result = model.advance(state, 1e-5, integrator=scheme)
    assert audit(model.to_dict(), state.to_dict(), result)["accepted"]
    assert result["fallback_intervals"] == 0
    assert all(step["method"] == scheme for step in result["steps"])
    assert max(map(abs, result["conservation_residuals"])) < 1e-6
    assert state.to_dict() == before
    assert all(len(step["stage_couplings"]) == 2 for step in result["steps"])
    if reverse:
        assert result["steps"][0]["stage_couplings"][0]["ports"]["boundary"]["volume_rate_m3_s"] < 0


@pytest.mark.parametrize("scheme", ["gauss4", "radau3"])
def test_closed_external_port_and_dense_sparse_agree(scheme):
    model, state = build()
    sparse = PROpenTransportNetwork.from_dict(
        {**model.to_dict(), "transport": {**model.transport.to_dict(), "dense_limit": 1}}
    )
    restart = replace(state, model_id=sparse.transport.model_id)
    a = model.advance(state, 1e-5, integrator=scheme, port_openings={"boundary": 0.0})
    b = sparse.advance(restart, 1e-5, integrator=scheme, port_openings={"boundary": 0.0})
    assert audit(sparse.to_dict(), restart.to_dict(), b, port_openings={"boundary": 0.0})[
        "accepted"
    ]
    for name, cell in a["state"]["nodes"].items():
        assert cell["pressure_pa"] == pytest.approx(
            b["state"]["nodes"][name]["pressure_pa"], abs=1e-5
        )
    assert a["boundary_component_moles"] == pytest.approx([1e-7, 2e-7], abs=1e-12)


def test_new_species_support_is_seeded_without_adding_artificial_stock():
    model, state = build()
    physical = model.transport
    state = physical.initialize(
        {
            node.name: {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [1.0, 0.0]}
            for node in physical.nodes
        }
    )
    stream = ComponentEnergyStream.from_pt(
        model.fluid, [0.0, 0.01], 1.01e6, 321.0, gravity_m_s2=physical.gravity_m_s2
    )
    model = PROpenTransportNetwork(physical, [PRTransportFeed("fresh", "inlet", stream)], [])
    result = model.advance(state, 1e-5, integrator="radau3")
    final = PREquilibriumTransportState.from_dict(result["state"])
    assert sum(s.component_moles[1] for _, s in final.nodes) == pytest.approx(1e-7, abs=1e-11)
    assert audit(model.to_dict(), state.to_dict(), result)["accepted"]


@pytest.mark.parametrize("corrupt", ["stage_flux", "endpoint", "integral", "identity"])
def test_auditor_rejects_changed_external_packet(corrupt, monkeypatch):
    model, state = build()
    result = model.advance(state, 1e-5, integrator="radau3")
    bad = copy.deepcopy(result)
    row = bad["steps"][0]
    if corrupt == "stage_flux":
        row["stage_couplings"][0]["ports"]["feed"]["component_rates_mol_s"][0] += 0.1
    elif corrupt == "endpoint":
        row["coupling"]["ports"]["boundary"]["total_energy_rate_w"] += 1.0
    elif corrupt == "integral":
        row["exchange_component_moles"][0] += 0.01
    else:
        row["stage_couplings"][0]["model_id"] = "other"

    def forbidden(*args, **kwargs):
        raise AssertionError("Auditor used production flash/flow")

    monkeypatch.setattr(PREquilibriumTransportNetwork, "_fluxes", forbidden)
    monkeypatch.setattr(type(model.fluid), "flash_pt", forbidden)
    monkeypatch.setattr(type(model.fluid), "flash_uv", forbidden)
    assert audit(model.to_dict(), state.to_dict(), result)["accepted"]
    with pytest.raises(ValueError):
        audit(model.to_dict(), state.to_dict(), bad)


def test_radau_third_order_against_independent_pr_wall_ode():
    model, state = wall_fixture()
    ref = WallReference(model, state)
    expected = solve_ivp(
        ref.rhs,
        (0.0, 0.5),
        [320.0, 360.0, 350.0, 300.0, 0.0],
        method="DOP853",
        rtol=1e-13,
        atol=1e-12,
    ).y[:4, -1]
    errors = []
    for dt in (0.125, 0.0625, 0.03125):
        solver = PRCollocation(model, scheme="radau3", phase_fallback=False)
        current = state
        for _ in range(round(0.5 / dt)):
            old = current
            current, row = solver._step(
                current, dt, model.default_openings, [0.0, 0.0], [320.0, 360.0], {}
            )
            assert row["nonlinear_solves"] == 1
            audit_gauss_step(
                model.to_dict(),
                old.to_dict(),
                row,
                model.fluid,
                {"openings": model.default_openings},
            )
        actual = [s.temperature_k for _, s in current.nodes] + [
            s.wall_temperature_k for _, s in current.nodes
        ]
        errors.append(max(abs(np.array(actual) - expected)))
    assert min(errors[i] / errors[i + 1] for i in (0, 1)) > 7


def test_radau_l_stability_and_exact_last_stage_endpoint():
    original, state = wall_fixture()
    nodes = [
        replace(
            n,
            wall=replace(
                n.wall,
                heat_capacity_j_k=1.0,
                fluid_ua_w_k=0.0,
                ambient_ua_w_k=1000.0,
                ambient_temperature_k=300.0,
                max_temperature_change_k=100.0,
            ),
        )
        for n in original.nodes
    ]
    model = PREquilibriumTransportNetwork(
        original.fluid, nodes, initial_step_s=0.02, maximum_step_s=0.25
    )
    state = replace(state, model_id=model.model_id)
    for scheme in ("gauss4", "radau3"):
        final, row = PRCollocation(model, scheme=scheme, phase_fallback=False)._step(
            state, 0.1, {}, [0.0, 0.0], [320.0, 360.0], {}
        )
        ratio = abs(final.nodes[0][1].wall_temperature_k - 300.0) / 50.0
        if scheme == "radau3":
            assert ratio < 0.03
            for name, cell in final.nodes:
                assert (
                    cell.wall_temperature_k == row["stage_states"][-1][name]["wall_temperature_k"]
                )
                assert cell.pressure_pa == row["stage_states"][-1][name]["pressure_pa"]
        else:
            assert ratio > 0.8


def test_open_fallback_preserves_external_ledger(monkeypatch):
    model, state = build()

    def fail(*args, **kwargs):
        raise HydraulicConvergenceError("forced stage failure")

    monkeypatch.setattr(PRCollocation, "_step", fail)
    result = model.advance(state, 1e-5, integrator="radau3")
    assert result["fallback_intervals"] > 0
    assert audit(model.to_dict(), state.to_dict(), result)["accepted"]


def test_external_stage_jacobian_includes_pressure_port_derivative(monkeypatch):
    model, state = build()
    original = PRCollocation._solve
    seen = []

    def check(self, fn, initial, *args):
        direction = torch.linspace(-0.01, 0.01, len(initial), dtype=initial.dtype)
        j = torch.func.jacfwd(fn)(initial)
        fd = (fn(initial + 1e-5 * direction) - fn(initial - 1e-5 * direction)) / (2e-5)
        assert float((j @ direction - fd).abs().max()) < 1e-6
        seen.append(1)
        return original(self, fn, initial, *args)

    monkeypatch.setattr(PRCollocation, "_solve", check)
    model.advance(state, 1e-5, integrator="radau3")
    assert len(seen) == 3  # coarse and both accepted half steps; no projection solves


def test_reverse_port_crosses_zero_in_time_with_correct_donor():
    from test_surface_compositional_separator import binary

    from surface import PREquilibriumNode
    from surface.pr_open_transport import PRTransportPressurePort

    fluid = binary()
    t = PREquilibriumTransportNetwork(
        fluid, [PREquilibriumNode("v", 0.001)], initial_step_s=1e-6, maximum_step_s=1e-4
    )
    state = t.initialize(
        {"v": {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [1.0, 0.0]}}
    )
    stream = ComponentEnergyStream.from_pt(fluid, [0.01, 0.0], 1.01e6, 320.0, gravity_m_s2=0.0)
    model = PROpenTransportNetwork(
        t,
        [PRTransportFeed("feed", "v", stream)],
        [
            PRTransportPressurePort(
                "port", "v", 1e6 + 0.1, 320.0, (1.0, 0.0), 1e-6, allow_reverse=True
            )
        ],
    )
    first = model.advance(state, 1e-6, integrator="radau3")
    assert audit(model.to_dict(), state.to_dict(), first)["accepted"]
    restarted = PREquilibriumTransportState.from_dict(first["state"])
    second = model.advance(restarted, 1e-4, integrator="radau3")
    assert audit(model.to_dict(), restarted.to_dict(), second)["accepted"]
    assert first["steps"][0]["stage_couplings"][0]["ports"]["port"]["volume_rate_m3_s"] < 0
    assert second["steps"][-1]["stage_couplings"][-1]["ports"]["port"]["volume_rate_m3_s"] > 0


def test_real_open_phase_cycle_retains_feed_and_wall_stocks():
    import json

    from check_pr_collocation import ROOT
    from run_pr_open_transport import run
    from test_surface_compositional_separator import binary

    config = json.loads((ROOT / "configs/surface_pr_collocation_wall_phase_cycle.json").read_text())
    fluid = binary()
    stream = ComponentEnergyStream.from_pt(fluid, [0.0, 1e-4], 1e6, 320.0, gravity_m_s2=0.0)
    physical = config["model"]
    config["model"] = {
        "model": "pr_open_transport_v1",
        "transport": physical,
        "feeds": [PRTransportFeed("feed", "cell", stream).to_dict()],
        "pressure_ports": [],
    }
    result = run(config, integrator="radau3")
    initial = result["initial"]
    for command, report in zip(config["schedule"], result["intervals"]):
        assert audit(
            result["model"],
            initial,
            report,
            **{k: v for k, v in command.items() if k != "duration_s"},
        )["accepted"]
        initial = report["state"]
    assert initial["nodes"]["cell"]["component_moles"][1] == pytest.approx(
        sum(result["initial"]["nodes"]["cell"]["component_moles"]) + 2e-4, abs=1e-9
    )
    assert len(result["intervals"][0]["state"]["nodes"]["cell"]["phase_roots"]) == 2
    assert len(initial["nodes"]["cell"]["phase_roots"]) == 1
    assert any(r["fallback_intervals"] for r in result["intervals"])
    assert max(map(abs, result["conservation_residuals"])) < 1e-5


def test_open_cli_failed_run_preserves_existing_output(tmp_path):
    import json
    import os
    import subprocess
    import sys

    from check_pr_collocation import ROOT

    config = json.loads((ROOT / "configs/surface_pr_open_collocation.json").read_text())
    config["schedule"][0]["port_openings"] = {"unknown": 1.0}
    request = tmp_path / "input.json"
    request.write_text(json.dumps(config))
    output = tmp_path / "output.json"
    output.write_text("previous successful result")
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_pr_open_transport.py"),
            str(request),
            "--integrator",
            "radau3",
            "--output",
            str(output),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "Unknown external port" in process.stderr
    assert output.read_text() == "previous successful result"


@pytest.mark.parametrize("scheme", ["gauss4", "radau3"])
@pytest.mark.parametrize("epoch", [0.0, 1e9])
def test_clock_rounding_does_not_leave_an_unsplittable_report_tail(scheme, epoch):
    from check_pr_collocation import thermal_fixture

    model, state = thermal_fixture()
    model = PREquilibriumTransportNetwork.from_dict(
        {**model.to_dict(), "initial_step_s": 2e-6, "maximum_step_s": 2e-6}
    )
    state = replace(state, model_id=model.model_id, time_s=epoch, next_step_s=2e-6)
    duration = 1e-5 if epoch == 0 else 81 * __import__("math").ulp(epoch)
    result = PRCollocation(model, scheme=scheme, phase_fallback=False).advance(
        state, duration, openings={"contact": False}
    )
    assert result["state"]["time_s"] == epoch + duration
    assert sum(s["dt_s"] for s in result["steps"]) == pytest.approx(
        (epoch + duration) - epoch, rel=1e-14
    )
    assert all(0 < s["dt_s"] <= model.maximum_step_s for s in result["steps"])
    assert result["fallback_intervals"] == 0
    assert len(result["steps"]) < 20


@pytest.mark.parametrize("scheme", ["gauss4", "radau3"])
def test_tiny_meaningful_interval_is_integrated_and_not_skipped(scheme):
    from check_pr_collocation import thermal_fixture

    model, state = thermal_fixture()
    result = PRCollocation(model, scheme=scheme, phase_fallback=False).advance(
        state, 1e-12, openings={"contact": False}
    )
    assert result["state"]["time_s"] == 1e-12
    assert len(result["steps"]) == 2
    assert sum(s["dt_s"] for s in result["steps"]) == 1e-12
