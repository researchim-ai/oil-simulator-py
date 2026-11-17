"""Batch pressure integrals and implicit AD against existing scalar physics."""

import pytest
import torch
from test_surface_blackoil_properties import model

from surface.blackoil_pipe import BlackOilPipe, ChokedFlowError
from surface.blackoil_pipe_batch import PipeBatch, required_inlets, tensor_state


def test_batched_properties_match_independent_scalar_rows_and_derivatives():
    m = model()
    masses = torch.tensor(
        [[0.85, 2.0, 0.06], [0.85, 0.2, 0.144], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
        dtype=torch.float64,
        requires_grad=True,
    )
    p = torch.tensor(
        [[1.2e6, 3e6], [1.5e6, 3.2e6], [1.1e6, 2.2e6], [1.3e6, 3.4e6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = tensor_state(m, p, masses)
    for i in range(4):
        scalar = m.tensor_state(p[i], masses[i])
        for key in scalar:
            torch.testing.assert_close(batch[key][i], scalar[key], rtol=1e-13, atol=1e-14)
        a = torch.autograd.grad(batch["specific_volume"][i].sum(), (p, masses), retain_graph=True)
        b = torch.autograd.grad(scalar["specific_volume"].sum(), (p, masses), retain_graph=True)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-12)


def test_heterogeneous_batch_matches_pressure_sensitivities_and_mechanical_energy():
    m = model()
    pipes = [
        BlackOilPipe("phase", 300, 0.08, 30, 1e-5),
        BlackOilPipe("gas", 100, 0.15, 5, 2e-5, 2),
        BlackOilPipe("water", 80, 0.03, 0, 1e-5),
    ]
    boundaries = [2.72e6, 1.1e6, 2.4e6]
    rates = [[0.85, 2, 0.06], [0.85, 1, 0.144], [0, 0.001, 0]]
    rows = PipeBatch(pipes, m).solve(boundaries, rates)
    for p, b, r, row in zip(pipes, boundaries, rates, rows):
        reference = p.required_inlet(b, r, m)
        assert row["inlet_pressure_pa"] == pytest.approx(reference["inlet_pressure_pa"], abs=0.001)
        assert abs(row["mechanical_balance_error_w"]) < 1e-5
        assert abs(row["length_residual_m"]) < 1e-5
        for key in row["pressure_derivatives"]:
            assert row["pressure_derivatives"][key] == pytest.approx(
                reference["pressure_derivatives"][key], rel=1e-7, abs=0.002
            )
        assert row["friction_dissipation_w"] == pytest.approx(
            reference["friction_dissipation_w"], rel=1e-8, abs=1e-8
        )


def test_real_physical_choking_is_not_hidden_by_scalar_retry():
    with pytest.raises(ChokedFlowError):
        required_inlets([BlackOilPipe("choked", 100, 0.01)], [2e6], [[0, 0, 1]], model())


def test_known_port_predictor_solves_constant_water_against_poiseuille():
    import math

    from test_surface_blackoil_properties import DENSITIES, pvt_data

    from surface.blackoil_properties import BlackOilMixture

    data = pvt_data()
    data["Bw"] = [1.0] * len(data["pressure_MPa"])
    m = BlackOilMixture(data, DENSITIES)
    pipe = BlackOilPipe("laminar", 80, 0.05)
    boundary, water = 2.4e6, 0.01
    # Independent constant-density laminar limit: Hagen-Poiseuille.
    tangent = 128 * 0.001 * pipe.length_m / (math.pi * 1000 * pipe.diameter_m**4)
    row = PipeBatch([pipe], m).solve([boundary], [[0, water, 0]])[0]
    assert row["inlet_pressure_pa"] == pytest.approx(boundary + tangent * water, abs=1e-8)
    assert row["pressure_derivatives"]["water_kg_s"] == pytest.approx(tangent, rel=1e-9)
    assert row["pressure_derivatives"]["boundary_pressure_pa"] == pytest.approx(1.0, abs=1e-12)
    assert abs(row["length_residual_m"]) < 1e-8
    assert row["pressure_evaluations"] == 2  # Bracket and verified physical root.


def test_scalar_retry_is_explicit_when_batch_budget_fails(monkeypatch):
    from surface.liquid import HydraulicConvergenceError

    def fail(*args):
        raise HydraulicConvergenceError("injected batch quadrature budget")

    monkeypatch.setattr(PipeBatch, "solve", fail)
    row = required_inlets([BlackOilPipe("water", 80, 0.03)], [2.4e6], [[0, 0.001, 0]], model())[0]
    assert row["hydraulic_backend"] == "scalar_retry"
    assert abs(row["length_residual_m"]) < 1e-5


def test_batched_implicit_composition_derivatives_against_finite_differences():
    p = [BlackOilPipe("bubble", 300, 0.08, 30, 1e-5), BlackOilPipe("other", 120, 0.11, 4, 2e-5)]
    b = [2.72e6, 1.3e6]
    rates = [[0.85, 2.0, 0.06], [0.3, 0.5, 0.03]]
    kernel = PipeBatch(p, model())
    reference = kernel.solve(b, rates)
    for j, (key, h) in enumerate(zip(("oil_kg_s", "water_kg_s", "gas_kg_s"), (1e-5, 1e-5, 1e-7))):
        import copy

        plus, minus = copy.deepcopy(rates), copy.deepcopy(rates)
        plus[0][j] += h
        minus[0][j] -= h
        a, c = kernel.solve(b, plus), kernel.solve(b, minus)
        fd = (a[0]["inlet_pressure_pa"] - c[0]["inlet_pressure_pa"]) / (2 * h)
        assert reference[0]["pressure_derivatives"][key] == pytest.approx(fd, rel=3e-6, abs=0.002)
        assert a[1]["inlet_pressure_pa"] == c[1]["inlet_pressure_pa"]


def test_network_batch_preserves_cross_well_jacobian_and_controls():
    from test_surface_blackoil_network import feeds, network

    from surface.blackoil_network import BlackOilProductionNetwork

    n = network(2.75e6)
    spec = n.to_dict()
    spec.update(hydraulic_backend="torch_batch", pipe_batch_size=2)
    batched = BlackOilProductionNetwork.from_dict(spec)
    assert batched.with_controls().hydraulic_backend == "torch_batch"
    assert batched.to_dict()["pipe_batch_size"] == 2
    for mode in ["dense", "operator"]:
        a, b = (
            n.solve(feeds(), sensitivity_mode=mode),
            batched.solve(feeds(), sensitivity_mode=mode),
        )
        assert b["hydraulic_execution"]["torch_batch_pipes"] == 2
        assert b["hydraulic_execution"]["scalar_retry_pipes"] == 0
        for node in a["node_pressures_pa"]:
            assert a["node_pressures_pa"][node] == pytest.approx(
                b["node_pressures_pa"][node], abs=0.001
            )
        if mode == "dense":
            for node, sources in a["pressure_derivatives_per_kg_s"].items():
                for source, row in sources.items():
                    assert row == pytest.approx(
                        b["pressure_derivatives_per_kg_s"][node][source], rel=1e-7, abs=0.002
                    )


def test_appearance_event_regression_from_existing_physical_fixture():
    import json
    from pathlib import Path

    from surface.blackoil_properties import BlackOilMixture
    from surface.process import StockDensities

    fixture = json.loads(
        (Path(__file__).parent / "data/surface/blackoil_phase_transition.json").read_text()
    )
    m = BlackOilMixture(fixture["pvt"], StockDensities(850, 1000, 1.2))
    pipe = BlackOilPipe(**fixture["pipe"])
    rates = fixture["component_mass_rates_kg_s"]
    direct = pipe.run(fixture["inlet_pressure_pa"], rates, m)
    row = PipeBatch([pipe], m).solve([direct["outlet_pressure_pa"]], [rates])[0]
    assert row["inlet_pressure_pa"] == pytest.approx(fixture["inlet_pressure_pa"], abs=0.001)
    assert row["inlet_state"]["phase_mass_rates_kg_s"][2] == 0
    assert row["outlet_state"]["phase_mass_rates_kg_s"][2] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_batch_matches_cpu_float64():
    pipes = [BlackOilPipe("a", 300, 0.08, 30, 1e-5), BlackOilPipe("b", 200, 0.1, 10)]
    boundaries, rates = [2.72e6, 1.2e6], [[0.85, 2.0, 0.06], [0.4, 1.0, 0.06]]
    cpu = PipeBatch(pipes, model()).solve(boundaries, rates)
    gpu = PipeBatch(pipes, model().to("cuda")).solve(boundaries, rates)
    for a, b in zip(cpu, gpu):
        assert b["compute_device"].startswith("cuda")
        assert a["inlet_pressure_pa"] == pytest.approx(b["inlet_pressure_pa"], abs=0.001)
        assert a["pressure_derivatives"] == pytest.approx(
            b["pressure_derivatives"], rel=1e-7, abs=0.002
        )


def test_fim_uses_batch_and_restart_preserves_accepted_solution():
    import pickle

    from test_fim_blackoil_surface import config, solver

    c = config()
    network = c["simulation"]["physical_surface_network"]
    network["well_block_backend"] = "tree_structured"
    network["network"]["hydraulic_backend"] = "torch_batch"
    s = solver(c)
    assert s.step(86400)
    execution = s.wells.surface_report["network"]["hydraulic_execution"]
    assert execution["torch_batch_pipes"] == 2
    assert execution["scalar_retry_pipes"] == 0
    restart = pickle.loads(pickle.dumps(s))
    assert s.step(86400) and restart.step(86400)
    torch.testing.assert_close(s.fluid.pressure, restart.fluid.pressure, rtol=0, atol=0)


def test_old_network_checkpoint_defaults_to_scalar_controls():
    from test_surface_blackoil_network import feeds, network

    n = network()
    del n.hydraulic_backend, n.pipe_batch_size
    restored = n.with_controls()
    assert restored.hydraulic_backend == "scalar"
    assert restored.solve(feeds())["hydraulic_execution"]["torch_batch_pipes"] == 0
