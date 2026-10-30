"""Independent hydraulic limits, conservation, cycles, and explicit failures."""

import json
import math
import sys
from pathlib import Path

import pytest
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    HydraulicConvergenceError,
    InvalidPressureError,
    Liquid,
    LiquidNetwork,
    NetworkResult,
    Node,
    Pipe,
)

WATER = Liquid(1000.0, 0.001)
OIL = Liquid(850.0, 0.1)


def audit(result):
    assert max(map(abs, result.node_mass_residuals_kg_s.values()), default=0.0) < 1e-8
    assert max(map(abs, result.pipe_equation_residuals_pa.values()), default=0.0) < 0.001
    assert min(result.pipe_dissipation_w.values(), default=0.0) >= 0.0
    scale = max(1.0, sum(result.pipe_dissipation_w.values()))
    assert abs(result.power_balance_error_w) < 1e-7 * scale


def test_poiseuille_rate_boundary_and_hydrostatics():
    q = 0.001
    pipe = Pipe("P", "well", "out", 100.0, 0.1)
    net = LiquidNetwork(
        OIL,
        [
            Node("well", elevation_m=0.0, source_mass_rate_kg_s=q * 850),
            Node("out", elevation_m=12.0, pressure_pa=2e5),
        ],
        [pipe],
    )
    r = net.solve()
    expected = 2e5 + 850 * 9.80665 * 12 + 128 * 0.1 * 100 * q / (math.pi * 0.1**4)
    assert r.pressures_pa["well"] == pytest.approx(expected, abs=1e-7)
    assert r.pipe_volume_rates_m3_s["P"] == pytest.approx(q, abs=1e-14)
    assert r.boundary_mass_rates_kg_s["out"] == pytest.approx(-q * 850)
    audit(r)


@pytest.mark.parametrize("re", [1e4, 1e5, 1e6])
@pytest.mark.parametrize("roughness", [0.0, 0.0001, 0.001])
def test_turbulent_pipe_against_implicit_colebrook(re, roughness):
    d = 0.1
    pipe = Pipe("P", "a", "b", 250.0, d, roughness_m=roughness)
    q = re * WATER.viscosity_pa_s * math.pi * d / (4 * WATER.density_kg_m3)
    f = brentq(
        lambda f: (
            1 / math.sqrt(f) + 2 * math.log10(roughness / (3.7 * d) + 2.51 / (re * math.sqrt(f)))
        ),
        0.005,
        0.2,
    )
    velocity = q / (math.pi * d * d / 4)
    reference = f * 250 / d * WATER.density_kg_m3 * velocity**2 / 2
    loss, gradient = pipe.loss_and_gradient(q, WATER)
    # Swamee-Jain is an explicit approximation, not the exact Colebrook root.
    assert loss == pytest.approx(reference, rel=0.022)
    assert gradient > 0


@pytest.mark.parametrize("re", [0.0, 10.0, 1999.0, 2000.0, 3000.0, 4000.0, 4001.0, 1e5])
def test_pipe_gradient_and_reversal(re):
    p = Pipe("P", "a", "b", 300.0, 0.15, roughness_m=2e-5, minor_loss_coefficient=3.0)
    q = re * 0.001 * math.pi * 0.15 / (4 * 1000)
    h = max(q, 1e-4) * 1e-7
    value, derivative = p.loss_and_gradient(q, WATER)
    fd = (p.loss_and_gradient(q + h, WATER)[0] - p.loss_and_gradient(q - h, WATER)[0]) / (2 * h)
    assert derivative == pytest.approx(fd, rel=2e-6)
    reverse, reverse_derivative = p.loss_and_gradient(-q, WATER)
    assert reverse == pytest.approx(-value)
    assert reverse_derivative == pytest.approx(derivative)
    assert math.isfinite(derivative) and derivative > 0


def test_ring_parallel_branches_and_dead_end_against_linear_circuit():
    # In laminar flow the two a->out routes have resistance 2R and 4R.
    nodes = [
        Node("a", source_mass_rate_kg_s=0.85),
        Node("b"),
        Node("c"),
        Node("out", pressure_pa=2e5),
        Node("dead"),
    ]
    pipes = [
        Pipe("ab", "a", "b", 100.0, 0.1),
        Pipe("bo", "b", "out", 100.0, 0.1),
        Pipe("ac", "a", "c", 200.0, 0.1),
        Pipe("co", "c", "out", 200.0, 0.1),
        Pipe("dead", "b", "dead", 10.0, 0.1),
    ]
    net = LiquidNetwork(OIL, nodes, pipes)
    r = net.solve()
    assert r.pipe_mass_rates_kg_s["ab"] == pytest.approx(0.85 * 2 / 3)
    assert r.pipe_mass_rates_kg_s["ac"] == pytest.approx(0.85 / 3)
    assert abs(r.pipe_mass_rates_kg_s["dead"]) < 1e-12
    assert sum(r.boundary_mass_rates_kg_s.values()) == pytest.approx(-0.85)
    audit(r)


def test_parallel_turbulent_pipes_equal_split_and_reversed_orientation():
    net = LiquidNetwork(
        WATER,
        [Node("a", source_mass_rate_kg_s=30.0), Node("b", pressure_pa=1e6)],
        [Pipe("forward", "a", "b", 500.0, 0.1), Pipe("backward", "b", "a", 500.0, 0.1)],
    )
    r = net.solve()
    assert r.pipe_mass_rates_kg_s["forward"] == pytest.approx(15.0)
    assert r.pipe_mass_rates_kg_s["backward"] == pytest.approx(-15.0)
    assert r.iterations > 0
    audit(r)


def test_multiple_pressure_boundaries_drive_flow_without_sources():
    net = LiquidNetwork(
        OIL,
        [Node("a", pressure_pa=2.1e5), Node("b", pressure_pa=2e5)],
        [Pipe("p", "a", "b", 100.0, 0.1)],
    )
    r = net.solve()
    resistance = 128 * 0.1 * 100 / (math.pi * 0.1**4)
    assert r.pipe_volume_rates_m3_s["p"] == pytest.approx(1e4 / resistance)
    assert r.boundary_mass_rates_kg_s["a"] == pytest.approx(-r.boundary_mass_rates_kg_s["b"])
    audit(r)


def test_zero_flow_hydrostatic_limit():
    net = LiquidNetwork(
        WATER,
        [Node("bottom", pressure_pa=1e6), Node("top", elevation_m=50.0)],
        [Pipe("p", "bottom", "top", 100.0, 0.1)],
    )
    r = net.solve()
    assert r.pressures_pa["top"] == pytest.approx(1e6 - 1000 * 9.80665 * 50)
    assert abs(r.pipe_mass_rates_kg_s["p"]) < 1e-12
    audit(r)


def test_operating_overrides_and_json_warm_start_do_not_mutate_model():
    net = LiquidNetwork(
        OIL,
        [Node("a", pressure_pa=300000), Node("b", pressure_pa=200000)],
        [Pipe("p", "a", "b", 100.0, 0.1)],
    )
    config = json.loads(json.dumps(net.to_dict()))
    r = net.solve(boundary_pressures_pa={"a": None}, source_mass_rates_kg_s={"a": 0.85})
    restored = LiquidNetwork.from_dict(config)
    initial = NetworkResult.from_dict(json.loads(json.dumps(r.to_dict())))
    rerun = restored.solve(
        boundary_pressures_pa={"a": None}, source_mass_rates_kg_s={"a": 0.85}, initial=initial
    )
    assert rerun.iterations == 0
    assert rerun.pressures_pa == r.pressures_pa
    assert net.to_dict() == config
    assert net.solve().pressures_pa["a"] == 300000
    audit(rerun)


def test_closed_pipe_does_not_leak_across_pressure_boundaries():
    net = LiquidNetwork(
        WATER,
        [Node("a", pressure_pa=1e6), Node("b", pressure_pa=2e5)],
        [Pipe("closed", "a", "b", 100.0, 0.1, open=False)],
    )
    r = net.solve()
    assert r.pipe_mass_rates_kg_s == {"closed": 0.0}
    assert r.pipe_equation_residuals_pa == {}
    audit(r)


def test_unanchored_component_and_closed_disconnection_are_rejected():
    for open_pipe in [False, True]:
        net = LiquidNetwork(
            WATER,
            [Node("a", source_mass_rate_kg_s=1.0), Node("b")],
            [Pipe("p", "a", "b", 100.0, 0.1, open=open_pipe)],
        )
        with pytest.raises(ValueError, match="No pressure boundary"):
            net.solve()


def test_infeasible_pressure_and_iteration_failure_are_not_clipped():
    net = LiquidNetwork(
        OIL,
        [Node("a", source_mass_rate_kg_s=-10.0), Node("b", pressure_pa=10000.0)],
        [Pipe("p", "a", "b", 1000.0, 0.01)],
    )
    with pytest.raises(InvalidPressureError):
        net.solve()
    turbulent = LiquidNetwork(
        WATER,
        [Node("a", pressure_pa=2e6), Node("b", pressure_pa=1e6)],
        [Pipe("p", "a", "b", 100.0, 0.1)],
    )
    with pytest.raises(HydraulicConvergenceError):
        turbulent.solve(max_iterations=1)


@pytest.mark.parametrize(
    "make",
    [
        lambda: Liquid(-1.0, 0.001),
        lambda: Liquid(1000.0, float("nan")),
        lambda: Pipe("p", "a", "b", 1.0, 0.0),
        lambda: Pipe("p", "a", "b", 1.0, 0.1, open="false"),
        lambda: LiquidNetwork(WATER, [Node("a"), Node("a")], []),
        lambda: LiquidNetwork(WATER, [Node("a")], [Pipe("p", "a", "missing", 1.0, 0.1)]),
    ],
)
def test_invalid_input_is_rejected(make):
    with pytest.raises((ValueError, TypeError)):
        make()


def test_independent_epanet_22_ring():
    """Saved output from the official v2.2 shared library, not our solver."""
    fixture = json.loads((Path(__file__).parent / "data/surface/epanet_22_ring.json").read_text())
    network = LiquidNetwork.from_dict(fixture["input"])
    result = network.solve()
    audit(result)
    for name, reference in fixture["epanet_flows_lps"].items():
        # rho=1000: L/s and kg/s have identical numerical values.
        assert result.pipe_mass_rates_kg_s[name] == pytest.approx(reference, abs=1e-5)
    for node in network.nodes:
        head = node.elevation_m + result.pressures_pa[node.name] / (1000 * 9.80665)
        # EPANET's gravity and rounded K constants differ from SI production constants.
        assert head == pytest.approx(fixture["epanet_heads_m"][node.name], abs=0.02)


def test_closed_pipe_geometry_is_validated():
    with pytest.raises(ValueError, match="elevation difference"):
        LiquidNetwork(
            WATER,
            [Node("a", pressure_pa=1e5), Node("b", elevation_m=20, pressure_pa=1e5)],
            [Pipe("bad", "a", "b", 10, 0.1, open=False)],
        )


def test_meshed_collection_network():
    """Many cycles, distributed sources and one outlet; audit whole-network balances."""
    size = 10
    nodes = [
        Node(
            f"{i}_{j}",
            elevation_m=i + j,
            pressure_pa=4e5 if (i, j) == (0, 0) else None,
            source_mass_rate_kg_s=0 if (i, j) == (0, 0) else 0.1,
        )
        for i in range(size)
        for j in range(size)
    ]
    pipes = []
    for i in range(size):
        for j in range(size):
            for a, b in [(i + 1, j), (i, j + 1)]:
                if a < size and b < size:
                    pipes.append(
                        Pipe(f"{i}_{j}_{a}_{b}", f"{i}_{j}", f"{a}_{b}", 100, 0.1, roughness_m=5e-5)
                    )
    result = LiquidNetwork(WATER, nodes, pipes).solve()
    audit(result)
    assert result.boundary_mass_rates_kg_s["0_0"] == pytest.approx(-9.9, abs=1e-8)
    assert len(result.pipe_mass_rates_kg_s) == 180
