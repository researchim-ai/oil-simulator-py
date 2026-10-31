"""Shared collector pressure coupling and composition mixing before PVT flash."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_surface_blackoil_properties import DENSITIES, model

from surface.blackoil_network import BlackOilProductionNetwork, ProductionNode, ProductionPipe
from surface.process import ComponentStream


def network(outlet=1.2e6):
    nodes = [
        ProductionNode("A", -40),
        ProductionNode("B", -30),
        ProductionNode("header", -10),
        ProductionNode("separator", 0, outlet),
    ]
    pipes = [
        ProductionPipe("a", "A", "header", 300, 0.06, 1e-5),
        ProductionPipe("b", "B", "header", 200, 0.06, 1e-5),
        ProductionPipe("export", "header", "separator", 700, 0.07, 1e-5),
    ]
    return BlackOilProductionNetwork(model(), nodes, pipes)


def feeds():
    return {
        "A": ComponentStream(0.0007, 0.0014, 0.035),
        "B": ComponentStream(0.0003, 0.0006, 0.025),
    }


def test_mixing_conservation_and_cross_well_pressure_feedback():
    n = network()
    r = n.solve(feeds())
    assert r["node_pressures_pa"]["A"] > r["node_pressures_pa"]["header"] > 1.2e6
    assert r["receiving_stream"] == pytest.approx(
        {"oil_m3_s": 0.001, "water_m3_s": 0.002, "gas_m3_s": 0.06}
    )
    assert (
        max(abs(v) for rr in r["node_component_mass_residuals_kg_s"].values() for v in rr.values())
        < 1e-15
    )
    assert abs(r["mechanical_balance_error_w"]) < 1e-5
    j = r["pressure_derivatives_per_kg_s"]
    assert j["A"]["B"]["water"] != 0 and j["B"]["A"]["water"] != 0
    more = feeds()
    more["B"] = ComponentStream(0.0003, 0.0012, 0.025)
    assert n.solve(more)["node_pressures_pa"]["A"] > r["node_pressures_pa"]["A"]


@pytest.mark.parametrize("outlet", [1.2e6, 2.75e6])
def test_full_component_pressure_jacobian_against_finite_differences(outlet):
    n = network(outlet)
    s = feeds()
    r = n.solve(s)
    if outlet > 2e6:
        assert r["pipes"]["a"]["inlet_state"]["phase_mass_rates_kg_s"][2] == 0
        assert r["pipes"]["a"]["outlet_state"]["phase_mass_rates_kg_s"][2] > 0
    for source in s:
        base = np.array(list(s[source].mass_rates(DENSITIES).values()))
        for k, (comp, h) in enumerate(zip(("oil", "water", "gas"), (1e-5, 1e-5, 1e-7))):
            plus = base.copy()
            minus = base.copy()
            plus[k] += h
            minus[k] -= h
            a = dict(s)
            b = dict(s)
            a[source] = ComponentStream.from_mass_rates(*plus, DENSITIES)
            b[source] = ComponentStream.from_mass_rates(*minus, DENSITIES)
            pa = n.solve(a)["node_pressures_pa"]
            pb = n.solve(b)["node_pressures_pa"]
            for node in ("A", "B", "header"):
                fd = (pa[node] - pb[node]) / (2 * h)
                assert r["pressure_derivatives_per_kg_s"][node][source][comp] == pytest.approx(
                    fd, rel=1e-5, abs=0.003
                )
    plus = network(outlet + 2).solve(s)
    minus = network(outlet - 2).solve(s)
    for node in ("A", "B", "header"):
        fd = (plus["node_pressures_pa"][node] - minus["node_pressures_pa"][node]) / 4
        assert r["pressure_derivatives_receiving_pressure"][node] == pytest.approx(fd, rel=1e-6)


def test_tree_validation_and_empty_branch_failure():
    n = network()
    with pytest.raises(ValueError, match="Unknown"):
        BlackOilProductionNetwork(
            model(), n.nodes, [ProductionPipe("x", "missing", "separator", 100, 0.1)]
        )
    with pytest.raises(ValueError, match="split"):
        BlackOilProductionNetwork(
            model(), n.nodes, [*n.pipes, ProductionPipe("x", "A", "separator", 100, 0.1)]
        )
    with pytest.raises(ValueError, match="known"):
        n.solve({"missing": ComponentStream(1, 0, 0)})
    with pytest.raises(ValueError, match="Zero-flow"):
        n.solve({"A": feeds()["A"]})
