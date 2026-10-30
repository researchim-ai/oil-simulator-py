"""Preserve the original VFP network's rate, WEFAC, ALQ, and pressure semantics."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.blackoil_fim import BlackOilWells
from surface.vfp_network import evaluate_vfp_network


class LinearTable:
    def bhp(self, oil, water, gas, upstream, alq):
        return upstream + 100 * oil + 200 * water + 300 * gas + alq


def test_existing_well_network_accounts_for_efficiency_status_and_lift():
    context = SimpleNamespace(
        network={
            "nodes": {"root": {"p": 100000.0}, "group": {"lift": True}},
            "branches": {"group": {"up": "root", "vfp": 1, "alq": 7.0}},
        },
        groups=(SimpleNamespace(well_group={"a": "group", "b": "group", "i": "group"}),),
        wells=[
            SimpleNamespace(name="a", type="producer", eff=0.5, alq=10.0),
            SimpleNamespace(name="b", type="producer", eff=1.0, alq=0.0),
            SimpleNamespace(name="i", type="injector", eff=1.0, alq=0.0),
        ],
        step_targets=[("rate", 1.0), ("shut", 0.0), ("rate", 1.0)],
        vfp={1: LinearTable()},
        alq_si=0.01,
        device=torch.device("cpu"),
    )
    q = torch.tensor(
        [[2.0, 4.0, 6.0], [100.0, 100.0, 100.0], [-5.0, 0.0, 0.0]], dtype=torch.float64
    )
    pressures = BlackOilWells._network_pressures(context, q)
    expected = 100000 + 100 * 1 + 200 * 2 + 300 * (3 + 0.5 * 10 * 0.01) + 7
    assert pressures == {"root": 100000.0, "group": expected}


def test_tree_aggregates_descendants_and_preserves_no_loss_branch():
    network = {
        "nodes": {"root": {"p": 1e5}},
        "branches": {
            "a": {"up": "root", "vfp": 1, "alq": 0.0},
            "b": {"up": "a", "vfp": 9999, "alq": 0.0},
        },
    }
    rates = {
        "a": [(torch.tensor([1.0, 0.0, 0.0]), None)],
        "b": [(torch.tensor([0.0, 2.0, 0.0]), None)],
    }
    r = evaluate_vfp_network(network, {1: LinearTable()}, rates, torch.zeros(3))
    assert r["a"] == 1e5 + 100 + 400
    assert r["b"] == r["a"]


def test_fixed_rate_evaluation_has_no_recursive_depth_cutoff():
    branches = {f"n{i}": {"up": f"n{i - 1}", "vfp": 9999, "alq": 0.0} for i in range(1, 1200)}
    r = evaluate_vfp_network(
        {"nodes": {"n0": {"p": 2e5}}, "branches": branches}, {}, {}, torch.zeros(3)
    )
    assert len(r) == 1200 and all(p == 2e5 for p in r.values())


def test_vfp_cycle_is_explicitly_rejected():
    network = {
        "nodes": {},
        "branches": {"a": {"up": "b", "vfp": 9999}, "b": {"up": "a", "vfp": 9999}},
    }
    with pytest.raises(ValueError, match="Cyclic VFP"):
        evaluate_vfp_network(network, {}, {}, torch.zeros(3))
