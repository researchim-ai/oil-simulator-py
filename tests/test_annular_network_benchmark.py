"""Protect paired comparisons against state mismatches and leaked overrides."""

from copy import deepcopy
from pathlib import Path

import pytest
from audit_fetrig_gas_profiles import run as gas_contract
from benchmark_annular_network import compare_states, measured_algorithm

import surface.annular_implicit_source as kernel
from surface.pr_active_annular_friction import PRActiveAnnularFriction


def test_temporary_reference_algorithm_restores_on_failure():
    original = kernel.implicit_annular_velocities
    apply = PRActiveAnnularFriction.apply
    with pytest.raises(RuntimeError, match="interrupted"), measured_algorithm("bisection"):
        assert kernel.implicit_annular_velocities is not original
        raise RuntimeError("interrupted")
    assert kernel.implicit_annular_velocities is original
    assert PRActiveAnnularFriction.apply is apply


def test_state_comparison_accepts_serialized_sequences_and_rejects_physical_or_topology_changes():
    a = {
        "time_s": 1.0,
        "cells": [{"gas": {"pressure_pa": 100000.0, "component_moles": (0.1, 0.2)}}],
    }
    b = deepcopy(a)
    b["cells"][0]["gas"]["component_moles"] = [0.1, 0.2]
    assert all(v == 0 for v in compare_states(a, b).values())
    b["cells"][0]["gas"]["pressure_pa"] += 1.0
    with pytest.raises(ValueError, match="pressure_pa"):
        compare_states(a, b)
    with pytest.raises(ValueError, match="time_s"):
        compare_states(a, {**a, "time_s": 2.0})
    with pytest.raises(ValueError, match="registry"):
        compare_states(a, {**a, "cells": []})


def test_public_velocity_profiles_do_not_become_wet_interfacial_shear():
    report = gas_contract(
        Path(__file__).resolve().parents[1] / "tests/data/surface/fetrig_gas_profiles"
    )
    assert len(report["profiles"]) == 5 and report["points"] == 160
    assert not report["direct_wet_interfacial_shear_available"]
    assert not report["experimental_wet_shear_qualification"]
    assert report["author_plotter_also_constructs_shifted_coordinates"]
    assert not report["wall_coordinate_shift_applied"]
    assert all(
        point["instrument_velocity_uncertainty_m_s"] == 0.2
        for p in report["profiles"]
        for point in p["points"]
    )


@pytest.mark.parametrize("branches", [0, 17, True])
def test_parallel_benchmark_rejects_invalid_branch_counts(branches):
    from benchmark_annular_network import parallel_config

    with pytest.raises(ValueError, match="parallel branches"):
        parallel_config({}, branches, "cpu")


def test_parallel_branches_share_reinitialized_column_and_preserve_input():
    import json

    from benchmark_annular_network import ROOT, parallel_config

    from surface.pr_active_surface import PRActiveSurfaceNetwork, PRActiveSurfaceState

    source = json.loads(
        (ROOT / "configs/surface_pr_active_surface_annular_implicit.json").read_text()
    )
    preserved = deepcopy(source)
    config = parallel_config(source, 2, "cpu")
    assert source == preserved
    assert len(config["system"]["pipes"]) == 2
    assert len(config["system"]["columns"]) == 1
    assert {p["column"] for p in config["system"]["ports"]} == {"separator"}
    old = source["initial"]["columns"]["separator"]
    new = config["initial"]["columns"]["separator"]
    assert new["component_moles"] == pytest.approx([2 * n for n in old["component_moles"]])
    assert new["total_energy_j"] == pytest.approx(2 * old["total_energy_j"])
    assert new["model_id"] != old["model_id"]
    model = PRActiveSurfaceNetwork.from_dict(config["system"])
    model.inspect(PRActiveSurfaceState.from_dict(config["initial"]))
    for period in config["schedule"]:
        assert set(period["commands"]["port_positions"]) == {"pickup_00", "pickup_01"}
