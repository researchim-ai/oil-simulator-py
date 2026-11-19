"""Explicit engineering reduction preserves stocks and rejects lost physics."""

import copy
import json
import math

import pytest
from check_pr_controlled_surface import audit
from run_pr_controlled_surface import run
from test_surface_pr_subcycling import small_network

from surface.pr_model_reduction import reduce_cells
from surface.pr_surface_control import PRControlledSurface


def parallel_config():
    original, _, _, _ = small_network()
    spec = json.loads(json.dumps(original.to_dict()))
    t = spec["plant"]["transport"]
    pipe = copy.deepcopy(next(n for n in t["nodes"] if n["name"] == "fast"))
    pipe["name"] = "parallel"
    t["nodes"].append(pipe)
    for source in list(t["links"]):
        if "fast" in (source["source"], source["target"]):
            link = copy.deepcopy(source)
            link["name"] += "_parallel"
            for key in ("source", "target"):
                if link[key] == "fast":
                    link[key] = "parallel"
            t["links"].append(link)
    model = PRControlledSurface.from_dict(spec)
    ts = model.plant.transport.initialize(
        {
            n.name: {
                "pressure_pa": 1.0001e6
                if n.name == "a"
                else 1.00005e6
                if n.kind == "pipe_cell" and n.name != "slow"
                else 1e6,
                "temperature_k": 350.0,
                "mole_fractions": [1.0, 0.0],
            }
            for n in model.plant.transport.nodes
        }
    )
    state = model.initialize(model.plant.initialize(ts, {}), {"v": 1.0})
    return json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"state": state.to_dict()},
                "schedule": [{"duration_s": 1e-5, "targets": {"v": 1.0}}],
            }
        )
    )


@pytest.fixture(scope="module")
def config():
    return parallel_config()


def totals(c):
    nodes = c["initial"]["state"]["plant"]["transport"]["nodes"].values()
    return [math.fsum(n["component_moles"][i] for n in nodes) for i in range(2)] + [
        math.fsum(n["total_energy_j"] for n in nodes),
        math.fsum(n["volume_m3"] for n in c["model"]["plant"]["transport"]["nodes"]),
    ]


def test_multiple_reductions_conserve_and_are_deterministic(config):
    original = copy.deepcopy(config)
    a, mapping = reduce_cells(config, ["parallel", "fast"])
    b, other = reduce_cells(config, ["fast", "parallel"])
    assert a == b and mapping == other
    assert config == original
    assert totals(a) == pytest.approx(totals(config), rel=1e-12, abs=1e-9)
    assert len(a["model"]["plant"]["darcy_connections"]) == 2
    assert a["model"]["valves"] == config["model"]["valves"]
    assert all(m["laminar_inertia_time_s"] > 0 for m in mapping["cells"])


@pytest.mark.parametrize(
    "key,value", [("heat_w", 1.0), ("utility_temperature_k", 400.0), ("loss_coefficients", 2.0)]
)
def test_scheduled_pipe_physics_is_not_silently_lost(config, key, value):
    c = copy.deepcopy(config)
    c["schedule"][0][key] = {"fast": value}
    with pytest.raises(ValueError, match="Scheduled pipe"):
        reduce_cells(c, ["fast"])


def test_failed_batch_is_atomic_and_controlled_pipe_is_not_removed(config):
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="another physical connection"):
        reduce_cells(config, ["fast", "slow"])
    assert config == before


def test_reduced_network_runs_and_audits_with_isolation(config):
    c = copy.deepcopy(config)
    c["schedule"].append(
        {"duration_s": 1e-5, "targets": {"v": 1.0}, "openings": {"af": False, "fb": False}}
    )
    reduced, _ = reduce_cells(c, ["fast", "parallel"])
    reduced = json.loads(json.dumps(reduced))
    assert reduced["schedule"][1]["openings"] == {"fast_darcy": False}
    result = json.loads(json.dumps(run(reduced)))
    assert audit(reduced, result)["accepted"]
    assert result["intervals"][-1]["state"]["plant"]["transport"]["time_s"] == 2e-5


def test_partial_isolation_requires_retained_line_storage(config):
    c = copy.deepcopy(config)
    c["schedule"][0]["openings"] = {"af": False}
    with pytest.raises(ValueError, match="Partial isolation"):
        reduce_cells(c, ["fast"])
