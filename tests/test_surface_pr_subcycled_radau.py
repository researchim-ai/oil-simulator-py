"""A stiffly accurate fast integrator retains the full split-network audit."""

import copy
import json
from dataclasses import replace

import pytest
from check_pr_controlled_surface import audit
from run_pr_controlled_surface import run
from test_surface_pr_subcycling import subcycle_config  # noqa: F401

from surface.pr_subcycled_surface import PRSubcycledSurface
from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState


@pytest.fixture(scope="module")
def radau_case(subcycle_config):  # noqa: F811
    config = copy.deepcopy(subcycle_config)
    original = PRControlledSurface.from_dict(config["model"])
    model = PRSubcycledSurface(
        original.reference_surface, original.pipe_nodes, fast_time_scheme="radau3"
    )
    old = PRControlledSurfaceState.from_dict(config["initial"]["state"])
    state = replace(old, model_id=model.model_id)
    model.inspect(state)
    config.update(model=model.to_dict(), initial={"state": state.to_dict()})
    return json.loads(json.dumps(config))


@pytest.fixture(scope="module")
def radau_result(radau_case):
    return json.loads(json.dumps(run(radau_case)))


def test_radau_connected_network_and_restart(radau_case, radau_result):
    checked = audit(radau_case, radau_result)
    assert checked["accepted"]
    model = PRControlledSurface.from_dict(radau_case["model"])
    assert model.fast_integrator.scheme == "radau3"
    assert json.loads(json.dumps(model.to_dict())) == radau_case["model"]
    final = radau_result["intervals"][-1]["state"]
    model.inspect(PRControlledSurfaceState.from_dict(final))
    methods = {
        record["method"]
        for interval in radau_result["intervals"]
        for step in interval["steps"]
        for stage in step["local_subcycling"]["stages"]
        if stage["operator"] == "fast"
        for record in stage["report"]["steps"]
    }
    assert methods == {"radau3"}
    assert final["plant"]["transport"]["time_s"] == 1e-8


def test_independent_audit_rejects_wrong_fast_scheme(radau_case, radau_result):
    report = copy.deepcopy(radau_result)
    stage = report["intervals"][0]["steps"][0]["local_subcycling"]["stages"][0]
    stage["report"]["steps"][0].pop("method")
    with pytest.raises(ValueError, match="integration method"):
        audit(radau_case, report)


def test_invalid_fast_scheme_rejected(subcycle_config):  # noqa: F811
    model = PRControlledSurface.from_dict(subcycle_config["model"])
    with pytest.raises(ValueError, match="time scheme"):
        PRSubcycledSurface(model.reference_surface, model.pipe_nodes, fast_time_scheme="explicit")
