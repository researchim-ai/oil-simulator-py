"""Common PR transport must use both relative slip AND total wall reaction."""

import json
from pathlib import Path

import pytest
import torch

from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork

ROOT = Path(__file__).resolve().parents[1]


def test_stratified_graph_wall_force_balances_and_independent_restart():
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    cfg = json.loads((ROOT / "configs/surface_pr_stratified_drift.json").read_text())
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    assert result["intervals"][1]["state"]["nodes"]["pipe"]["component_moles"] == pytest.approx(
        result["intervals"][0]["state"]["nodes"]["pipe"]["component_moles"], abs=1e-12
    )
    restart = {
        **cfg,
        "initial": {"state": result["intervals"][-1]["state"]},
        "schedule": [{"duration_s": 1e-7}],
    }
    assert audit(restart, json.loads(json.dumps(run(restart))))["accepted"]


def test_stratified_common_jacobian_includes_relative_shear_and_wall_load():
    cfg = json.loads((ROOT / "configs/surface_pr_stratified_drift.json").read_text())
    model = PREquilibriumTransportNetwork.from_dict(cfg["model"])
    state = model.initialize(cfg["initial"]["nodes"])
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, *_ = model._problem(state, seeds, 1e-6, model.default_openings, [0.0] * 2)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-0.01, 0.01, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-5, atol=2e-7)
