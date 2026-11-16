"""The scalar/batch qualification must reject changed physics and fake execution."""

import copy
import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_pipe_batch_forecast import compare
from run_fim_blackoil_surface import run_case

BUDGETS = {
    "final_pressure_pa": 0.1,
    "final_s_w": 1e-8,
    "final_s_g": 1e-8,
    "final_rs": 1e-5,
    "withdrawal_relative": 1e-6,
}


@pytest.fixture(scope="module")
def trajectories():
    config = json.loads((ROOT / "configs/fim_blackoil_surface.json").read_text())
    config["fluid"]["pvt_path"] = str(ROOT / config["fluid"]["pvt_path"])
    config["simulation"]["physical_surface_network"]["well_block_backend"] = "tree_structured"
    bundles = []
    for mode in ["scalar", "torch_batch"]:
        c = copy.deepcopy(config)
        c["simulation"]["physical_surface_network"]["network"]["hydraulic_backend"] = mode
        bundles.append({"config": c, "requested_days": 1, "result": run_case(c, 1e6, 1, 1, "cpu")})
    return bundles


def save(path, bundle):
    with gzip.open(path, "wt") as handle:
        json.dump(bundle, handle)


def test_real_batch_forecast_matches_scalar_and_passes_audits(trajectories, tmp_path):
    a, b = [tmp_path / name for name in ["a.gz", "b.gz"]]
    for p, r in zip([a, b], trajectories):
        save(p, r)
    result = compare(a, b, BUDGETS)
    assert result["accepted"]
    assert result["accepted_step_pipe_counts"]["torch_batch_pipes"] == 2


@pytest.mark.parametrize(
    "mutation", ["physics", "execution", "component", "budget", "accuracy", "nonfinite", "counts"]
)
def test_corrupt_qualification_is_rejected(trajectories, tmp_path, mutation):
    base, batch = copy.deepcopy(trajectories)
    budgets = copy.deepcopy(BUDGETS)
    if mutation == "physics":
        batch["config"]["reservoir"]["porosity"] = 0.33
    elif mutation == "execution":
        batch["result"]["accepted_substeps"][0]["surface"]["network"]["hydraulic_execution"][
            "torch_batch_pipes"
        ] = 0
    elif mutation == "component":
        batch["result"]["accepted_substeps"][0]["surface"]["network"][
            "receiving_component_mass_rates_kg_s"
        ]["oil"] += 1
    elif mutation == "budget":
        budgets["final_pressure_pa"] = float("inf")
    elif mutation == "nonfinite":
        batch["result"]["final_pressure_pa"][-1] = float("nan")
    elif mutation == "counts":
        batch["result"]["accepted_substeps"][0]["surface"]["network"]["hydraulic_execution"][
            "scalar_pipes"
        ] += 1
    else:
        batch["result"]["final_pressure_pa"][0] += 10
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"
    save(a, base)
    save(b, batch)
    if mutation == "accuracy":
        assert not compare(a, b, budgets)["accepted"]
    else:
        with pytest.raises(ValueError):
            compare(a, b, budgets)
