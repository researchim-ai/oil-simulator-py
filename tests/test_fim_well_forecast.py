"""Reject corrupted accepted material transfers in an actual structured FIM run."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_fim_well_forecast import audit
from run_fim_blackoil_surface import run_case


@pytest.fixture(scope="module")
def forecast():
    config = json.loads((ROOT / "configs/fim_blackoil_surface.json").read_text())
    config["fluid"]["pvt_path"] = str(ROOT / config["fluid"]["pvt_path"])
    config["simulation"]["physical_surface_network"]["well_block_backend"] = "tree_structured"
    # Bind at the network pressure, so this exercises the pressure row too.
    for well in config["wells"]:
        well["control_value"] = 10000
    return config, run_case(config, 1e6, 1, 1, "cpu")


def test_accepted_producing_forecast_passes_material_audit(forecast):
    c, r = forecast
    checked = audit(c, r, expected_days=1)
    assert checked["accepted"] and checked["mode_samples"]["surface"] > 0


@pytest.mark.parametrize(
    "mutation",
    [
        "dense",
        "backend",
        "cap",
        "bhp",
        "receiver",
        "stock",
        "withdrawal",
        "inventory",
        "clock",
        "registry",
    ],
)
def test_audit_rejects_corrupt_accepted_transfers(forecast, mutation):
    c, original = forecast
    r = copy.deepcopy(original)
    step = r["accepted_substeps"][0]
    physical = step["surface"]
    row = next(iter(physical["wells"].values()))
    if mutation == "dense":
        physical["execution"]["dense_B_stored"] = True
    elif mutation == "backend":
        physical["execution"]["well_block_backend"] = "dense"
    elif mutation == "cap":
        row["standard_control_rate_m3_s"] = 1e6
    elif mutation == "bhp":
        row["bhp_pa"] += 1e6
    elif mutation == "receiver":
        physical["network"]["receiving_component_mass_rates_kg_s"]["oil"] += 1
    elif mutation == "stock":
        physical["network"]["receiving_stream"]["oil_m3_s"] += 1
    elif mutation == "withdrawal":
        step["withdrawn_component_mass_kg"][0] += 1
    elif mutation == "inventory":
        r["final_component_mass_kg"][0] += 1e5
    elif mutation == "clock":
        step["time_days"] += 1
    elif mutation == "registry":
        physical["wells"].pop(next(iter(physical["wells"])))
    with pytest.raises(ValueError):
        audit(c, r, expected_days=1)
