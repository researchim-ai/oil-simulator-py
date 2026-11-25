"""Industrial KPI units and integrals must survive unequal accepted steps."""

import copy
import json

import pytest
from export_field_forecast import export, production_series


def example():
    config = {
        "simulation": {
            "coupled_facility": {
                "facility": {
                    "stock_densities": {"oil_kg_m3": 850, "water_kg_m3": 1000, "gas_kg_m3": 1.2}
                }
            }
        }
    }

    def step(time, dt, masses, power):
        return {
            "time_days": time,
            "dt_s": dt,
            "withdrawn_component_mass_kg": masses,
            "facility_coupling": {
                "fraction": 0.5,
                "facility": {
                    "measurements": {"facility_drive_power_total_w": {"value": power, "unit": "W"}}
                },
            },
            "surface": {"wells": {"P1": {"bhp_pa": 1e6}, "P2": {"bhp_pa": 2e6}}},
        }

    result = {
        "component_order": ["oil", "water", "gas"],
        "accepted_substeps": [
            step(1, 86400, [850, 1000, 1.2], 1000),
            step(1.5, 43200, [1700, 0, 0], 500),
        ],
    }
    return config, result


def test_stock_units_and_actual_stage_integrals():
    config, result = example()
    rows = production_series(config, result)
    assert rows[0]["water_cut_stock"] == 0.5
    assert rows[0]["gas_oil_ratio_stock_m3_m3"] == 1
    assert rows[1]["oil_rate_stock_m3_day"] == 4
    assert rows[1]["oil_cumulative_stock_m3"] == 3
    assert rows[1]["water_cumulative_stock_m3"] == 1
    assert rows[1]["gas_cumulative_stock_m3"] == 1
    assert rows[1]["cumulative_equipment_drive_energy_kwh"] == 30
    assert rows[1]["bhp_min_pa"] == 1e6
    assert rows[1]["bhp_max_pa"] == 2e6
    assert result == example()[1]


def test_zero_flow_ratios_are_undefined_not_invented():
    config, result = example()
    result["accepted_substeps"] = [result["accepted_substeps"][0]]
    result["accepted_substeps"][0]["withdrawn_component_mass_kg"] = [0, 0, 1.2]
    row = production_series(config, result)[0]
    assert row["water_cut_stock"] is None
    assert row["gas_oil_ratio_stock_m3_m3"] is None
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("problem", ["density", "order", "injection", "power_unit"])
def test_incompatible_accounting_is_rejected(problem):
    config, result = copy.deepcopy(example())
    if problem == "density":
        config["simulation"]["coupled_facility"]["facility"]["stock_densities"]["gas_kg_m3"] = 0
    elif problem == "order":
        result["component_order"].reverse()
    elif problem == "injection":
        result["accepted_substeps"][0]["withdrawn_component_mass_kg"][0] = -1
    else:
        result["accepted_substeps"][0]["facility_coupling"]["facility"]["measurements"][
            "facility_drive_power_total_w"
        ]["unit"] = "kW"
    with pytest.raises(ValueError):
        production_series(config, result)


def test_changed_trajectory_cannot_publish_kpis(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "sources_unchanged": True,
                "audit": {"elapsed_days": 1},
                "trajectory_sha256": "forged",
            }
        )
    )
    (source / "progress.json").write_text('{"status":"complete"}')
    (source / "trajectory.json.gz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        export(source, output)
    assert not output.exists()
