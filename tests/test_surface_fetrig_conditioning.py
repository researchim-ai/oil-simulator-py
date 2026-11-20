"""Independent instrument clocks, input-only selection and honest qualification."""

import copy
from pathlib import Path

import pytest
from fetrig_labview import window_statistics
from validate_fetrig_film import run, steady_windows


def series(name, value, offset=0.0):
    return {
        "name": name,
        "times_excel_days": [45000 + (i * 0.5 + offset) / 86400 for i in range(121)],
        "values_native": [value] * 121,
        "source_rows": list(range(2, 123)),
    }


def test_windows_pair_by_time_and_reject_channel_gaps_without_interpolation():
    inputs = {
        "liquid_flow": series("mass", 160.2),
        "gas_volume_flow": series("gas", 0.013),
        "liquid_inlet_temperature": series("T", 25.0, 0.1),
        "gas_density": series("rho", 1.17, 0.2),
        "gas_viscosity": series("mu", 1.8e-5, 0.3),
    }
    result = steady_windows(inputs)
    assert result["fixed_windows"] == 2 and len(result["accepted"]) == 2
    assert result["accepted"][0]["candidate_SI_means"]["liquid_flow"] == pytest.approx(0.0445)
    assert result["accepted"][0]["candidate_SI_means"]["liquid_inlet_temperature"] == 298.15
    # A missing block of one instrument cannot borrow another instrument's rows.
    broken = copy.deepcopy(inputs)
    for key in ("times_excel_days", "values_native", "source_rows"):
        del broken["gas_density"][key][10:20]
    assert len(steady_windows(broken)["accepted"]) == 1
    assert (
        steady_windows(broken)["rejected_by_input"]["insufficient_independent_clock_coverage"] == 1
    )
    assert result["selection_uses_thickness_or_model_error"] is False
    assert result["conversion_confirmed"] is False


def test_input_transient_is_not_selected_as_a_steady_operating_point():
    s = series("mass", 160.2)
    s["values_native"][:40] = [10 * i for i in range(40)]
    inputs = {
        "liquid_flow": s,
        "gas_volume_flow": series("gas", 0),
        "gas_density": series("rho", 1.17),
        "gas_viscosity": series("mu", 1.8e-5),
    }
    result = steady_windows(inputs)
    assert result["rejected_by_input"]["liquid_input_not_steady"] == 1
    assert len(result["accepted"]) == 2
    assert window_statistics(s, 45000 + 100 / 86400, 45000 + 120 / 86400) is None


def test_actual_six_cases_keep_unconfirmed_units_and_clock_alignment_visible():
    data = Path(__file__).resolve().parents[1] / "tests/data/surface/fetrig"
    report = run(data)
    assert len(report["cases"]) == 6 and report["experimentally_qualified"] is False
    assert sum(len(r["references"]) for r in report["cases"]) == 31
    for case in report["cases"]:
        assert not case["experimental_pass"]
        assert "cci_labview_clock_alignment_unavailable" in case["qualification_blockers"]
        assert all(
            channel["unit_from_workbook"] is None for channel in case["input_channels"].values()
        )
        if case["cci_member"].endswith("ReG0.csv"):
            lo, hi = case["shear_free_reference_range_m"]
            assert 0.00049 < lo <= hi < 0.00051
        else:
            assert case["shear_free_reference_range_m"] is None
        assert all(not row["within_existing_annular_reynolds_domain"] for row in case["references"])


def test_gas_conversion_is_applied_before_zero_flow_classification():
    from validate_fetrig_film import CANDIDATE_UNITS

    inputs = {
        "liquid_flow": series("mass", 160.2),
        "gas_volume_flow": series("gas", 0.36),
        "gas_density": series("rho", 1.17),
        "gas_viscosity": series("mu", 1.8e-5),
    }
    # Same physical signal in two units, with fluctuations above 2% CV but
    # below the declared near-zero absolute threshold in SI.
    inputs["gas_volume_flow"]["values_native"] = [0.18 + (i % 2) * 0.18 for i in range(121)]
    units = copy.deepcopy(CANDIDATE_UNITS)
    units["gas_volume_flow"]["scale"] = 1 / 3600
    converted = steady_windows(inputs, units=units)
    si = copy.deepcopy(inputs)
    si["gas_volume_flow"]["values_native"] = [
        v / 3600 for v in si["gas_volume_flow"]["values_native"]
    ]
    direct = steady_windows(si)
    assert len(converted["accepted"]) == len(direct["accepted"]) > 0
    for a, b in zip(converted["accepted"], direct["accepted"]):
        assert a["candidate_SI_means"]["gas_volume_flow"] == pytest.approx(
            b["candidate_SI_means"]["gas_volume_flow"]
        )


def test_public_metadata_unit_conflict_is_not_silently_resolved():
    data = Path(__file__).resolve().parents[1] / "tests/data/surface/fetrig"
    report = run(data)
    declarations = report["external_unit_evidence"]["gas_unit_declarations"]
    assert len(declarations) == 2
    assert all(r["unit"] == "m^3/h" for r in declarations)
    case = next(c for c in report["cases"] if "ReG12900.csv" in c["cci_member"])
    audit = case["gas_unit_consistency"]
    assert not audit["automatically_selected_conversion"]
    assert not audit["uses_film_thickness"]
    for row in audit["windows"]:
        assert 10000 < row["reynolds_if_native_m3_s"] < 15000
        assert row["reynolds_if_native_m3_h"] == pytest.approx(
            row["reynolds_if_native_m3_s"] / 3600
        )


def test_new_public_middle_station_covers_all_gas_settings_without_fitting():
    data = Path(__file__).resolve().parents[1] / "tests/data/surface/fetrig_middle"
    report = run(data)
    assert len(report["cases"]) == 6
    assert sum(len(c["references"]) for c in report["cases"]) == 19
    assert {
        c["gas_unit_consistency"]["nominal_filename_reynolds_not_boundary_condition"]
        for c in report["cases"]
    } == {0, 4900, 8900, 12900, 16900, 20900}
    assert not report["fit_performed"] and not report["experimentally_qualified"]
    zero = next(c for c in report["cases"] if c["cci_member"].endswith("ReG0.csv"))
    assert zero["observed_cci"]["go_mean_thickness_m"] == pytest.approx(0.0004996232328111007)
    lo, hi = zero["conditional_reference_relative_difference_range"]
    assert -0.008 < lo < hi < -0.004
