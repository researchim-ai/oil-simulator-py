"""Pressure inversion is a diagnostic, never an experimental validation pass."""

import pytest
from diagnose_annular_experiment import implied_interface, pressure_inversion

from surface.annular_film import countercurrent_film


@pytest.mark.parametrize("velocity,flow", [(3.0, 0.03), (5.0, 0.06), (7.0, 0.09)])
@pytest.mark.parametrize("law", ["interface", "superficial"])
def test_independent_inverse_recovers_forward_state(velocity, flow, law):
    rho = 1.2
    forward = countercurrent_film(velocity, flow, rho, shear_velocity=law)
    loss = -float(forward["pressure_gradient_pa_m"]) - rho * 9.80665
    inverse = implied_interface(velocity, flow, rho, loss)
    for name in ("liquid_holdup", "film_thickness_m"):
        assert inverse[name] == pytest.approx(float(forward[name]), rel=1e-9)
    assert inverse["implied_shear_pa"] == pytest.approx(
        float(forward["interfacial_shear_pa"]), rel=1e-9
    )
    assert abs(inverse["liquid_force_residual_n_m"]) < 1e-9
    if law == "superficial":
        assert inverse["implied_to_wallis_ratio"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("loss", [-1, 0, float("nan"), 1e6])
def test_nonphysical_or_unresolved_inverse_is_not_clipped(loss):
    with pytest.raises(ValueError):
        implied_interface(5.0, 0.06, 1.2, loss)


def test_failed_pressure_reconstruction_is_retained_not_used_to_fit():
    raw = [
        {
            "sheet": "synthetic",
            "first_row": 3,
            "last_row": 62,
            "gas_volume_flow_m3_h": 130.0,
            "liquid_mass_flow_kg_s": 0.06,
            "absolute_pressure_pa": 101325.0,
            "temperature_k": 293.15,
        }
    ]
    external = {
        "roughness_sensitivity": {
            "0.0": {
                "roughness_m": 0.0,
                "cases": [{**raw[0], "measured_wet_increment_pa": -10000.0}],
            }
        }
    }
    result = pressure_inversion(raw, external)
    assert len(result["cases"]) == 1
    assert len(result["cases"][0]["outcomes"]) == 3
    assert all(not x["identified_within_assumptions"] for x in result["cases"][0]["outcomes"])
    assert not result["experimental_validation"]
    assert not result["fit_performed"]
    assert not result["replaces_runtime_closure"]


def test_density_scaling_uses_dry_observations_and_preserves_original_gate():
    from diagnose_annular_experiment import dry_density_sensitivity

    # Fixed volume flow and density-proportional loss; target density 1.5.
    dry = [
        {"q": 100.0, "dp": 20.0, "absolute_pressure_pa": 287.05 * 300.0, "temperature_k": 300.0},
        {"q": 200.0, "dp": 40.0, "absolute_pressure_pa": 287.05 * 300.0, "temperature_k": 300.0},
    ]
    wet = {
        "sheet": "synthetic",
        "first_row": 3,
        "last_row": 62,
        "gas_volume_flow_m3_h": 150.0,
        "absolute_pressure_pa": 1.5 * 287.05 * 300.0,
        "temperature_k": 300.0,
        "measured_dry_baseline_pa": 30.0,
    }
    report = dry_density_sensitivity(
        {"dry_baseline": dry, "roughness_sensitivity": {"0.0": {"cases": [wet]}}}
    )
    assert report["cases"][0]["density_scaled_dry_baseline_pa"] == pytest.approx(45.0)
    assert report["cases"][0]["change_in_wet_increment_pa"] == pytest.approx(-15.0)
    assert wet["measured_dry_baseline_pa"] == 30.0
    assert not report["replaces_experimental_gate"]
