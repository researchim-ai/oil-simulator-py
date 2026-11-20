"""Published velocity convention, conservation and independent steady solution."""

import math

import pytest
import torch
from diagnose_annular_experiment import independent_film

from surface.annular_film import countercurrent_film
from surface.pr_active_pipe_friction import PRActivePipeFriction


@pytest.mark.parametrize("jg,ml", [(5.0, 0.04), (8.0, 0.08), (9.0, 0.10)])
def test_published_velocity_law_and_independent_radial_solution(jg, ml):
    rho = 1.2
    result = countercurrent_film(jg, ml, rho, shear_velocity="superficial")
    alpha, gradient = independent_film(jg, ml, rho, shear_velocity="superficial")
    assert float(result["liquid_holdup"]) == pytest.approx(alpha, abs=1e-10)
    assert float(result["pressure_gradient_pa_m"]) == pytest.approx(gradient, abs=1e-7)
    delta = float(result["film_thickness_m"])
    tau = 0.5 * 0.005 * (1 + 300 * delta / 0.085) * rho * jg**2
    assert float(result["interfacial_shear_pa"]) == pytest.approx(tau, rel=1e-12)
    # Force balance per unit length, upward positive.
    al = math.pi * 0.085**2 / 4 * alpha
    force = -al * (gradient + 998.2 * 9.80665) + float(
        result["liquid_interfacial_force_n_m"] + result["liquid_wall_force_n_m"]
    )
    assert abs(force) < 1e-8


def test_vector_inputs_and_scalar_solutions_agree():
    speeds = torch.tensor([5.0, 8.0, 9.0], dtype=torch.float64)
    masses = torch.tensor([0.04, 0.08, 0.10], dtype=torch.float64)
    batch = countercurrent_film(speeds, masses, 1.2, shear_velocity="superficial")
    for i in range(3):
        scalar = countercurrent_film(speeds[i], masses[i], 1.2, shear_velocity="superficial")
        for key in batch:
            torch.testing.assert_close(batch[key][i], scalar[key], rtol=1e-10, atol=1e-12)


def test_no_silent_low_reynolds_or_hybrid_extension():
    with pytest.raises(ValueError, match="turbulent"):
        countercurrent_film(0.1, 0.04, 1.2, shear_velocity="superficial")
    with pytest.raises(ValueError, match="interface"):
        countercurrent_film(8.0, 0.04, 1.2, shear_velocity="superficial", dry_consistent=True)
    with pytest.raises(ValueError, match="Unknown"):
        PRActivePipeFriction.from_dict({"model": "laminar_annular_wallis_superficial_v1"})


def test_external_report_identifies_correct_reference_law_without_claiming_validation():
    from validate_annular_film_experiment import validate
    from validate_multiphase_experiments import ROOT

    report = validate(
        ROOT / "tests/data/surface/hannover_countercurrent", shear_velocity="superficial"
    )
    assert report["closure"] == "laminar_annular_wallis_superficial_v1"
    assert report["comparison_contract"]["reference_shear_law_reproduced"] is True
    assert report["held_out"] is False
    assert report["accepted_as_validated_for_annular_countercurrent"] is False
    assert report["matched_wet_windows"] == 16
    assert report["roughness_sensitivity"]["0.0"]["mean_absolute_error_pa"] == pytest.approx(
        34.955090001, abs=1e-6
    )


def test_superficial_law_refuses_undeclared_roughness_correction():
    with pytest.raises(ValueError, match="roughness"):
        countercurrent_film(8.0, 0.04, 1.2, shear_velocity="superficial", roughness_m=0.0002)
