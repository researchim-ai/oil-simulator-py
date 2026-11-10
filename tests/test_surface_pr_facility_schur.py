"""Joint column Newton tangents and independently audited physical roots."""

import json
from pathlib import Path

import pytest
import torch
from run_pr_capillary_facility import run
from test_surface_pr_capillary_facility import setup

from surface.liquid import HydraulicConvergenceError
from surface.pr_capillary_facility import PRCapillaryFacility, PRCapillaryFacilityState
from surface.pr_facility_schur import FacilitySchurProblem


def case(device="cpu", *, two_columns=False, budget=96, iterations=12):
    base, old = setup(device=device, self_loop=not two_columns)
    raw = base.to_dict()
    raw["lines"] = {"out": raw["lines"]["out"]}
    raw.update(
        coupling="newton", maximum_schur_unknowns=budget, maximum_coupling_iterations=iterations
    )
    model = PRCapillaryFacility.from_dict(json.loads(json.dumps(raw)), device=device)
    return model, model.initialize(
        {"out": dict(old.pipes)["out"]}, dict(old.columns), layer_slots=dict(old.layer_slots)
    )


def test_joint_schur_matrix_contains_cross_column_derivatives_and_matches_another_direction():
    model, state = case(two_columns=True)
    problem = FacilitySchurProblem(model, state, 2e-6, True, None)
    point = torch.zeros_like(problem.reference)
    residual = problem.evaluate(point)[0]
    matrix = problem.jacobian(point, residual)
    assert matrix.shape == (6, 6)
    assert matrix[:3, 3:].abs().max() > 1e-8
    assert matrix[3:, :3].abs().max() > 1e-8
    direction = torch.linspace(-0.7, 0.8, 6, dtype=point.dtype)
    h = 2e-5
    independent = (
        problem.evaluate(point + h * direction)[0] - problem.evaluate(point - h * direction)[0]
    ) / (2 * h)
    torch.testing.assert_close(matrix @ direction, independent, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_newton_matrix_is_actually_solved_during_interface_disappearance(device, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = json.loads(
        (
            Path(__file__).parents[1] / "configs/surface_pr_capillary_facility_newton_cycle.json"
        ).read_text()
    )
    snapshot = json.loads(json.dumps(config))
    result = run(config, device=device)
    assert config == snapshot
    assert len(result["independent_audits"]) == 2
    for interval, expected in zip(result["intervals"], (1, 2)):
        assert not interval["result"]["rejected"]
        for report in interval["result"]["steps"]:
            assert len(report["state"]["columns"]["A"]["phase_roots"]) == expected
            assert report["coupling"] == "newton" and report["coupling_converged"]
            assert report["schur_newton"]["jacobian_builds"] >= 1
            assert report["schur_newton"]["linear_solves"] >= 1
            assert report["schur_newton"]["matrix_device"].startswith(device)
    (tmp_path / ("newton-cycle-" + device + ".json")).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    restored = PRCapillaryFacility.from_dict(config["system"], device=device)
    checkpoint = PRCapillaryFacilityState.from_dict(result["final"])
    restored.inspect(checkpoint)


def test_declared_matrix_budget_and_unconverged_newton_do_not_modify_stocks():
    model, old = case(budget=2)
    snapshot = old.to_dict()
    with pytest.raises(ValueError, match="Schur unknowns exceed"):
        model.step(old, 1e-7)
    assert old.to_dict() == snapshot
    model, old = case(iterations=1)
    snapshot = old.to_dict()
    with pytest.raises(HydraulicConvergenceError, match="iteration budget"):
        model.step(old, 1e-7, heaters={"A": {"power_w": 1e11, "temperature_k": 450.0}})
    assert old.to_dict() == snapshot
