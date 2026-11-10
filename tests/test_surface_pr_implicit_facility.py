"""End-state flows must close the actual finite-inventory nonlinear coupling."""

import json

import pytest
import torch
from check_pr_capillary_facility import audit
from test_surface_pr_capillary_facility import setup

from surface.liquid import HydraulicConvergenceError
from surface.pr_capillary_facility import PRCapillaryFacility, PRCapillaryFacilityState


def case(device="cpu", *, maximum=12):
    base, old = setup(device=device, self_loop=True)
    raw = base.to_dict()
    raw["lines"] = {"out": raw["lines"]["out"]}
    raw.update(coupling="implicit", maximum_coupling_iterations=maximum)
    model = PRCapillaryFacility.from_dict(json.loads(json.dumps(raw)), device=device)
    state = model.initialize(
        {"out": dict(old.pipes)["out"]}, dict(old.columns), layer_slots=dict(old.layer_slots)
    )
    return model, state


@pytest.mark.parametrize("normal_relaxation", [False, True])
def test_end_profiles_close_fluxes_and_survive_checkpoint_restart(normal_relaxation):
    model, old = case()
    snapshot = old.to_dict()
    final, record = model.step(old, 1e-7, normal_relaxation=normal_relaxation)
    assert record["coupling"] == "implicit" and record["coupling_converged"]
    assert 1 < record["nonlinear_coupling_iterations"] <= 12
    assert audit(model.to_dict(), old, final, record, normal_relaxation=normal_relaxation)[
        "accepted"
    ]
    assert old.to_dict() == snapshot
    restored = PRCapillaryFacility.from_dict(json.loads(json.dumps(model.to_dict())))
    state = PRCapillaryFacilityState.from_dict(json.loads(json.dumps(old.to_dict())))
    repeated, _ = restored.step(state, 1e-7, normal_relaxation=normal_relaxation)
    assert repeated == final


def test_unconverged_boundary_iteration_is_a_failure_not_an_accepted_balanced_split():
    model, old = case(maximum=1)
    snapshot = old.to_dict()
    with pytest.raises(HydraulicConvergenceError, match="boundary coupling failed"):
        model.step(old, 1e-7)
    assert old.to_dict() == snapshot


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_implicit_heat_cycle_removes_and_recreates_interface_at_both_actual_endpoints(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, state = case(device)
    for power, temperature, expected in ((1e11, 450.0, 1), (-1e11, 160.0, 2)):
        before = state
        heaters = {"A": {"power_w": power, "temperature_k": temperature}}
        state, record = model.step(before, 1e-7, heaters=heaters)
        assert len(dict(state.columns)["A"].phase_roots) == expected
        assert audit(model.to_dict(), before, state, record, heaters=heaters)["accepted"]
        assert record["coupling_converged"] and record["nonlinear_coupling_iterations"] > 1
        # The accepted pipe used the END-profile phase registry, including
        # pure gas during heating and both phases on the cooling interval.
        for boundary in record["pipe_models"]["out"]["boundaries"]:
            assert len(boundary) == expected
