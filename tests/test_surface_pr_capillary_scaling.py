"""Sparse capillary corrections must retain the full physical equations."""

import json

import numpy as np
import pytest
import torch
from check_pr_capillary_axial import audit as axial_audit
from check_pr_capillary_pressure import audit as normal_audit
from test_surface_pr_wetting_coupling import setup

from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_linear import CapillaryLinearSystem
from surface.pr_capillary_pressure import PRCapillaryPressure


def test_sparse_normal_and_axial_match_dense_and_independent_equations():
    source, old = setup()
    parameters = {k: v for k, v in source.to_dict().items() if k not in ("model", "pipe_model_id")}
    sparse = PRCapillaryPressure(source.pipe, **parameters, linear_solver="sparse")
    expected, _ = source.apply(old, 0.001)
    actual, record = sparse.apply(old, 0.001)
    assert normal_audit(sparse, old, actual, record)["accepted"]
    assert record["linear_solver"]["backend"] == "torch_colored_sparse_gmres"
    for ca, cb in zip(expected.cells, actual.cells):
        for (_, a), (_, b) in zip(ca, cb):
            assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-10)
            assert a.volume_m3 == pytest.approx(b.volume_m3, abs=1e-25)
    models = [
        PRCapillaryAxialPipe(source, volume_diffusion="none", linear_solver=mode)
        for mode in ("dense", "sparse")
    ]
    states = []
    for model in models:
        end, r = model.step(old, 1e-12)
        assert axial_audit(model, old, end, r)["accepted"]
        states.append(end)
        assert (
            PRCapillaryAxialPipe.from_dict(json.loads(json.dumps(model.to_dict()))).to_dict()
            == model.to_dict()
        )
    for ca, cb in zip(states[0].cells, states[1].cells):
        for (_, a), (_, b) in zip(ca, cb):
            np.testing.assert_allclose(a.component_moles, b.component_moles, rtol=1e-12, atol=1e-25)
            assert a.temperature_k == pytest.approx(b.temperature_k, abs=1e-10)


def test_sparse_true_jvp_check_rejects_an_omitted_nonlocal_dependency():
    x = torch.zeros(4, dtype=torch.float64)
    matrix = torch.eye(4, dtype=x.dtype)
    matrix[0, 3] = 0.2
    function = lambda y: matrix @ y - 1
    solver = CapillaryLinearSystem([[i] for i in range(4)], [], x.device, radius=1, mode="sparse")
    with pytest.raises(RuntimeError, match="true-residual"):
        solver.solve(function, x, function(x))


def test_large_line_has_linear_geometry_and_sparse_storage_without_dense_jacfwd(monkeypatch):
    from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe

    base, seed = setup()
    count = 64
    raw = base.pipe.to_dict()
    raw["lengths_m"] = [base.pipe.lengths[0]] * count
    pipe = PRActiveMultifluidPipe.from_dict(raw)
    parameters = {k: v for k, v in base.to_dict().items() if k not in ("model", "pipe_model_id")}
    parameters["maximum_unknowns"] = 5 * count
    source = PRCapillaryPressure(pipe, **parameters)
    # A repeating physical pattern excites transport and capillary coupling.
    old = pipe.initialize([dict(seed.cells[i % 4]) for i in range(count)])
    model = PRCapillaryAxialPipe(source, maximum_unknowns=9 * count, volume_diffusion="none")

    def forbidden(*_a, **_k):
        raise AssertionError("A large capillary solve allocated a dense Jacobian")

    monkeypatch.setattr(torch.func, "jacfwd", forbidden)
    monkeypatch.setattr(PRCapillaryPressure, "incidence", property(forbidden))
    end, report = model.step(old, 1e-12)
    assert axial_audit(model, old, end, report)["accepted"]
    stats = report["linear_solver"]
    assert stats["backend"] == "torch_colored_sparse_gmres"
    assert stats["stored_jacobian_entries"] <= count * 5 * 9**2
    assert stats["stored_jacobian_entries"] < stats["unknowns"] ** 2 / 10
    assert stats["colors"] <= 9 * 9
    assert source.edge_left.numel() == count
    # The normal split must also stay sparse and reuse its compiled graph.
    end, normal = source.apply(old, 1e-12)
    assert normal_audit(source, old, end, normal)["accepted"]
    assert normal["linear_solver"]["backend"] == "torch_colored_sparse_gmres"


def test_sparse_finite_junction_graph_keeps_cross_port_derivatives():
    from check_pr_capillary_branch import audit
    from test_surface_pr_capillary_branch import setup as branch_case

    from surface.pr_active_multifluid_pipe import PRActiveMultifluidState
    from surface.pr_capillary_branch import PRCapillaryBranchNetwork

    base, state = branch_case()
    model = PRCapillaryBranchNetwork(
        base.pipes,
        base.junction_volumes,
        {**base.parameters, "linear_solver": "sparse"},
        linear_solver="sparse",
    )
    old = model.initialize(dict(state.pipes), {n: dict(c) for n, c in state.junctions})
    _end, report = model.advance(old, 1e-12, initial_step_s=1e-12)
    step = report["advance"]["steps"][0]
    axial = PRActiveMultifluidState.from_dict(step["axial"]["state"])
    final = PRActiveMultifluidState.from_dict(step["normal"]["state"])
    assert step["axial"]["linear_solver"]["backend"] == "torch_colored_sparse_gmres"
    assert audit(model, model._view(old), axial, step["axial"])["accepted"]
    assert normal_audit(model.source, axial, final, step["normal"])["accepted"]
