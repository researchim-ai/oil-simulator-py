"""Physical node/two-stage blocks preserve full graph equations and wall stocks."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from audit_pr_collocation import audit_gauss_step
from check_pr_collocation_walls import wall_fixture

from surface import PRCollocation, PREquilibriumTransportNetwork
from surface.pr_collocation import TorchNewton
from surface.torch_linear import BlockJacobi


def test_node_wall_and_stage_partition_and_bounded_local_factors():
    blocks = PRCollocation._physical_blocks([0, 4, 11], [1], 12, 2)
    assert blocks == [
        list(range(4)) + list(range(12, 16)),
        list(range(4, 12)) + list(range(16, 24)),
    ]
    assert sorted(i for b in blocks for i in b) == list(range(24))
    many = PRCollocation._physical_blocks([0, 200, 500], [1], 501, 2)
    assert max(map(len, many)) <= 256
    assert sorted(i for b in many for i in b) == list(range(1002))


@pytest.mark.parametrize("scheme", ["gauss4", "radau3"])
@pytest.mark.parametrize("partial", [False, True])
def test_real_stage_and_endpoint_blocks_match_dense_and_independent_balances(
    scheme, partial, monkeypatch
):
    model, state = wall_fixture(dense_limit=1, partial=partial)
    solver = PRCollocation(model, scheme=scheme, phase_fallback=False)
    original = TorchNewton.solve
    checked = []

    def observed(self, fn, initial, *args, **kwargs):
        blocks = kwargs["blocks"]
        assert sorted(i for b in blocks for i in b) == list(range(initial.numel()))
        assert len(blocks) == len(model.nodes)
        jac = torch.func.jacfwd(fn)(initial)
        pattern = kwargs["pattern"]
        values = jac[pattern[0], pattern[1]]
        preconditioner = BlockJacobi(*pattern, blocks, initial.numel(), initial.device)
        preconditioner.update(values)
        rhs = torch.linspace(0.1, 1.0, initial.numel(), dtype=initial.dtype)
        actual = preconditioner(rhs)
        for b in blocks:
            expected = torch.linalg.solve(jac[b][:, b], rhs[b])
            torch.testing.assert_close(actual[b], expected, atol=2e-11, rtol=2e-11)
        result = original(self, fn, initial, *args, **kwargs)
        assert self.block_preconditioner is not None
        checked.append(len(blocks[0]))
        return result

    monkeypatch.setattr(TorchNewton, "solve", observed)
    args = (0.02, model.default_openings, [0.0, 0.0], [320.0, 360.0], {})
    final, record = solver._step(state, *args)
    assert len(checked) == (2 if scheme == "gauss4" else 1)
    assert checked[0] > 8  # Contains BOTH stages and wall unknown, not scalar Jacobi.
    assert audit_gauss_step(
        model.to_dict(),
        state.to_dict(),
        record,
        model.fluid,
        {"openings": model.default_openings},
    )["accepted"]
    monkeypatch.setattr(TorchNewton, "solve", original)
    dense = PREquilibriumTransportNetwork.from_dict({**model.to_dict(), "dense_limit": 128})
    reference, _ = PRCollocation(dense, scheme=scheme, phase_fallback=False)._step(
        replace(state, model_id=dense.model_id),
        0.02,
        dense.default_openings,
        [0.0, 0.0],
        [320.0, 360.0],
        {},
    )
    np.testing.assert_allclose(
        [s.temperature_k for _, s in final.nodes],
        [s.temperature_k for _, s in reference.nodes],
        atol=1e-9,
        rtol=0,
    )
    for (_, a), (_, b) in zip(final.nodes, reference.nodes):
        if a.wall_temperature_k is not None:
            assert abs(a.wall_temperature_k - b.wall_temperature_k) < 1e-9
    assert max(map(abs, record["conservation_residuals"])) < 1e-7


def test_diagnostic_diagonal_mode_keeps_full_sparse_operator(monkeypatch):
    model, state = wall_fixture(dense_limit=1)
    model._collocation_block_preconditioning = False
    solver = PRCollocation(model, scheme="radau3", phase_fallback=False)
    original = TorchNewton.solve
    seen = []

    def observed(self, *args, **kwargs):
        assert kwargs["blocks"] is None
        result = original(self, *args, **kwargs)
        assert self.block_preconditioner is None and self.matrix is not None
        seen.append(self.matrix.values.numel())
        return result

    monkeypatch.setattr(TorchNewton, "solve", observed)
    solver._step(state, 0.02, model.default_openings, [0.0, 0.0], [320.0, 360.0], {})
    assert seen and seen[0] > len(state.nodes) * 10


def test_singular_local_factor_keeps_valid_global_sparse_solution():
    model, _state = wall_fixture(dense_limit=1)
    solver = PRCollocation(model, scheme="radau3")
    matrix = model.fluid.tensor(
        [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]
    )
    target = matrix @ torch.ones(4, dtype=matrix.dtype)

    def fn(x):
        return matrix @ x - target

    fn.linear_blocks = [[0, 1], [2, 3]]
    pattern = torch.nonzero(matrix).T.tolist()
    point = matrix.new_zeros(4)
    root, _, residual = solver._solve(
        fn, point, point - 10, point + 10, pattern, point.new_full((4,), float("inf"))
    )
    torch.testing.assert_close(root, torch.ones_like(root), atol=1e-11, rtol=0)
    assert residual < 3e-12


def test_block_setup_cannot_mask_false_convergence_status(monkeypatch):
    from surface.liquid import HydraulicConvergenceError

    model, _ = wall_fixture(dense_limit=1)
    solver = PRCollocation(model)

    def fn(x):
        return x - 1.0

    fn.linear_blocks = [[0, 1], [2, 3]]

    def false_success(self, function, point, *args, **kwargs):
        return point, torch.zeros_like(point), point.new_tensor(True, dtype=torch.bool), 0

    monkeypatch.setattr(TorchNewton, "solve", false_success)
    point = model.fluid.tensor([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(HydraulicConvergenceError, match="solve rejected"):
        solver._solve(
            fn,
            point,
            point - 10,
            point + 10,
            ([0, 1, 2, 3], [0, 1, 2, 3]),
            point.new_full((4,), float("inf")),
        )
