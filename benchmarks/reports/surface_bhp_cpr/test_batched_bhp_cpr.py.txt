"""Device-side BHP seeding and current-solve CPR structure reuse."""
from types import SimpleNamespace

import pytest
import torch

from simulator.blackoil_fim import I_P
from simulator.blackoil_surface_coupling import SurfaceCoupledBlackOilWells
from simulator.cpr_gpu import CPRPreconditionerGPU


@pytest.mark.parametrize('previous', [None, [2., -1., 11.], [float('nan'), 6., float('inf')]])
def test_batched_bhp_matches_scalar_minimum_and_strict_warm_start(previous):
    owners = torch.tensor([2, 0, 1, 0, 2])
    cells = torch.tensor([4, 2, 0, 3, 1])
    v = torch.zeros((20, 5), dtype=torch.float64)
    v[I_P] = torch.tensor([6., 13., 5., 3., 12.])
    heads = torch.tensor([1., .5, 0., 1., 2.], dtype=torch.float64)
    prior = None if previous is None else torch.tensor(previous, dtype=torch.float64)
    obj = SimpleNamespace(p_bh=torch.zeros(3, dtype=torch.float64), n_w=3,
                          conn_well=owners, conn_cell=cells, conn_head=heads)
    expected = []
    for i in range(3):
        top = float((v[I_P][cells[owners == i]]-heads[owners == i]).min())
        p = None if prior is None else float(prior[i])
        expected.append(p if p is not None and 0 < p < top else .95*top)
    result = SurfaceCoupledBlackOilWells._previous_surface_bhp(obj, v, prior)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
    assert result.data_ptr() != obj.p_bh.data_ptr()


def test_cpr_shares_only_structure_and_uses_current_matrix_values():
    crow = torch.tensor([0, 2, 4]); col = torch.tensor([0, 1, 0, 1])
    rows = torch.tensor([0, 0, 1, 1]); diag = torch.tensor([0, 3])
    eye = torch.eye(3, dtype=torch.float64)
    rhs = torch.arange(1., 7., dtype=torch.float64)
    for factor in (1., 2.):
        values = torch.stack([eye*4*factor, -eye, -eye, eye*3*factor])
        reference = CPRPreconditionerGPU(values, crow, col, 2, torch.device('cpu'))
        shared = CPRPreconditionerGPU(values, crow, col, 2, torch.device('cpu'),
                                      matrix_rows=rows, diag_positions=diag)
        assert shared.rows is rows
        torch.testing.assert_close(shared.inv_diag, reference.inv_diag, rtol=0, atol=0)
        torch.testing.assert_close(shared.matvec(rhs), reference.matvec(rhs), rtol=0, atol=0)
