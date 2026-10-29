"""TOLCRIT truncates table samples before interpolation and endpoint extraction."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator.eclipse_deck import EclipseCase
from simulator.endpoint_scaling import table_endpoints
from simulator.blackoil_props import ThreePhaseRelPerm


@pytest.mark.parametrize('tol', [None, 0., 1e-6, 2e-6])
def test_tolcrit_applies_before_interpolation_and_critical_endpoints(tmp_path, tol):
    source = Path(__file__).parent / 'data/spe1/SPE1CASE1.DATA'
    props = '''SWOF
 .1 0 1 0
 .2 1e-6 .6 0
 .3 2e-6 1e-6 0
 .9 1 0 0 /
SGOF
 0 0 1 0
 .1 1e-6 .5 0
 .2 2e-6 1e-6 0
 .8 1 0 0 /
'''
    if tol is not None:
        props += f'TOLCRIT\n {tol} /\n'
    path = tmp_path / 'tolcrit.DATA'
    path.write_text(source.read_text().replace('\nSOLUTION\n', '\n' + props + 'SOLUTION\n'))
    case = EclipseCase(path)
    effective = 1e-6 if tol is None else tol
    tables = {key: case.region_tables[key][0] for key in ('swof', 'sgof')}
    expected = .5 * sum(v if v > effective else 0. for v in (1e-6, 2e-6))
    kr = ThreePhaseRelPerm(tables=tables, device=torch.device('cpu'))
    krw = kr(torch.tensor([.25], dtype=torch.float64), torch.zeros(1, dtype=torch.float64))[0]
    assert float(krw) == pytest.approx(expected, abs=1e-15)
    ep = table_endpoints(tables['swof'], tables['sgof'])
    assert ep['SWCR'] == pytest.approx(.1 if effective == 0 else (.3 if effective >= 2e-6 else .2))
    assert ep['SGCR'] == pytest.approx(0 if effective == 0 else (.2 if effective >= 2e-6 else .1))
    assert tables['swof']['pcow'] == [0., 0., 0., 0.]
