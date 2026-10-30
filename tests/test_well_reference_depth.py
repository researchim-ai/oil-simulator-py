"""WELSPECS pressure datum must not depend on future completions."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from simulator.eclipse_deck import EclipseCase
from simulator.simulation import Simulator


def _case(tmp_path, depth='1*', mode='INPUT', first_status='OPEN'):
    text=(Path(__file__).parent/'data/spe1/SPE1CASE1.DATA').read_text()
    text=text.replace("10\t10\t8400\t'OIL'",f"10\t10\t{depth}\t'OIL'")
    original="'PROD'\t10\t10\t3\t3\t'OPEN'\t1*\t1*\t0.5 /"
    assert original in text
    text=text.replace(original,f"'PROD' 10 10 3 3 '{first_status}' 1* 1* 0.5 /\n 'PROD' 10 10 2 2 'OPEN' 1* 1* 0.5 /")
    pos=text.index('\nCOMPDAT',text.index('SCHEDULE'))
    text=text[:pos]+f"\nCOMPORD\n 'PROD' {mode} /\n/\n"+text[pos:]
    text=text[:text.index('\nTSTEP',text.index('SCHEDULE'))]+"""
TSTEP
 1 /
COMPDAT
 'PROD' 10 10 1 1 'OPEN' 1* 1* 0.5 /
/
COMPORD
 'PROD' DEPTH /
/
TSTEP
 1 /
END
"""
    p=tmp_path/'DATUM.DATA';p.write_text(text)
    case=EclipseCase(p)
    res,fluid,wells,params=case.build(device=torch.device('cpu'))
    return case,Simulator(res,fluid,wells,params,device=torch.device('cpu'))


@pytest.mark.parametrize('depth',['1*','-1'])
@pytest.mark.parametrize('mode,layer',[('INPUT',2),('TRACK',1),('DEPTH',1)])
def test_default_datum_uses_first_initial_connection(tmp_path,depth,mode,layer):
    case,sim=_case(tmp_path,depth,mode)
    expected=float(case.cell_depth_abs[9,9,layer])-case.depth_top
    i=next(i for i,w in enumerate(sim.fim_solver.wells.wells) if w.name=='PROD')
    assert sim.fim_solver.wells.ref_depth[i]==pytest.approx(expected)
    for step in [0,1]:
        cfg=next(c for c in case.well_configs(step) if c['name']=='PROD')
        assert cfg['ref_depth']==pytest.approx(expected)
    m=sim.fim_solver.wells.conn_well==i
    cells=sim.fim_solver.wells.conn_cell[m].tolist()
    pos=cells.index((9*10+9)*3+layer)
    assert float(sim.fim_solver.wells.conn_dz[m][pos])==pytest.approx(0.)


def test_closed_but_defined_first_connection_sets_datum(tmp_path):
    case,sim=_case(tmp_path,first_status='SHUT')
    cfg=next(c for c in case.well_configs() if c['name']=='PROD')
    assert not cfg['connections'][0]['open']
    assert cfg['ref_depth']==pytest.approx(float(case.cell_depth_abs[9,9,2])-case.depth_top)


def test_explicit_welspecs_depth_is_preserved(tmp_path):
    case,sim=_case(tmp_path,depth='8500')
    cfg=next(c for c in case.well_configs() if c['name']=='PROD')
    assert cfg['ref_depth']==pytest.approx(8500*.3048-case.depth_top)
