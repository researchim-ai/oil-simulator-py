"""Engineering targets do not authorize changing physics or the mass audit."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from check_facility_tolerance_pair import compare, policy_config

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'benchmarks/reports/surface_engineering_forecast'
BUDGETS=json.loads((ROOT/'benchmarks/reports/surface_forecast_profiles/budgets.json').read_text())


@pytest.fixture
def pair(tmp_path):
    for name in ('strict_short','continuation_short'):shutil.copytree(DATA/name,tmp_path/name)
    return tmp_path/'strict_short',tmp_path/'continuation_short'


def test_real_pair_passes_original_engineering_targets(pair):
    r=compare(*pair,BUDGETS,1.)
    assert r['accepted'] and r['work'][1]['fim']<r['work'][0]['fim']


@pytest.mark.parametrize('value',[True,0,-1,float('nan'),float('inf')])
def test_invalid_tolerance_is_rejected(value):
    c=json.loads((ROOT/'configs/fim_blackoil_facility_depletion_forecast.json').read_text())
    c['simulation']['newton_tol_cnv']=value
    with pytest.raises(ValueError,match='Invalid nonlinear'):policy_config(c,False)


def test_physical_change_is_not_numerical_policy(pair):
    p=pair[1]/'trajectory.json.gz'
    with gzip.open(p,'rt') as f:b=json.load(f)
    b['config']['reservoir']['dimensions'][0]+=1
    with gzip.open(p,'wt') as f:json.dump(b,f)
    s=pair[1]/'summary.json';summary=json.loads(s.read_text())
    summary['trajectory_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();s.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Only declared'):compare(*pair,BUDGETS,1.)


def test_wrong_pvt_and_failed_run_are_rejected(pair):
    p=pair[1]/'protocol.json';protocol=json.loads(p.read_text())
    with gzip.open(pair[1]/'trajectory.json.gz','rt') as f:b=json.load(f)
    protocol['source_sha256'][str(Path(b['config']['fluid']['pvt_path']).resolve())]='changed'
    p.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='PVT contents'):compare(*pair,BUDGETS,1.)
    (pair[1]/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='Failed forecast'):compare(*pair,BUDGETS,1.)
