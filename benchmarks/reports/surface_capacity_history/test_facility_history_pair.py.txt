"""History optimization must not hide changes to physics or execution setup."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from check_facility_history_pair import compare

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/reports/surface_capacity_history'


@pytest.fixture
def pair(tmp_path):
    for name in ('short_False','quadratic_short'):
        shutil.copytree(ROOT/name,tmp_path/name)
    return tmp_path/'short_False',tmp_path/'quadratic_short'


def test_real_short_history_pair_passes_and_reduces_work(pair):
    result=compare(*pair,1.)
    assert result['accepted'] and result['fewer_fim_solves']


@pytest.mark.parametrize('change',['physics','flag'])
def test_only_boolean_history_switch_may_differ(pair,change):
    target=pair[1]/'trajectory.json.gz'
    with gzip.open(target,'rt') as f:bundle=json.load(f)
    if change=='flag':
        bundle['config']['simulation']['coupled_facility']['control']['reuse_capacity_history']=1
    else:
        bundle['config']['simulation']['max_time_step_days'] *= 2
    with gzip.open(target,'wt') as f:json.dump(bundle,f)
    summary=pair[1]/'summary.json';s=json.loads(summary.read_text())
    s['trajectory_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    summary.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Only enabling'):
        compare(*pair,1.)


def test_wrong_device_or_failed_run_cannot_qualify(pair):
    p=pair[1]/'protocol.json';s=json.loads(p.read_text());s['cuda_visible_devices']='different'
    p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Identical GPU'):
        compare(*pair,1.)
    (pair[1]/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='Failed forecast'):
        compare(*pair,1.)


def test_changed_pvt_at_same_path_cannot_qualify(pair):
    protocol=pair[1]/'protocol.json';data=json.loads(protocol.read_text())
    with gzip.open(pair[1]/'trajectory.json.gz','rt') as f:bundle=json.load(f)
    data['source_sha256'][str(Path(bundle['config']['fluid']['pvt_path']).resolve())]='changed'
    protocol.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='PVT contents'):
        compare(*pair,1.)
