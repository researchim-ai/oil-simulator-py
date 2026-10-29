"""Well topology must survive non-monotone depths and schedule changes."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator.fim_wells import G, FIMWells, connection_order
from simulator import blackoil_fim as bo
from simulator.eclipse_deck import EclipseCase


def test_track_follows_grid_path_instead_of_vertical_depth():
    perfs = [(0, 0, 2), (2, 0, 1), (1, 0, 0), (2, 0, 2)]
    depths = [300., 200., 100., 250.]
    assert connection_order(perfs, depths, (0, 0), 'TRACK') == [0, 2, 1, 3]
    assert connection_order(perfs, depths, (0, 0), 'DEPTH') == [2, 1, 3, 0]
    assert connection_order(perfs, depths, (0, 0), 'INPUT') == [0, 1, 2, 3]
    assert connection_order([], [], (0, 0)) == []


@pytest.mark.parametrize('stopped', [False, True])
def test_flash_hydrostatics_integrates_signed_segments_along_well(monkeypatch, stopped):
    rho = torch.tensor([800., 1000., 100.], dtype=torch.float64)
    v = torch.zeros((bo.N_PROPS, 3), dtype=torch.float64)
    v[bo.I_P] = 1e7
    # Water, then oil, then gas: each connection supplies one surface m3/s.
    v[bo.I_MWW, 0], v[bo.I_MOO, 1], v[bo.I_MGG, 2] = rho[1], rho[0], rho[2]
    fake = SimpleNamespace(
        conn_cell=torch.arange(3), conn_well=torch.zeros(3, dtype=torch.long),
        conn_wi=torch.ones(3), conn_sched=torch.ones(3), rho_sc=rho,
        n_w=1, device=torch.device('cpu'), p_bh=torch.tensor([1e7]),
        conn_head=torch.zeros(3), conn_dz=torch.tensor([100., 300., 200.], dtype=torch.float64),
        wells=[SimpleNamespace(type='producer', stopped=stopped)], well_conn_order=[torch.arange(3)],
        pvt=SimpleNamespace(at=lambda c: None))
    monkeypatch.setattr(bo, 'surface_to_reservoir', lambda pvt, p, qo, qw, qg: (qo, qw, qg))
    heads = bo.BlackOilWells._flash_heads(fake, v, torch.full((3,), 600.))
    # The final segment goes UP 100 m and carries gas alone.
    expected = torch.tensor([100 * 1900 / 3, 100 * 1900 / 3 + 200 * 450,
                             100 * 1900 / 3 + 200 * 450 - 100 * 100], dtype=torch.float64) * G
    if stopped:
        expected = torch.tensor([100., 300., 200.], dtype=torch.float64) * 600 * G
    torch.testing.assert_close(heads, expected)


def test_compord_is_snapshotted_in_schedule(tmp_path):
    source = Path(__file__).parent / 'data/spe9/SPE9_CP.DATA'
    text = source.read_text()
    text = text.replace('SPE9.GRDECL', str((source.parent / 'SPE9.GRDECL').resolve()))
    text = text.replace('PERMVALUES.DATA', "'" + str((source.parent / 'PERMVALUES.DATA').resolve()) + "'")
    pos = text.index('TSTEP', text.index('SCHEDULE'))
    second = text.index('TSTEP', pos + len('TSTEP'))
    text = text[:second] + "COMPORD\n '*' DEPTH /\n/\n" + text[second:]
    text = text[:pos] + "COMPORD\n '*' INPUT /\n/\n" + text[pos:]
    path = tmp_path / 'ordered.DATA'
    path.write_text(text)
    case = EclipseCase(path)
    assert set(case.compord_steps[0].values()) == {'INPUT'}
    assert set(case.compord_steps[-1].values()) == {'DEPTH'}
    assert all(c['present'] or not c['open'] for w in case.wells for c in case.well_connections(w, 0))


def test_future_connections_do_not_change_current_well_path():
    # The future connection is nearest the well head, but is not drilled yet.
    well = SimpleNamespace(i=0, j=0, connection_order='TRACK',
                           perforations=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    fake = SimpleNamespace(wells=[well], ref_depth=[0.],
                           conn_dz=torch.tensor([100., 200., 50.]), device='cpu')
    conns = [{'present': False}, {'present': True}, {'present': True}]
    assert FIMWells._well_connection_order(fake, 0, conns).tolist() == [1, 2]
