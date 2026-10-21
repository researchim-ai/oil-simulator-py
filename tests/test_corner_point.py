import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.corner_point import CornerPointGrid  # noqa: E402


def _box_grid(dx, dy, dz, top=1000.0, dip=0.0):
    """COORD/ZCORN прямоугольной сетки (вертикальные столбы), опционально с наклоном слоёв по x."""
    nx, ny, nz = len(dx), len(dy), len(dz)
    xn, yn = np.concatenate([[0], np.cumsum(dx)]), np.concatenate([[0], np.cumsum(dy)])
    zn = top + np.concatenate([[0], np.cumsum(dz)])
    coord = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord += [xn[i], yn[j], zn[0], xn[i], yn[j], zn[-1] + 100.0 * (1 + dip)]
    zc = np.zeros((2 * nz, 2 * ny, 2 * nx))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for a in (0, 1):
                    shift = dip * xn[i + a]
                    zc[2 * k, 2 * j:2 * j + 2, 2 * i + a] = zn[k] + shift
                    zc[2 * k + 1, 2 * j:2 * j + 2, 2 * i + a] = zn[k + 1] + shift
    return np.array(coord), zc.reshape(-1)


def test_box_grid_matches_cartesian_tpfa():
    dx, dy, dz = np.array([10.0, 20.0, 15.0]), np.array([5.0, 8.0]), np.array([2.0, 3.0, 4.0, 1.0])
    rng = np.random.default_rng(0)
    shape = (3, 2, 4)
    kx, ky, kz = (rng.uniform(1, 100, shape) for _ in range(3))
    g = CornerPointGrid(*_box_grid(dx, dy, dz), shape)
    assert np.allclose(g.volume, dx[:, None, None] * dy[None, :, None] * dz[None, None, :])
    assert np.allclose(g.dx, dx[:, None, None]) and np.allclose(g.dz, dz[None, None, :])
    tx, ty, tz = g.transmissibility(kx, ky, kz)
    a = dy[None, :, None] * dz[None, None, :]
    ref_x = a / (dx[:-1, None, None] / (2 * kx[:-1]) + dx[1:, None, None] / (2 * kx[1:]))
    a = dx[:, None, None] * dz[None, None, :]
    ref_y = a / (dy[None, :-1, None] / (2 * ky[:, :-1]) + dy[None, 1:, None] / (2 * ky[:, 1:]))
    a = dx[:, None, None] * dy[None, :, None]
    ref_z = a / (dz[None, None, :-1] / (2 * kz[:, :, :-1]) + dz[None, None, 1:] / (2 * kz[:, :, 1:]))
    assert np.allclose(tx, ref_x) and np.allclose(ty, ref_y) and np.allclose(tz, ref_z)


def test_dipping_grid_volume_and_depth():
    dx, dy, dz = np.full(4, 100.0), np.full(2, 50.0), np.full(3, 10.0)
    g = CornerPointGrid(*_box_grid(dx, dy, dz, dip=0.1), (4, 2, 3))
    # сдвиг столбцов по вертикали сохраняет объём; глубина центра растёт на dip·Δx
    assert np.allclose(g.volume, 100.0 * 50.0 * 10.0)
    assert np.allclose(np.diff(g.depth, axis=0), 10.0)
    tx, _, _ = g.transmissibility(np.ones((4, 2, 3)), np.ones((4, 2, 3)), np.ones((4, 2, 3)))
    # T по x: A·d/|d|² с d = (50, 0, 5) — меньше, чем на горизонтальной сетке
    half = 50.0 * 10.0 * 50.0 / (50.0 ** 2 + 5.0 ** 2)
    assert np.allclose(tx, half / 2)


def test_fault_throw_gives_nnc_with_conserved_area():
    dx, dy, dz = np.full(2, 100.0), np.full(1, 50.0), np.full(4, 10.0)
    coord, zc = _box_grid(dx, dy, dz)
    zc = zc.reshape(8, 2, 4)
    zc[:, :, 2:] += 4.0                       # сброс 4 м по правому блоку
    g = CornerPointGrid(coord, zc.reshape(-1), (2, 1, 4))
    one = np.ones((2, 1, 4))
    c = g.connections(one, one, one)
    x = c["axis"] == 0
    # 4 прямых (k→k) + 3 NNC (k→k−1); через каждую грань перекрытие 6 м и 4 м
    assert x.sum() == 7 and c["nnc"][x].sum() == 3
    assert np.all(c["c2"][x & c["nnc"]] % 4 == c["c1"][x & c["nnc"]] % 4 - 1)
    t_full = g.transmissibility(one, one, one)[0][0, 0, 0]
    assert np.isclose(c["T"][x].sum(), 4 * t_full - 0.4 * t_full)   # нижние 4 м блока A ни с чем не граничат


NORNE = os.environ.get("NORNE_DIR", "/tmp/ot2/norne")


@pytest.mark.skipif(not os.path.exists(os.path.join(NORNE, "ECL.2014.2", "NORNE_ATW2013.INIT")),
                    reason="нет данных Norne (OPM opm-tests); задайте NORNE_DIR")
def test_norne_static_model_matches_eclipse():
    from simulator.ecl_io import read_ecl_arrays
    from simulator.eclipse_deck import EclipseCase
    case = EclipseCase(os.path.join(NORNE, "NORNE_ATW2013.DATA"), grid_only=True)
    s = case.static_model()
    init = dict(read_ecl_arrays(os.path.join(NORNE, "ECL.2014.2", "NORNE_ATW2013.INIT")))
    assert s["active"].sum() == (init["PORV"] > 0).sum() == 44431
    nx, ny, nz = case.dims
    gidx = np.arange(nx * ny * nz).reshape(nz, ny, nx).transpose(2, 1, 0).reshape(-1)
    g2a = -np.ones(nx * ny * nz, dtype=np.int64)
    g2a[np.flatnonzero(init["PORV"] > 0)] = np.arange(int((init["PORV"] > 0).sum()))
    c = s["conn"]
    T = c["T"] * 0.00852702 / 9.869233e-16
    for ax, name in enumerate(("TRANX", "TRANY", "TRANZ")):
        ref = init[name].astype(np.float64)
        ours = np.zeros_like(ref)
        sel = (c["axis"] == ax) & ~c["nnc"]
        np.add.at(ours, g2a[gidx[c["c1"][sel]]], T[sel])
        big = ref > 1e-6 * ref.max()
        rel = np.abs(ours[big] / ref[big] - 1)
        assert np.median(rel) < 1e-5 and np.mean(rel > 1e-3) < 0.02, name
    assert c["nnc"].sum() >= len(init["TRANNNC"])
    assert np.isclose(T[c["nnc"]].sum(), init["TRANNNC"].astype(np.float64).sum(), rtol=1e-3)
