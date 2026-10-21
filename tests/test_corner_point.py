import os
import sys

import numpy as np

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
