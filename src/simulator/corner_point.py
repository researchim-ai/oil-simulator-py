"""
Геометрия угловой (corner-point) сетки Eclipse: COORD / ZCORN / ACTNUM.

Для каждой ячейки считаются объём, центр, размеры (для Писмена) и проводимости TPFA между
соседями по i, j, k. Полупроводимость — как в OPM: T_i = (K·A)·d / |d|², d — вектор от центра
ячейки к центру грани, A — вектор площади грани; T = 1 / (1/T_L + 1/T_R).
Поддерживаются согласованные сетки (без разломов и NNC): несовпадение углов соседних ячеек — ошибка.
Все массивы — numpy с индексами [i, j, k]; z — глубина (вниз положительна).
"""
import numpy as np

# локальная нумерация углов: бит 0 — i, бит 1 — j, бит 2 — k (0 — кровля)
_FACES = {
    "i-": (0, 2, 6, 4), "i+": (1, 3, 7, 5),
    "j-": (0, 1, 5, 4), "j+": (2, 3, 7, 6),
    "k-": (0, 1, 3, 2), "k+": (4, 5, 7, 6),
}


def cell_corners(coord, zcorn, dims):
    """Углы ячеек: массив [nx, ny, nz, 8, 3]."""
    nx, ny, nz = dims
    coord = np.asarray(coord, dtype=np.float64).reshape(ny + 1, nx + 1, 6)
    z = np.asarray(zcorn, dtype=np.float64).reshape(2 * nz, 2 * ny, 2 * nx)
    out = np.empty((nx, ny, nz, 8, 3))
    for c in range(8):
        a, b, kk = c & 1, (c >> 1) & 1, (c >> 2) & 1
        zc = z[kk::2, b::2, a::2].transpose(2, 1, 0)                     # [nx, ny, nz]
        pil = coord[b:b + ny, a:a + nx].transpose(1, 0, 2)[:, :, None, :]  # [nx, ny, 1, 6]
        xt, yt, zt, xb, yb, zb = (pil[..., m] for m in range(6))
        dzp = zb - zt
        frac = np.where(np.abs(dzp) > 1e-12, (zc - zt) / np.where(np.abs(dzp) > 1e-12, dzp, 1.0), 0.0)
        out[..., c, 0] = xt + (xb - xt) * frac
        out[..., c, 1] = yt + (yb - yt) * frac
        out[..., c, 2] = zc
    return out


def _face(corners, name):
    """Центр и вектор площади грани (ориентирован наружу по оси для «+», внутрь для «−»)."""
    p0, p1, p2, p3 = (corners[..., m, :] for m in _FACES[name])
    center = 0.25 * (p0 + p1 + p2 + p3)
    area = 0.5 * np.cross(p2 - p0, p3 - p1)
    return center, area


def _volume(corners, center):
    """Объём шестигранника: разбиение на 24 тетраэдра (центр ячейки, центр грани, ребро грани)."""
    vol = np.zeros(corners.shape[:3])
    for name, ids in _FACES.items():
        fc = 0.25 * sum(corners[..., m, :] for m in ids)
        for e in range(4):
            a = corners[..., ids[e], :] - center
            b = corners[..., ids[(e + 1) % 4], :] - center
            vol += np.abs(np.einsum("...i,...i", np.cross(a, b), fc - center)) / 6.0
    return vol


class CornerPointGrid:
    def __init__(self, coord, zcorn, dims, actnum=None):
        self.dims = tuple(int(v) for v in dims)
        nx, ny, nz = self.dims
        self.corners = cell_corners(coord, zcorn, dims)
        self.center = self.corners.mean(axis=3)
        self.depth = self.center[..., 2]
        self.volume = _volume(self.corners, self.center)
        self.faces = {n: _face(self.corners, n) for n in _FACES}
        self.actnum = (np.ones(self.dims, dtype=bool) if actnum is None
                       else np.asarray(actnum).reshape(nz, ny, nx).transpose(2, 1, 0) > 0)
        self.actnum &= self.volume > 0
        # размеры ячейки как DX/DY/DZ в Eclipse: горизонтальные расстояния между центрами граней i/j, толщина по z
        self.dx = np.linalg.norm((self.faces["i+"][0] - self.faces["i-"][0])[..., :2], axis=-1)
        self.dy = np.linalg.norm((self.faces["j+"][0] - self.faces["j-"][0])[..., :2], axis=-1)
        self.dz = self.faces["k+"][0][..., 2] - self.faces["k-"][0][..., 2]
        self._check_matching()

    def _check_matching(self):
        c = self.corners
        pairs = ((0, (1, 3, 5, 7), (0, 2, 4, 6)), (1, (2, 3, 6, 7), (0, 1, 4, 5)), (2, (4, 5, 6, 7), (0, 1, 2, 3)))
        for axis, hi, lo in pairs:
            if c.shape[axis] < 2:
                continue
            a = np.take(c, range(c.shape[axis] - 1), axis=axis)[..., hi, :]
            b = np.take(c, range(1, c.shape[axis]), axis=axis)[..., lo, :]
            gap = np.abs(a - b).max()
            if gap > 1e-6 * max(1.0, np.abs(c).max()):
                raise NotImplementedError(f"corner-point: несогласованные грани по оси {'ijk'[axis]} "
                                          f"(разломы/NNC пока не поддерживаются), зазор {gap:.3g}")

    def half_trans(self, face, k):
        """Полупроводимость ячеек через грань face при диагональной проницаемости k = (kx, ky, kz)."""
        fc, area = self.faces[face]
        d = fc - self.center
        ka = np.stack([k[0] * area[..., 0], k[1] * area[..., 1], k[2] * area[..., 2]], axis=-1)
        return np.abs(np.einsum("...i,...i", ka, d)) / np.maximum(np.einsum("...i,...i", d, d), 1e-300)

    def transmissibility(self, kx, ky, kz):
        """(Tx [nx-1,ny,nz], Ty [nx,ny-1,nz], Tz [nx,ny,nz-1]) в единицах k·длина."""
        k = (kx, ky, kz)
        out = []
        for axis, (lo, hi) in enumerate((("i-", "i+"), ("j-", "j+"), ("k-", "k+"))):
            n = self.dims[axis]
            tl = np.take(self.half_trans(hi, k), range(n - 1), axis=axis)
            tr = np.take(self.half_trans(lo, k), range(1, n), axis=axis)
            act = np.take(self.actnum, range(n - 1), axis=axis) & np.take(self.actnum, range(1, n), axis=axis)
            with np.errstate(divide="ignore"):
                t = np.where(act & (tl > 0) & (tr > 0), 1.0 / (1.0 / np.maximum(tl, 1e-300) + 1.0 / np.maximum(tr, 1e-300)), 0.0)
            out.append(t)
        return tuple(out)
