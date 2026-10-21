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


def _half(k, area, d):
    """Полупроводимость k·|A·d| / |d|², k — проницаемость по направлению грани (как Eclipse NEWTRAN / OPM)."""
    return k * np.abs(np.einsum("...i,...i", area, d)) / np.maximum(np.einsum("...i,...i", d, d), 1e-300)


def _overlap(ta0, ta1, ba0, ba1, tb0, tb1, bb0, bb1):
    """
    Перекрытие двух граней на общей паре столбов в координатах (s ∈ [0,1], z): кровля/подошва линейны по s.
    Возвращает (∫ h⁺ ds, s̄, z̄) — «площадь» в единицах z и центр перекрытия; h = min(подошв) − max(кровель).
    Интегрирование точное: [0,1] режется в точках пересечения кровель, подошв и в нулях h.
    """
    def cross(a0, a1, b0, b1):
        den = (a1 - a0) - (b1 - b0)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(den) > 1e-300, (b0 - a0) / np.where(np.abs(den) > 1e-300, den, 1.0), 0.0)
        return np.clip(s, 0.0, 1.0)

    top = lambda s: np.maximum(ta0 + (ta1 - ta0) * s, tb0 + (tb1 - tb0) * s)
    bot = lambda s: np.minimum(ba0 + (ba1 - ba0) * s, bb0 + (bb1 - bb0) * s)
    pts = np.sort(np.stack([np.zeros_like(ta0), cross(ta0, ta1, tb0, tb1), cross(ba0, ba1, bb0, bb1),
                            np.ones_like(ta0)]), axis=0)
    cuts = [pts[0]]
    for m in range(3):
        a, b = pts[m], pts[m + 1]
        ha, hb = bot(a) - top(a), bot(b) - top(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((ha * hb < 0), a + (b - a) * ha / np.where(ha != hb, ha - hb, 1.0), a)
        cuts += [r, b]
    cuts = np.sort(np.stack(cuts), axis=0)
    area = np.zeros_like(ta0)
    ms = np.zeros_like(ta0)
    mz = np.zeros_like(ta0)
    for m in range(cuts.shape[0] - 1):
        a, b = cuts[m], cuts[m + 1]
        w = b - a
        sm = 0.5 * (a + b)
        vals = []
        for s in (a, sm, b):
            t, bo = top(s), bot(s)
            h = np.maximum(bo - t, 0.0)
            vals.append((h, s * h, 0.5 * (t + bo) * h))
        pos = (bot(sm) - top(sm)) > 0
        for n, acc in enumerate((area, ms, mz)):
            acc += np.where(pos, (vals[0][n] + 4 * vals[1][n] + vals[2][n]) * w / 6.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_c = np.where(area > 0, ms / np.where(area > 0, area, 1.0), 0.5)
        z_c = np.where(area > 0, mz / np.where(area > 0, area, 1.0), 0.5 * (ta0 + ba0))
    return area, s_c, z_c


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
        try:
            self._check_matching()
            self.matching = True
        except NotImplementedError:
            self.matching = False

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
        """Полупроводимость ячеек через грань face; k = (kx, ky, kz), берётся компонента по направлению грани."""
        fc, area = self.faces[face]
        return _half(k["ijk".index(face[0])], area, fc - self.center)

    def connections(self, kx, ky, kz, active=None, pinch=None):
        """
        Все связи TPFA сетки, включая разломы (смещённые по разлому слои) и выклинивания:
        возвращает dict(c1, c2, T, axis, nnc) — плоские индексы [i,j,k] (k быстрее всех), T = 1/(1/T1+1/T2).
        axis: 0/1/2 — направление (i/j/k); nnc — связь не с прямым соседом (i+1,j,k)/(i,j+1,k)/(i,j,k+1).
        pinch: None или dict(threshold, gap: bool, max_gap) — связи по k через неактивные ячейки колонки.
        """
        nx, ny, nz = self.dims
        act = self.actnum if active is None else active
        k = (kx, ky, kz)
        flat = lambda i, j, kk: (i * ny + j) * nz + kk
        out = {"c1": [], "c2": [], "T": [], "axis": [], "nnc": []}

        def add(a_idx, b_idx, t, axis, nnc):
            keep = t > 0
            out["c1"].append(a_idx[keep]); out["c2"].append(b_idx[keep]); out["T"].append(t[keep])
            out["axis"].append(np.full(int(keep.sum()), axis, dtype=np.int8))
            out["nnc"].append(np.full(int(keep.sum()), nnc, dtype=bool))

        c = self.corners
        for axis in (0, 1):
            if self.dims[axis] < 2:
                continue
            # углы граней на общих столбах: (верх s0, верх s1, низ s0, низ s1) у ячейки A (+грань) и B (−грань)
            ia = (1, 3, 5, 7) if axis == 0 else (2, 3, 6, 7)
            ib = (0, 2, 4, 6) if axis == 0 else (0, 1, 4, 5)
            sl_a = (slice(0, nx - 1), slice(None)) if axis == 0 else (slice(None), slice(0, ny - 1))
            sl_b = (slice(1, nx), slice(None)) if axis == 0 else (slice(None), slice(1, ny))
            ca, cb = c[sl_a], c[sl_b]
            ii, jj = np.meshgrid(np.arange(ca.shape[0]), np.arange(ca.shape[1]), indexing="ij")
            if axis == 0:
                ib_i, ib_j = ii + 1, jj
            else:
                ib_i, ib_j = ii, jj + 1
            for dk in range(-(nz - 1), nz):
                ka = np.arange(max(0, -dk), min(nz, nz - dk))
                if ka.size == 0:
                    continue
                kb = ka + dk
                A, B = ca[:, :, ka], cb[:, :, kb]
                za = [A[..., m, 2] for m in ia]
                zb = [B[..., m, 2] for m in ib]
                area, _, _ = _overlap(*za, *zb)
                full = 0.5 * ((za[2] - za[0]) + (za[3] - za[1]))
                ok = (area > 1e-12 * np.maximum(full, 1e-300)) & (full > 0)
                ai = ii[:, :, None] + 0 * ka
                aj = jj[:, :, None] + 0 * ka
                bi = ib_i[:, :, None] + 0 * ka
                bj = ib_j[:, :, None] + 0 * ka
                ok &= act[ai, aj, ka] & act[bi, bj, kb]
                if not ok.any():
                    continue
                # как Eclipse NEWTRAN: d — до центра полной грани ячейки, площадь — доля перекрытия от её грани
                ca_c = self.center[sl_a][:, :, ka]
                cb_c = self.center[sl_b][:, :, kb]
                fa_c, fa_a = (v[sl_a][:, :, ka] for v in self.faces["i+" if axis == 0 else "j+"])
                fb_c, fb_a = (v[sl_b][:, :, kb] for v in self.faces["i-" if axis == 0 else "j-"])
                full_b = 0.5 * ((zb[2] - zb[0]) + (zb[3] - zb[1]))
                t1 = _half(k[axis][sl_a][:, :, ka], fa_a * (area / np.maximum(full, 1e-300))[..., None], fa_c - ca_c)
                t2 = _half(k[axis][sl_b][:, :, kb], fb_a * (area / np.maximum(full_b, 1e-300))[..., None], fb_c - cb_c)
                t = np.where(ok & (t1 > 0) & (t2 > 0), 1.0 / (1.0 / np.maximum(t1, 1e-300) + 1.0 / np.maximum(t2, 1e-300)), 0.0)
                KA = np.broadcast_to(ka, t.shape)
                add(flat(ai, aj, KA).ravel(), flat(bi, bj, KA + dk).ravel(), t.ravel(), axis, dk != 0)

        # по k: следующая активная ячейка колонки (через выклинившиеся — по правилам PINCH)
        th = self.dz
        t_dn = _half(k[2], self.faces["k+"][1], self.faces["k+"][0] - self.center)
        t_up = _half(k[2], self.faces["k-"][1], self.faces["k-"][0] - self.center)
        cur = np.full((nx, ny), -1)
        gap = np.zeros((nx, ny))
        blocked = np.zeros((nx, ny), dtype=bool)
        A_i, A_j, A_k, B_k = [], [], [], []
        I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        for kk in range(nz):
            a = act[:, :, kk]
            link = a & (cur >= 0)
            if pinch is None:
                link &= (cur == kk - 1)
            else:
                skipped = kk - 1 - cur
                link &= (skipped == 0) | (~blocked & (gap <= pinch.get("max_gap", 1e20)))
            A_i.append(I[link]); A_j.append(J[link]); A_k.append(cur[link]); B_k.append(np.full(int(link.sum()), kk))
            gap = np.where(a, 0.0, gap + th[:, :, kk])
            if pinch is not None:
                thin = th[:, :, kk] < pinch["threshold"]
                # без GAP связь возможна только через ячейки тоньше порога
                blocked = np.where(a, False, blocked | (~thin & (not pinch.get("gap", True))))
            cur = np.where(a, kk, cur)
        if A_i:
            ai, aj, akk, bkk = (np.concatenate(v) for v in (A_i, A_j, A_k, B_k))
            t1 = t_dn[ai, aj, akk]
            t2 = t_up[ai, aj, bkk]
            t = np.where((t1 > 0) & (t2 > 0), 1.0 / (1.0 / np.maximum(t1, 1e-300) + 1.0 / np.maximum(t2, 1e-300)), 0.0)
            add(flat(ai, aj, akk), flat(ai, aj, bkk), t, 2, False)
        res = {key: np.concatenate(v) if v else np.zeros(0) for key, v in out.items()}
        res["c1"] = res["c1"].astype(np.int64)
        res["c2"] = res["c2"].astype(np.int64)
        return res

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
