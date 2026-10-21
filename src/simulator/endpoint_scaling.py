"""
Масштабирование концевых точек ОФП и Pc (ENDSCALE, Eclipse): ячеечные SWL, SWCR, SWU, SOWCR, SGL, SGCR,
SGU, SOGCR, KRW, KRO, KRG, PCW, PCG поверх таблиц SWOF/SGOF региона SATNUM.

Горизонтальное масштабирование — кусочно-линейное отображение насыщенности ячейки в насыщенность таблицы:

    krw(Sw):   двухточечное (SWCR, SWU);         трёхточечное (SCALECRS YES): (SWCR, 1−SOWCR, SWU)
    krow(Sw):  (SWL, 1−SOWCR);                   трёхточечное: (SWL, SWCR, 1−SOWCR)
    krg(Sg):   (SGCR, SGU);                      трёхточечное: (SGCR, 1−SOGCR−SWL, SGU)
    krog(Sg):  (SGL, 1−SWL−SOGCR);               трёхточечное: (SGL, SGCR, 1−SWL−SOGCR)
    Pcow(Sw):  (SWL, SWU);  Pcog(Sg): (SGL, SGU)

Кривые нефти отображаются прямо по Sw и Sg (эквивалентно отображению So), поэтому при совпадающих
концевых точках отображение — точное тождество (без ошибок округления в узлах таблиц).
Вертикальное: krw·KRW/krw_t(SWU_t), krg·KRG/krg_t(SGU_t), kro·KRO/kro_t(SWL_t), Pc·PCW/max Pc_t (PCG).
Трёхфазная kro — та же модель (по умолчанию Eclipse или Stone II), что и без масштабирования.
Гистерезис (hysteresis.py) подключается через self.hyst: подменяет krow и krg кривыми сканирования.
"""
import numpy as np
import torch

from .blackoil_props import interp1

KEYS = ("SWL", "SWCR", "SWU", "SOWCR", "SGL", "SGCR", "SGU", "SOGCR", "KRW", "KRO", "KRG", "PCW", "PCG")


def table_endpoints(swof, sgof):
    """Концевые точки таблиц SWOF/SGOF (как их определяет Eclipse)."""
    ep = {}
    if swof is not None:
        sw, krw, kro, pc = (list(map(float, swof[k])) for k in ("sw", "krw", "kro", "pcow"))
        ep["SWL"], ep["SWU"] = sw[0], sw[-1]
        ep["SWCR"] = max(s for s, k in zip(sw, krw) if k <= 0.0) if any(k <= 0 for k in krw) else sw[0]
        zero_o = [s for s, k in zip(sw, kro) if k <= 0.0]
        ep["SOWCR"] = 1.0 - min(zero_o) if zero_o else 1.0 - sw[-1]
        ep["KRW"], ep["KRO"] = krw[-1], kro[0]
        ep["PCW"] = max(abs(v) for v in pc)
    else:
        ep.update(SWL=0.0, SWU=1.0, SWCR=0.0, SOWCR=0.0, KRW=0.0, KRO=1.0, PCW=0.0)
    if sgof is not None:
        sg, krg, kro, pc = (list(map(float, sgof[k])) for k in ("sg", "krg", "kro", "pcog"))
        ep["SGL"], ep["SGU"] = sg[0], sg[-1]
        ep["SGCR"] = max(s for s, k in zip(sg, krg) if k <= 0.0) if any(k <= 0 for k in krg) else sg[0]
        zero_o = [s for s, k in zip(sg, kro) if k <= 0.0]
        ep["SOGCR"] = (1.0 - ep["SWL"] - min(zero_o)) if zero_o else 0.0
        ep["KRG"] = krg[-1]
        ep["PCG"] = max(abs(v) for v in pc)
    else:
        ep.update(SGL=0.0, SGU=1.0 - ep["SWL"], SGCR=0.0, SOGCR=0.0, KRG=0.0, PCG=0.0)
    return ep


def piecewise_map(s, src, dst):
    """
    Кусочно-линейное s ↦ насыщенность таблицы: узлы src (тензоры ячеек, по возрастанию) → dst (числа);
    вне диапазона — концы. При src == dst — точное тождество (без ошибок округления в узлах таблиц).
    """
    out = None
    for a, b, ya, yb in zip(src[:-1], src[1:], dst[:-1], dst[1:]):
        a = torch.as_tensor(a, dtype=s.dtype, device=s.device).expand_as(s)
        b = torch.as_tensor(b, dtype=s.dtype, device=s.device).expand_as(s)
        wide = (b - a) > 1e-12
        same = (a == ya) & (b == yb)
        slope = torch.where(wide & ~same, (yb - ya) / torch.where(wide, b - a, torch.ones_like(b)), torch.ones_like(b))
        sc = torch.minimum(torch.where(s >= a, s, a), b)
        seg = sc + (ya - a) + (sc - a) * (slope - 1.0)
        out = seg if out is None else torch.where(s >= a, seg, out)
    return out


class ScaledRelPerm:
    """
    ОФП и Pc с масштабированием концевых точек. tables — [(swof, sgof)] по регионам (тензорные словари),
    reg — номер таблицы ячейки (0-based), cell — {ключ: тензор [n_cells]} (уже заполнен значениями таблиц
    региона), tab_ep — концевые точки таблиц по регионам.
    """

    def __init__(self, tables, reg, cell, tab_ep, three_point=False, kro_model="default"):
        self.tables, self.reg, self.cell, self.tab_ep = tables, reg.long(), cell, tab_ep
        self.three_point = three_point
        self.kro_model = kro_model
        self.swc = float(cell["SWL"].min())
        self.hyst = None
        self._t = tab_ep[0]

    # ------------------------------------------------------------ гистерезис
    def _hyst_krow(self, sw, kd):
        return kd if self.hyst is None else self.hyst.krow(sw, kd)

    def _hyst_krg(self, sg, kd):
        return kd if self.hyst is None else self.hyst.krg(sg, kd)

    def _select(self, fn):
        out = None
        for r in range(len(self.tables)):
            self._t = self.tab_ep[r]
            v = fn(r)
            out = v if out is None else torch.where(self.reg == r, v, out)
        return out

    def krow_curve(self, sw):
        """Двухфазная krow(Sw) системы нефть–вода без гистерезиса (масштабированная), все ячейки."""
        def f(r):
            tw = self.tables[r][0]
            return interp1(self._sw_o(sw), tw["sw"], tw["kro"]) * (self.cell["KRO"] / max(self._t["KRO"], 1e-30))
        return self._select(f)

    def krg_curve(self, sg):
        """Двухфазная krg(Sg) системы газ–нефть без гистерезиса (масштабированная), все ячейки."""
        def f(r):
            tg, t, c = self.tables[r][1], self._t, self.cell
            if tg is None:
                return 0.0 * sg
            return interp1(self._sg_kr(sg), tg["sg"], tg["krg"]) * (c["KRG"] / max(t["KRG"], 1e-30))
        return self._select(f)

    # ------------------------------------------------------------ отображения насыщенностей (регион self._t)
    def _sw_kr(self, sw):
        t, c = self._t, self.cell
        if self.three_point:
            return piecewise_map(sw, [c["SWCR"], 1.0 - c["SOWCR"], c["SWU"]], [t["SWCR"], 1.0 - t["SOWCR"], t["SWU"]])
        return piecewise_map(sw, [c["SWCR"], c["SWU"]], [t["SWCR"], t["SWU"]])

    def _sw_o(self, sw):
        """Sw таблицы для krow: узлы (SWL, [SWCR], 1−SOWCR) — эквивалент отображения So по (SOWCR, [1−SWCR], 1−SWL)."""
        t, c = self._t, self.cell
        if self.three_point:
            return piecewise_map(sw, [c["SWL"], c["SWCR"], 1.0 - c["SOWCR"]],
                                 [t["SWL"], t["SWCR"], 1.0 - t["SOWCR"]])
        return piecewise_map(sw, [c["SWL"], 1.0 - c["SOWCR"]], [t["SWL"], 1.0 - t["SOWCR"]])

    def _sg_kr(self, sg):
        t, c = self._t, self.cell
        if self.three_point:
            return piecewise_map(sg, [c["SGCR"], 1.0 - c["SOGCR"] - c["SWL"], c["SGU"]],
                                 [t["SGCR"], 1.0 - t["SOGCR"] - t["SWL"], t["SGU"]])
        return piecewise_map(sg, [c["SGCR"], c["SGU"]], [t["SGCR"], t["SGU"]])

    def _sg_o(self, sg):
        """Sg таблицы для krog (So = 1 − SWL − Sg): узлы (SGL, [SGCR], 1−SWL−SOGCR)."""
        t, c = self._t, self.cell
        if self.three_point:
            return piecewise_map(sg, [c["SGL"], c["SGCR"], 1.0 - c["SWL"] - c["SOGCR"]],
                                 [t["SGL"], t["SGCR"], 1.0 - t["SWL"] - t["SOGCR"]])
        return piecewise_map(sg, [c["SGL"], 1.0 - c["SWL"] - c["SOGCR"]], [t["SGL"], 1.0 - t["SWL"] - t["SOGCR"]])

    # ------------------------------------------------------------ ОФП и Pc
    def _region(self, r, sw, sg):
        tw, tg = self.tables[r]
        t, c = self.tab_ep[r], self.cell
        self._t = t
        swl = c["SWL"]
        krw = interp1(self._sw_kr(sw), tw["sw"], tw["krw"]) * (c["KRW"] / max(t["KRW"], 1e-30))
        kr_o_scale = c["KRO"] / max(t["KRO"], 1e-30)
        krow = self._hyst_krow(sw, interp1(self._sw_o(sw), tw["sw"], tw["kro"]) * kr_o_scale)
        sw_pc = piecewise_map(sw, [swl, c["SWU"]], [t["SWL"], t["SWU"]])
        pcow = interp1(sw_pc, tw["sw"], tw["pcow"]) * (c["PCW"] / t["PCW"] if t["PCW"] > 0 else 0.0)
        if tg is not None:
            krg = self._hyst_krg(sg, interp1(self._sg_kr(sg), tg["sg"], tg["krg"]) * (c["KRG"] / max(t["KRG"], 1e-30)))
            sg_pc = piecewise_map(sg, [c["SGL"], c["SGU"]], [t["SGL"], t["SGU"]])
            pcog = interp1(sg_pc, tg["sg"], tg["pcog"]) * (c["PCG"] / t["PCG"] if t["PCG"] > 0 else 0.0)
            krog = interp1(self._sg_o(sg), tg["sg"], tg["kro"]) * kr_o_scale
        else:
            krg, pcog = 0.0 * sg, 0.0 * sg
            krog = krow * 0.0 + c["KRO"]
        if self.kro_model == "default":
            sw_e = torch.where(sw >= swl, sw, swl)
            den = sg + sw_e - swl
            small = den < 1e-12
            # krow при той же So (Sw' = Sw + Sg), krog при той же So (Sg' = Sg + Sw − SWL)
            krow_o, krog_o = self._two_at(r, sw_e + sg, sg + sw_e - swl)
            kro = torch.where(small, krow_o, (sg * krog_o + (sw_e - swl) * krow_o) / torch.where(small, 1.0, den))
        else:
            cw = c["KRO"]
            kro = cw * ((krow / cw + krw) * (krog / cw + krg) - krw - krg)
        return krw, kro.clamp_min(0.0), krg, pcow, pcog

    def _two_at(self, r, sw, sg):
        """krow(Sw) и krog(Sg) с масштабированием (для модели kro по умолчанию)."""
        tw, tg = self.tables[r]
        c = self.cell
        kr_o_scale = c["KRO"] / max(self._t["KRO"], 1e-30)
        krow = self._hyst_krow(sw, interp1(self._sw_o(sw), tw["sw"], tw["kro"]) * kr_o_scale)
        if tg is None:
            return krow, krow * 0.0 + c["KRO"]
        return krow, interp1(self._sg_o(sg), tg["sg"], tg["kro"]) * kr_o_scale

    def _all(self, sw, sg):
        outs = [self._region(r, sw, sg) for r in range(len(self.tables))]
        res = list(outs[0])
        for r in range(1, len(outs)):
            m = self.reg == r
            res = [torch.where(m, o, b) for o, b in zip(outs[r], res)]
        return res

    def __call__(self, sw, sg):
        krw, kro, krg, _, _ = self._all(sw, sg)
        return krw, kro, krg

    def pc(self, sw, sg):
        _, _, _, pcow, pcog = self._all(sw, sg)
        return pcow, pcog

    # ------------------------------------------------------------ для равновесия (numpy, по ячейкам)
    def _pc_parts(self, idx, phase, pc_scale):
        lo_k, hi_k, val_k, col, sk, pk = (("SWL", "SWU", "PCW", 0, "sw", "pcow") if phase == "w"
                                          else ("SGL", "SGU", "PCG", 1, "sg", "pcog"))
        idx = np.arange(self.reg.numel())[idx]
        reg = self.reg.cpu().numpy()[idx]
        lo = self.cell[lo_k].cpu().numpy()[idx]
        hi = self.cell[hi_k].cpu().numpy()[idx]
        val = self.cell[val_k].cpu().numpy()[idx] if pc_scale is None else np.asarray(pc_scale, dtype=float)
        for r in np.unique(reg):
            k = np.flatnonzero(reg == r)
            tab = self.tables[r][col]
            if tab is None:
                continue
            t = self.tab_ep[r]
            st, pt = tab[sk].cpu().numpy(), tab[pk].cpu().numpy()
            b = (hi[k] - lo[k]) / max(t[hi_k] - t[lo_k], 1e-12)
            c = val[k] / t[val_k] if t[val_k] > 0 else np.zeros(k.size)
            yield k, st, pt, lo[k], t[lo_k], b, c

    def pc_value(self, idx, phase, s, pc_scale=None):
        """Pc (масштабированный) в ячейках idx при насыщенности s (Sw для 'w', Sg для 'g')."""
        out = np.zeros(len(np.arange(self.reg.numel())[idx]))
        for k, st, pt, lo, tl, b, c in self._pc_parts(idx, phase, pc_scale):
            s_t = tl + (np.asarray(s)[k] - lo) / np.where(b > 0, b, 1.0)
            out[k] = np.interp(s_t, st, pt) * c
        return out

    def invert_pc(self, idx, phase, y, pc_scale=None):
        """Насыщенность, при которой Pc ячейки равен y (Pcow убывает по Sw, Pcog растёт по Sg); NaN — Pc ≡ 0."""
        out = np.full(len(np.arange(self.reg.numel())[idx]), np.nan)
        for k, st, pt, lo, tl, b, c in self._pc_parts(idx, phase, pc_scale):
            if np.ptp(pt) <= 0:
                continue
            yt = np.asarray(y)[k] / np.where(c > 0, c, np.inf)
            s_t = np.interp(yt, pt[::-1], st[::-1]) if phase == "w" else np.interp(yt, pt, st)
            out[k] = np.where(c > 0, lo + (s_t - tl) * b, np.nan)
        return out

    def endpoint(self, key, idx):
        return self.cell[key].cpu().numpy()[idx]
