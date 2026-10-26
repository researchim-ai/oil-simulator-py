"""
Таблицы лифта VFPPROD (Eclipse): забойное давление на опорной глубине таблицы как функция
дебита (FLO), устьевого давления (THP), обводнённости (WFR), газового фактора (GFR) и газлифта (ALQ).

Интерполяция многолинейная по пяти осям, за пределами осей — линейная экстраполяция крайним
интервалом (как в OPM, VFPHelpers). Всё в СИ: дебиты — ст. м³/с, давления — Па, отношения — м³/м³.
"""
import itertools

import torch

DAY = 86400.0


def _axis_weights(axis, x):
    """Индекс левого узла и доля внутри интервала (может выходить за [0, 1] — экстраполяция)."""
    n = axis.numel()
    if n == 1:
        return torch.zeros_like(x, dtype=torch.long), torch.zeros_like(x)
    i = (torch.searchsorted(axis, x.detach().contiguous(), right=True) - 1).clamp(0, n - 2)
    x0, x1 = axis[i], axis[i + 1]
    return i, (x - x0) / (x1 - x0)


class VFPProd:
    FLO = ("OIL", "LIQ", "GAS")
    WFR = ("WOR", "WCT", "WGR")
    GFR = ("GOR", "GLR", "OGR")

    def __init__(self, records, u, depth_shift=0.0):
        """records — записи ключевого слова VFPPROD; u — переводные множители deck → СИ (EclipseCase.u)."""
        head = records[0]
        g = lambda k, d=None: head[k] if len(head) > k and head[k] is not None else d
        self.number = int(head[0])
        self.datum = float(head[1]) * u["L"] - depth_shift
        self.flo_type = str(g(2, "OIL")).upper()
        self.wfr_type = str(g(3, "WCT")).upper()
        self.gfr_type = str(g(4, "GOR")).upper()
        thp_type = str(g(5, "THP")).upper()
        alq_type = str(g(6, "")).upper()
        tab_type = str(g(8, "BHP")).upper()
        if self.flo_type not in self.FLO or self.wfr_type not in self.WFR or self.gfr_type not in self.GFR:
            raise NotImplementedError(f"VFPPROD {self.number}: оси {self.flo_type}/{self.wfr_type}/{self.gfr_type}")
        if thp_type != "THP" or tab_type != "BHP":
            raise NotImplementedError(f"VFPPROD {self.number}: {thp_type}/{tab_type} (поддерживаются THP и BHP)")
        if alq_type not in ("", "GRAT", "IGLR", "TGLR", "PUMP", "COMP", "BEAN", "ALQ"):
            raise NotImplementedError(f"VFPPROD {self.number}: ALQ {alq_type}")
        p, liq, gas = u["p"], u["liq"] / DAY, u["gas"] / DAY
        flo_u = gas if self.flo_type == "GAS" else liq
        wfr_u = {"WOR": 1.0, "WCT": 1.0, "WGR": u["liq"] / u["gas"]}[self.wfr_type]
        gfr_u = {"GOR": u["gas"] / u["liq"], "GLR": u["gas"] / u["liq"], "OGR": u["liq"] / u["gas"]}[self.gfr_type]
        T = lambda rec, s: torch.tensor([float(v) * s for v in rec], dtype=torch.float64)
        self.flo, self.thp = T(records[1], flo_u), T(records[2], p)
        self.wfr, self.gfr, self.alq = T(records[3], wfr_u), T(records[4], gfr_u), T(records[5], 1.0)
        shape = (self.thp.numel(), self.wfr.numel(), self.gfr.numel(), self.alq.numel(), self.flo.numel())
        tab = torch.full(shape, float("nan"), dtype=torch.float64)
        for rec in records[6:]:
            if not rec:
                continue
            it, iw, ig, ia = (int(v) - 1 for v in rec[:4])
            tab[it, iw, ig, ia] = torch.tensor([float(v) * p for v in rec[4:4 + shape[4]]], dtype=torch.float64)
        if torch.isnan(tab).any():
            raise ValueError(f"VFPPROD {self.number}: таблица заполнена не полностью")
        self.table = tab

    def _ratios(self, qo, qw, qg):
        div = lambda a, b: torch.where(b.abs() > 1e-30, a / torch.where(b.abs() > 1e-30, b, torch.ones_like(b)),
                                       torch.zeros_like(a))
        flo = {"OIL": qo, "LIQ": qo + qw, "GAS": qg}[self.flo_type]
        wfr = {"WOR": div(qw, qo), "WCT": div(qw, qo + qw), "WGR": div(qw, qg)}[self.wfr_type]
        gfr = {"GOR": div(qg, qo), "GLR": div(qg, qo + qw), "OGR": div(qo, qg)}[self.gfr_type]
        return flo, wfr, gfr

    def bhp(self, qo, qw, qg, thp, alq=0.0):
        """BHP на опорной глубине таблицы, Па; дебиты добычи (> 0), ст. м³/с; тензоры одной формы."""
        dev = qo.device
        flo, wfr, gfr = self._ratios(qo, qw, qg)
        thp = torch.as_tensor(thp, dtype=torch.float64, device=dev).expand_as(flo)
        alq = torch.as_tensor(alq, dtype=torch.float64, device=dev).expand_as(flo)
        axes = [a.to(dev) for a in (self.thp, self.wfr, self.gfr, self.alq, self.flo)]
        idx, frac = zip(*(_axis_weights(a, x) for a, x in zip(axes, (thp, wfr, gfr, alq, flo))))
        tab = self.table.to(dev)
        out = torch.zeros_like(flo)
        for corner in itertools.product((0, 1), repeat=5):
            w = torch.ones_like(flo)
            ii = []
            for k, c in enumerate(corner):
                n = axes[k].numel()
                if c and n == 1:
                    w = None
                    break
                w = w * (frac[k] if c else 1.0 - frac[k])
                ii.append(idx[k] + c)
            if w is not None:
                out = out + w * tab[ii[0], ii[1], ii[2], ii[3], ii[4]]
        return out

    def thp_from_bhp(self, bhp, qo, qw, qg, alq=0.0):
        """Обратная задача: THP, при котором таблица даёт bhp (BHP линейна по THP на интервале оси)."""
        thp = self.thp.to(bhp.device)
        vals = torch.stack([self.bhp(qo, qw, qg, t, alq) for t in thp])          # [n_thp, ...]
        n = thp.numel()
        if n == 1:
            return thp[0].expand_as(bhp)
        k = ((vals <= bhp.unsqueeze(0)).sum(0) - 1).clamp(0, n - 2)
        b0 = vals.gather(0, k.unsqueeze(0)).squeeze(0)
        b1 = vals.gather(0, (k + 1).unsqueeze(0)).squeeze(0)
        t0, t1 = thp[k], thp[k + 1]
        return t0 + (bhp - b0) * (t1 - t0) / torch.where((b1 - b0).abs() > 0, b1 - b0, torch.ones_like(b0))
