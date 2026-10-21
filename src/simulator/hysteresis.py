"""
Гистерезис ОФП (SATOPTS HYSTER, EHYSTR модель 0 / 1, только KR) — модель Карлсона для несмачивающих фаз.

Несмачивающие фазы: нефть в системе нефть–вода (krow), газ в системе газ–нефть (krg).
Кривые дренирования — таблицы SATNUM с концевыми точками SWL…SOGCR; пропитки — таблицы IMBNUM с ISWL…ISOGCR.

Точка разворота — исторический экстремум насыщенности несмачивающей фазы в ячейке (min Sw для нефти,
max Sg для газа), обновляется после каждого принятого шага (commit). Пока насыщенность несмачивающей фазы
не меньше исторического максимума — кривая дренирования; иначе — кривая пропитки, сдвинутая по
насыщенности так, чтобы пройти через точку разворота (Карлсон):

    krn(S) = krn_imb(S + ΔS),   krn_imb(S_rev + ΔS) = krn_drain(S_rev).

Кривая непрерывна в точке разворота; ΔS в пределах шага заморожен, поэтому якобиан Ньютона точный.
Смачивающие фазы (krw, krog): модель 0 — кривая дренирования, модель 1 — кривая пропитки.
"""
import torch


def _bisect(fn, target, lo, hi, increasing, iters=60):
    """Решение fn(s) = target на [lo, hi] (по ячейкам) для монотонной fn."""
    lo, hi = lo.clone(), hi.clone()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = fn(mid) > target
        go_left = above if increasing else ~above
        hi = torch.where(go_left, mid, hi)
        lo = torch.where(go_left, lo, mid)
    return 0.5 * (lo + hi)


class CarlsonHysteresis:
    def __init__(self, drain, imb, model=0):
        if model not in (0, 1):
            raise NotImplementedError(f"EHYSTR: модель {model} не поддерживается (только 0 и 1 — Карлсон)")
        self.drain, self.imb, self.model = drain, imb, model
        n = drain.reg.numel()
        dev = drain.reg.device
        self.sw_min = torch.ones(n, dtype=torch.float64, device=dev)
        self.sg_max = torch.zeros(n, dtype=torch.float64, device=dev)
        self.shift_w = torch.zeros(n, dtype=torch.float64, device=dev)
        self.shift_g = torch.zeros(n, dtype=torch.float64, device=dev)
        self.on_w = torch.zeros(n, dtype=torch.bool, device=dev)
        self.on_g = torch.zeros(n, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------ кривые со сдвигом
    def krow(self, sw, kd):
        if not bool(self.on_w.any()):
            return kd
        scan = self.imb.krow_curve(sw + self.shift_w)
        return torch.where(self.on_w & (sw > self.sw_min), torch.minimum(scan, kd), kd)

    def krg(self, sg, kd):
        if not bool(self.on_g.any()):
            return kd
        scan = self.imb.krg_curve(sg + self.shift_g)
        return torch.where(self.on_g & (sg < self.sg_max), torch.minimum(scan, kd), kd)

    # ------------------------------------------------------------ обновление точек разворота
    @torch.no_grad()
    def commit(self, sw, sg):
        sw = sw.reshape(-1).to(torch.float64)
        sg = sg.reshape(-1).to(torch.float64)
        self.sw_min = torch.minimum(self.sw_min, sw)
        self.sg_max = torch.maximum(self.sg_max, sg)
        d, i, c = self.drain, self.imb, self.imb.cell
        # нефть: krow убывает по Sw
        h = self.drain.hyst
        d.hyst = None
        try:
            k_rev = d.krow_curve(self.sw_min)
            self.on_w = k_rev > 0
            lo, hi = c["SWL"], 1.0 - c["SOWCR"]
            s_imb = _bisect(i.krow_curve, k_rev, lo.clone(), hi.clone(), increasing=False)
            self.shift_w = torch.where(self.on_w, s_imb - self.sw_min, 0.0 * s_imb)
            # газ: krg растёт по Sg
            k_rev = d.krg_curve(self.sg_max)
            self.on_g = k_rev > 0
            lo, hi = c["SGCR"], c["SGU"]
            s_imb = _bisect(i.krg_curve, k_rev, lo.clone(), hi.clone(), increasing=True)
            self.shift_g = torch.where(self.on_g, s_imb - self.sg_max, 0.0 * s_imb)
        finally:
            d.hyst = h

    def trapped_gas(self):
        """Остаточная (защемлённая) газонасыщенность на текущей кривой сканирования: krg = 0 при Sg ≤ Sgt."""
        return torch.where(self.on_g, self.imb.cell["SGCR"] - self.shift_g, self.drain.cell["SGCR"])
