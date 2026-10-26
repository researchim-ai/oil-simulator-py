"""
Свойства флюидов и породы для black-oil FIM на torch (дифференцируемы, для torch.func.jvp).
"""
import math

import torch

_BOUNDS = {}


def _bounds(xp):
    """Границы таблицы числами Python: без синхронизации с GPU на каждом вызове."""
    hit = _BOUNDS.get(id(xp))
    if hit is None or hit[0] is not xp:
        hit = _BOUNDS[id(xp)] = (xp, float(xp[0]), float(xp[-1]))
    return hit[1], hit[2]


def interp1(x, xp, fp):
    """Кусочно-линейная интерполяция с постоянной экстраполяцией; дифференцируема по x."""
    n = xp.numel()
    if n == 1:
        return fp[0] + 0.0 * x
    lo, hi = _bounds(xp)
    # where вместо clamp: у clamp под vmap(jvp) медленное разложение
    xc = torch.where(x < lo, lo, torch.where(x > hi, hi, x))
    i = torch.searchsorted(xp, xc.detach().contiguous(), right=True).clamp(1, n - 1)
    x0, x1, y0, y1 = xp.index_select(0, i - 1), xp.index_select(0, i), fp.index_select(0, i - 1), fp.index_select(0, i)
    return y0 + (y1 - y0) * (xc - x0) / (x1 - x0)


def interp1_lin(x, xp, fp):
    """Кусочно-линейная интерполяция с линейной экстраполяцией за краями таблицы."""
    n = xp.numel()
    if n == 1:
        return fp[0] + 0.0 * x
    i = torch.searchsorted(xp, x.detach().contiguous(), right=True).clamp(1, n - 1)
    x0, x1, y0, y1 = xp.index_select(0, i - 1), xp.index_select(0, i), fp.index_select(0, i - 1), fp.index_select(0, i)
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def interp_rows(x, XP, FP):
    """Построчная интерполяция с линейной экстраполяцией: x [N], XP/FP [N, M] (строки возрастают)."""
    m = XP.shape[1]
    i = torch.searchsorted(XP, x.detach().contiguous().unsqueeze(1), right=True).clamp(1, m - 1)
    x0, x1 = XP.gather(1, i - 1).squeeze(1), XP.gather(1, i).squeeze(1)
    y0, y1 = FP.gather(1, i - 1).squeeze(1), FP.gather(1, i).squeeze(1)
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class DeckPVT:
    """
    PVT из deck Eclipse (в СИ), интерполяция как в OPM Flow:

    * PVTO — живая нефть. Насыщенная линия: Rs_sat(P), 1/Bo и 1/(Bo·μo) по P (линейная экстраполяция).
      Недонасыщенная нефть: для каждой записи Rs_i — ветвь по P; 1/Bo и 1/(Bo·μo) интерполируются
      по P внутри ветвей и линейно по Rs между соседними записями. Запись без ветви достраивается
      по ближайшей записи с ветвью в относительных величинах (как extendPvtoTable в OPM).
    * PVDO — мёртвая нефть; PVDG — сухой газ (интерполяция 1/Bg и 1/(Bg·μg));
    * PVTW — вода: Bw = Bw_ref / (1 + X + X²/2), X = c_w (P - P_ref);
      Bw·μw = Bw_ref·μw_ref / (1 + Y + Y²/2), Y = (c_w - c_v)(P - P_ref).
    """

    def __init__(self, tables, rho_sc, device):
        dt = torch.float64
        T = lambda a: torch.as_tensor(a, dtype=dt, device=device)
        self.device = device
        self.rho_sc = tuple(float(r) for r in rho_sc)
        self.has_pvt = True

        pvto = tables.get("pvto")
        if pvto:
            recs = [(float(rs), [tuple(map(float, row)) for row in rows]) for rs, rows in pvto]
            recs.sort(key=lambda r: r[0])
            recs = self._extend_pvto(recs)
            self.rs_tab = T([r[0] for r in recs])
            self.sat_p = T([r[1][0][0] for r in recs])
            sat_B = T([r[1][0][1] for r in recs])
            sat_mu = T([r[1][0][2] for r in recs])
            self.sat_b, self.sat_bm = 1.0 / sat_B, 1.0 / (sat_B * sat_mu)
            m = max(len(r[1]) for r in recs)
            P = torch.zeros((len(recs), m), dtype=dt)
            b = torch.zeros_like(P)
            bm = torch.zeros_like(P)
            for t, (_, rows) in enumerate(recs):
                p_ = [r[0] for r in rows]
                b_ = [1.0 / r[1] for r in rows]
                bm_ = [1.0 / (r[1] * r[2]) for r in rows]
                while len(p_) < m:
                    k = len(p_)
                    s_b = (b_[-1] - b_[-2]) / (p_[-1] - p_[-2])
                    s_m = (bm_[-1] - bm_[-2]) / (p_[-1] - p_[-2])
                    dp = 1e9 * k
                    p_.append(p_[-1] + dp)
                    b_.append(b_[-1] + s_b * dp)
                    bm_.append(bm_[-1] + s_m * dp)
                P[t], b[t], bm[t] = T(p_), T(b_), T(bm_)
            self.us_P, self.us_b, self.us_bm = P.to(device), b.to(device), bm.to(device)
            self.rs_max = float(self.rs_tab.max())
            self.live = True
            self.has_pvt = True
        else:
            pvdo = tables["pvdo"]
            self.live = False
            self.has_pvt = False
            self.rs_max = 0.0
            self.do_p = T([r[0] for r in pvdo])
            B = T([r[1] for r in pvdo])
            mu = T([r[2] for r in pvdo])
            self.do_b, self.do_bm = 1.0 / B, 1.0 / (B * mu)

        pvtg = tables.get("pvtg")
        self.wet = bool(pvtg) and bool(tables.get("vapoil", True))
        self.rv_max = 0.0
        if pvtg:
            self._init_pvtg(pvtg, T, dt)
        else:
            pvdg = tables["pvdg"]
            self.g_p = T([r[0] for r in pvdg])
            Bg = T([r[1] for r in pvdg])
            mug = T([r[2] for r in pvdg])
            self.g_b, self.g_bm = 1.0 / Bg, 1.0 / (Bg * mug)
        self.pvtw = tuple(float(v) for v in tables["pvtw"])

    def _init_pvtg(self, pvtg, T, dt):
        """
        PVTG: записи (P, [(Rv, Bg, μg), ...]), первая строка записи — насыщенный газ Rv_sat(P).
        Насыщенная ветвь — по P; недонасыщенная (Rv < Rv_sat): для каждой записи интерполяция 1/Bg и 1/(Bg·μg)
        по Rv (линейная экстраполяция), между записями — линейно по P (как WetGasPvt в OPM).
        Запись без недонасыщенной ветви достраивается по ближайшей записи с ветвью в относительных величинах.
        """
        recs = sorted(((float(p), [tuple(map(float, r)) for r in rows]) for p, rows in pvtg), key=lambda r: r[0])
        self.g_p = T([r[0] for r in recs])
        sat = [r[1][0] for r in recs]
        self.rv_tab = T([r[0] for r in sat])
        Bs = T([r[1] for r in sat])
        self.g_b, self.g_bm = 1.0 / Bs, 1.0 / (Bs * T([r[2] for r in sat]))
        self.rv_max = float(self.rv_tab.max())
        # недонасыщенные строки — только с Rv < Rv_sat записи (вырожденные дубли Rv_sat отбрасываются)
        recs = [(p, [rows[0]] + [r for r in rows[1:] if r[0] < rows[0][0] - 1e-12 * max(1.0, abs(rows[0][0]))])
                for p, rows in recs]
        branch = [len(r[1]) > 1 for r in recs]
        full = []
        for i, (p, rows) in enumerate(recs):
            if not branch[i]:
                order = list(range(i + 1, len(recs))) + list(range(i - 1, -1, -1))
                m = next((j for j in order if branch[j]), None)
                rv0, B0, mu0 = rows[0]
                if m is None:
                    rows = [rows[0], (rv0 - 1.0, B0, mu0)]
                else:
                    mr = recs[m][1]
                    mrv0, mB0, mmu0 = mr[0]
                    rows = [rows[0]] + [(rv0 + (r[0] - mrv0), B0 * r[1] / mB0, mu0 * r[2] / mmu0) for r in mr[1:]]
            rows = sorted(rows, key=lambda r: r[0])
            full.append(rows)
        m = max(len(r) for r in full)
        RV = torch.zeros((len(full), m), dtype=dt)
        b = torch.zeros_like(RV)
        bm = torch.zeros_like(RV)
        for t, rows in enumerate(full):
            rv_ = [r[0] for r in rows]
            b_ = [1.0 / r[1] for r in rows]
            bm_ = [1.0 / (r[1] * r[2]) for r in rows]
            while len(rv_) < m:
                s_b = (b_[-1] - b_[-2]) / (rv_[-1] - rv_[-2])
                s_m = (bm_[-1] - bm_[-2]) / (rv_[-1] - rv_[-2])
                d = 1.0 * len(rv_)
                rv_.append(rv_[-1] + d)
                b_.append(b_[-1] + s_b * d)
                bm_.append(bm_[-1] + s_m * d)
            RV[t], b[t], bm[t] = T(rv_), T(b_), T(bm_)
        dev = self.g_p.device
        self.ug_RV, self.ug_b, self.ug_bm = RV.to(dev), b.to(dev), bm.to(dev)
        self.pvtg = True

    @staticmethod
    def _extend_pvto(recs):
        out = []
        for i, (rs, rows) in enumerate(recs):
            if len(rows) > 1:
                out.append((rs, rows))
                continue
            order = list(range(i + 1, len(recs))) + list(range(i - 1, -1, -1))
            master = next((j for j in order if len(recs[j][1]) > 1), None)
            if master is None:
                raise ValueError("PVTO: ни одна запись не содержит недонасыщенной ветви")
            p0, B0, mu0 = rows[0][:3]
            mp0, mB0, mmu0 = recs[master][1][0][:3]
            ext = [rows[0][:3]] + [(p0 + r[0] - mp0, B0 * r[1] / mB0, mu0 * r[2] / mmu0)
                                   for r in recs[master][1][1:]]
            out.append((rs, ext))
        return out

    def at(self, idx):
        return self

    def region(self, r):
        return self

    def rs_sat(self, p):
        if not self.live:
            return 0.0 * p
        return interp1_lin(p, self.sat_p, self.rs_tab).clamp_min(0.0)

    def oil(self, p, rs, sat):
        if not self.live:
            b, bm = interp1_lin(p, self.do_p, self.do_b), interp1_lin(p, self.do_p, self.do_bm)
            return b, b / bm
        # насыщенная нефть — та же 2D-интерполяция в точке (Rs_sat(P), P), что и недонасыщенная (как LiveOilPvt
        # в OPM): ветви совпадают на линии насыщения, иначе переключения дают скачки массы и Ньютон зацикливается
        rs = torch.where(sat, self.rs_sat(p), rs)
        n_t = self.rs_tab.numel()
        i = torch.searchsorted(self.rs_tab, rs.detach().contiguous(), right=True).clamp(1, n_t - 1)
        lo, hi = i - 1, i
        w = ((rs - self.rs_tab.index_select(0, lo)) / (self.rs_tab.index_select(0, hi) - self.rs_tab.index_select(0, lo))).clamp(0.0, 1.0)
        b_u = (1 - w) * interp_rows(p, self.us_P.index_select(0, lo), self.us_b.index_select(0, lo)) + w * interp_rows(p, self.us_P.index_select(0, hi), self.us_b.index_select(0, hi))
        bm_u = (1 - w) * interp_rows(p, self.us_P.index_select(0, lo), self.us_bm.index_select(0, lo)) + w * interp_rows(p, self.us_P.index_select(0, hi), self.us_bm.index_select(0, hi))
        return b_u, b_u / bm_u

    def water(self, p):
        p_ref, Bw_ref, cw, mu_ref, cv = self.pvtw
        X = cw * (p - p_ref)
        Y = (cw - cv) * (p - p_ref)
        bw = (1.0 + X + 0.5 * X * X) / Bw_ref
        bm = (1.0 + Y + 0.5 * Y * Y) / (Bw_ref * mu_ref)
        return bw, bw / bm

    def rv_sat(self, p):
        if not self.wet:
            return 0.0 * p
        return interp1_lin(p, self.g_p, self.rv_tab).clamp_min(0.0)

    def _gas_2d(self, p, rv):
        n_t = self.g_p.numel()
        i = torch.searchsorted(self.g_p, p.detach().contiguous(), right=True).clamp(1, n_t - 1)
        lo, hi = i - 1, i
        w = (p - self.g_p.index_select(0, lo)) / (self.g_p.index_select(0, hi) - self.g_p.index_select(0, lo))
        b = (1 - w) * interp_rows(rv, self.ug_RV.index_select(0, lo), self.ug_b.index_select(0, lo)) + w * interp_rows(rv, self.ug_RV.index_select(0, hi), self.ug_b.index_select(0, hi))
        bm = (1 - w) * interp_rows(rv, self.ug_RV.index_select(0, lo), self.ug_bm.index_select(0, lo)) + w * interp_rows(rv, self.ug_RV.index_select(0, hi), self.ug_bm.index_select(0, hi))
        return b, bm

    def gas(self, p, rv=None, sat=None):
        """
        1/Bg и μg. Сухой газ (PVDG или PVTG без VAPOIL — Rv = 0); жирный: насыщенный (sat) или по Rv.
        За верхним краем таблицы — линейная экстраполяция (как Eclipse/OPM), но 1/Bg и 1/(Bg·μg) не ниже половины
        значений на краю: у жирного газа они к краю убывают, и линейно продолженная μg уходит в ноль и ниже.
        """
        b, bm = self._gas_b_bm(p, rv, sat)
        p_hi = getattr(self, "_g_p_hi", None)
        if p_hi is None:
            p_hi = self._g_p_hi = float(self.g_p[-1])
        # на GPU — без проверки (синхронизация дороже; при P ≤ края max(b, b/2) = b)
        if p.is_cuda or bool((p.detach() > p_hi).any()):
            ph = p.clamp(max=p_hi)
            rv_h = None if rv is None or not self.wet else torch.minimum(rv, self.rv_sat(ph))
            b_h, bm_h = self._gas_b_bm(ph, rv_h, sat)
            b, bm = torch.maximum(b, 0.5 * b_h), torch.maximum(bm, 0.5 * bm_h)
        return b, b / bm

    def _gas_b_bm(self, p, rv, sat):
        if not getattr(self, "pvtg", False):
            return interp1_lin(p, self.g_p, self.g_b), interp1_lin(p, self.g_p, self.g_bm)
        if not self.wet:
            return self._gas_2d(p, 0.0 * p)
        b_s, bm_s = interp1_lin(p, self.g_p, self.g_b), interp1_lin(p, self.g_p, self.g_bm)
        if rv is None:
            return b_s, bm_s
        # недонасыщенный газ: насыщенное значение + приращение 2D-таблицы от Rv_sat(P) до Rv —
        # ветви совпадают на линии насыщения точно (иначе переключения состояния дают скачки массы)
        b2, bm2 = self._gas_2d(p, rv)
        b2s, bm2s = self._gas_2d(p, self.rv_sat(p))
        b_u, bm_u = b_s + (b2 - b2s), bm_s + (bm2 - bm2s)
        if sat is not None:
            b_u, bm_u = torch.where(sat, b_s, b_u), torch.where(sat, bm_s, bm_u)
        return b_u, bm_u


class BlackOilPVT:
    """
    PVT black-oil на torch. Насыщенная ветвь — таблица по P. Недонасыщенная нефть (Rs < Rs_sat(P)):
    Bo = Bo_sat(Pb)·exp(-c_o (P - Pb)), μo = μo_sat(Pb)·exp(c_μ (P - Pb)), Pb = Rs_sat^-1(Rs);
    c_o и c_μ берутся из участка таблицы выше давления насыщения (где Rs уже не растёт), иначе c_oil.
    Без PVT-таблицы — мёртвая нефть и несмешивающийся газ с постоянными сжимаемостями.
    """

    def __init__(self, fluid, device):
        self.device = device
        dt = torch.float64
        self.rho_sc = (float(fluid.rho_oil_ref), float(fluid.rho_water_ref), float(fluid.rho_gas_ref))
        self.has_pvt = fluid.pvt is not None
        self.wet, self.rv_max = False, 0.0
        if not self.has_pvt:
            self.c = (fluid.oil_compressibility, fluid.water_compressibility, fluid.gas_compressibility)
            self.mu = (fluid.mu_oil, fluid.mu_water, fluid.mu_gas)
            self.rs_max = 0.0
            return
        t = fluid.pvt
        T = lambda a: torch.as_tensor(a, dtype=dt, device=device)
        self.P = T(t.P * 1e6)
        self.Bo, self.Bw, self.Bg = T(t.Bo), T(t.Bw), T(t.Bg)
        self.mu_o, self.mu_w, self.mu_g = T(t.mu_o * 1e-3), T(t.mu_w * 1e-3), T(t.mu_g * 1e-3)
        self.Rs = T(t.Rs)
        rs = t.Rs
        k = int(rs.argmax())
        keep = [0] + [i for i in range(1, k + 1) if rs[i] > rs[i - 1] * (1 + 1e-12)]
        self.inv_rs, self.inv_p = self.Rs[keep], self.P[keep]
        self.rs_max = float(rs[k])
        self.p_bmax = float(self.P[k])
        n = len(rs)
        if k < n - 1 and self.P[-1] > self.P[k]:
            dP = float(self.P[-1] - self.P[k])
            self.c_o = -math.log(float(self.Bo[-1] / self.Bo[k])) / dP
            self.c_mu = math.log(float(self.mu_o[-1] / self.mu_o[k])) / dP
        else:
            self.c_o = float(fluid.oil_compressibility)
            self.c_mu = 0.0

    def at(self, idx):
        return self

    def region(self, r):
        return self

    def rs_sat(self, p):
        if not self.has_pvt:
            return 0.0 * p
        return interp1(p, self.P, self.Rs)

    def oil(self, p, rs, sat):
        if not self.has_pvt:
            return 1.0 + self.c[0] * (p - 1e5), self.mu[0] + 0.0 * p
        Bo_s, mu_s = interp1(p, self.P, self.Bo), interp1(p, self.P, self.mu_o)
        pb = interp1(rs, self.inv_rs, self.inv_p)
        Bo_u = interp1(pb, self.P, self.Bo) * torch.exp(-self.c_o * (p - pb))
        mu_u = interp1(pb, self.P, self.mu_o) * torch.exp(self.c_mu * (p - pb))
        return 1.0 / torch.where(sat, Bo_s, Bo_u), torch.where(sat, mu_s, mu_u)

    def water(self, p):
        if not self.has_pvt:
            return 1.0 + self.c[1] * (p - 1e5), self.mu[1] + 0.0 * p
        return 1.0 / interp1(p, self.P, self.Bw), interp1(p, self.P, self.mu_w)

    def rv_sat(self, p):
        return 0.0 * p

    def gas(self, p, rv=None, sat=None):
        if not self.has_pvt:
            return 1.0 + self.c[2] * (p - 1e5), self.mu[2] + 0.0 * p
        return 1.0 / interp1(p, self.P, self.Bg), interp1(p, self.P, self.mu_g)


class ThreePhaseRelPerm:
    """
    Трёхфазные ОФП: krw(Sw), krg(Sg) и kro по Stone II (как в Eclipse STONE2):
        kro = krocw·[(krow/krocw + krw)(krog/krocw + krg) - krw - krg],  kro ≥ 0
    Двухфазные кривые — Corey по параметрам fluid.relative_permeability или таблицы SWOF/SGOF.
    При Sg = 0 kro совпадает с krow.

    ``kro_model='default'`` — модель Eclipse/OPM по умолчанию (без ключевых слов STONE):
        kro = (Sg·krog(Sg + Sw - Swc) + (Sw - Swc)·krow(Sw + Sg)) / (Sg + Sw - Swc)
    (krow и krog берутся при той же нефтенасыщенности So, что и в ячейке).
    """

    def __init__(self, fluid=None, device=None, tables=None, kro_model='stone2'):
        self.device = device
        self.tables = None
        self.kro_model = kro_model
        if tables is None and fluid is not None and getattr(fluid, 'relperm_model', 'corey') == 'table' \
                and fluid.relperm_tables:
            tables = fluid.relperm_tables
        if tables is not None:
            T = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
            self.tables = {}
            for key, tbl in tables.items():
                self.tables[key] = {k: T(v) for k, v in tbl.items()}
            sw_tab = self.tables.get('swof')
            self.krocw = float(sw_tab['kro'][0]) if sw_tab is not None else 1.0
            self.swc = float(sw_tab['sw'][0]) if sw_tab is not None else 0.0
        else:
            self.swc = float(fluid.sw_cr)
            self.sor = float(fluid.so_r)
            self.sgc = float(getattr(fluid, 'sg_cr', 0.0))
            self.nw = float(fluid.nw)
            self.now = float(getattr(fluid, 'now', fluid.no))
            self.ng = float(fluid.ng)
            self.nog = float(getattr(fluid, 'nog', fluid.ng))
            self.krw_end = float(getattr(fluid, 'krw_end', 1.0))
            self.krg_end = float(getattr(fluid, 'krg_end', 1.0))
            self.krocw = float(getattr(fluid, 'ko_end_w', 1.0))
            self.krog_end = float(getattr(fluid, 'ko_end_g', 1.0))
        self.pc_scale = float(getattr(fluid, 'pc_scale', 0.0))
        self.pc_exp = float(getattr(fluid, 'pc_exponent', 1.5))
        self.pcg_scale = float(getattr(fluid, 'pc_og_scale', 0.0))
        self.pcg_exp = float(getattr(fluid, 'pc_og_exponent', 1.5))

    def region(self, r):
        return self

    def _two_phase(self, sw, sg):
        """krw(Sw), krow(Sw), krg(Sg), krog(Sg)."""
        if self.tables is not None:
            tw, tg = self.tables.get('swof'), self.tables.get('sgof')
            krw = interp1(sw, tw['sw'], tw['krw']) if tw is not None else 0.0 * sw
            krow = interp1(sw, tw['sw'], tw['kro']) if tw is not None else 0.0 * sw + 1.0
            krg = interp1(sg, tg['sg'], tg['krg']) if tg is not None else 0.0 * sg
            krog = interp1(sg, tg['sg'], tg['kro']) if tg is not None else 0.0 * sg + self.krocw
            return krw, krow, krg, krog
        dw = 1.0 - self.swc - self.sor
        swn = torch.clamp((sw - self.swc) / dw, 0.0, 1.0)
        krw = self.krw_end * swn ** self.nw
        krow = self.krocw * (1.0 - swn) ** self.now
        sgn = torch.clamp((sg - self.sgc) / max(dw - self.sgc, 1e-12), 0.0, 1.0)
        krg = self.krg_end * sgn ** self.ng
        sgo = torch.clamp(sg / dw, 0.0, 1.0)
        krog = self.krog_end * (1.0 - sgo) ** self.nog
        return krw, krow, krg, krog

    def __call__(self, sw, sg):
        krw, krow, krg, krog = self._two_phase(sw, sg)
        if self.kro_model == 'default':
            sw_e = torch.clamp(sw, min=self.swc)
            _, krow_o, _, krog_o = self._two_phase(sw_e + sg, sg + sw_e - self.swc)
            den = sg + sw_e - self.swc
            small = den < 1e-12
            den_s = torch.where(small, torch.ones_like(den), den)
            kro = torch.where(small, krow_o, (sg * krog_o + (sw_e - self.swc) * krow_o) / den_s)
            return krw, torch.clamp(kro, min=0.0), krg
        c = self.krocw
        kro = c * ((krow / c + krw) * (krog / c + krg) - krw - krg)
        return krw, torch.clamp(kro, min=0.0), krg

    def pc(self, sw, sg):
        """Pcow = Po - Pw, Pcog = Pg - Po (те же определения, что в Fluid)."""
        if self.tables is not None:
            tw, tg = self.tables.get('swof'), self.tables.get('sgof')
            pcow = interp1(sw, tw['sw'], tw['pcow']) if tw is not None else 0.0 * sw
            pcog = interp1(sg, tg['sg'], tg['pcog']) if tg is not None else 0.0 * sg
            return pcow, pcog
        dw = 1.0 - self.swc - self.sor
        pcow = 0.0 * sw
        pcog = 0.0 * sg
        if self.pc_scale:
            swn = torch.clamp((sw - self.swc) / dw, 0.0, 1.0)
            pcow = self.pc_scale * (1.0 - swn + 1e-6) ** (-self.pc_exp)
        if self.pcg_scale:
            sgn = torch.clamp(sg / dw, 0.0, 1.0)
            pcog = self.pcg_scale * (1.0 - sgn + 1e-6) ** (-self.pcg_exp)
        return pcow, pcog
