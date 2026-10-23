"""
Полностью неявный трёхфазный black-oil (FIM): нефть, вода, газ с растворённым газом Rs.

Неизвестные в ячейке: P (давление нефти), Sw и переменная X с переключением (как в OPM Flow):
  насыщенная ячейка (нефть и газ):            X = Sg,  Rs = Rs_sat(P), Rv = Rv_sat(P)
  недонасыщенная нефть (свободного газа нет): X = Rs,  Sg = 0
  недонасыщенный газ (нефти нет, VAPOIL):     X = Rv,  Sg = 1 − Sw
Уравнения — массовые балансы компонентов [кг/с]:
  нефть: d/dt(φ (So bo + Sg bg Rv) ρo_sc) + div(ρo_sc (bo kro/μo ∇Φo + Rv bg krg/μg ∇Φg))
  вода:  d/dt(φ Sw bw ρw_sc) + div(ρw_sc bw krw/μw ∇Φw)
  газ:   d/dt(φ (Sg bg + So bo Rs) ρg_sc) + div(ρg_sc (bg krg/μg ∇Φg + Rs bo kro/μo ∇Φo))
где b = 1/B. Свойства ячейки и их производные по (P, Sw, X) берутся прямым режимом AD (torch.func.jvp),
потоки по граням и скважины собираются вручную с upwind по потенциалу каждой фазы.
"""
import torch
import torch.nn.functional as F
from torch.func import jvp

from .blackoil_props import BlackOilPVT, ThreePhaseRelPerm
from .fim_wells import FIMWells, G, DAY
from .manual_assembly_solver import ManualFIMSolver
from .sparse_utils import SparseMatrixBuilder

OIL, WAT, GAS = 0, 1, 2
PHASE_IDX = {"oil": OIL, "water": WAT, "gas": GAS}

# строки массива свойств ячейки
I_ACC = 0          # 0..2  массы компонентов в ячейке, кг
I_P = 3            # 3..5  давления фаз o, w, g
I_RHO = 6          # 6..8  пластовые плотности фаз
I_MOO, I_MWW, I_MGG, I_MGO = 9, 10, 11, 12  # массовые подвижности компонент/фаза
I_LT = 13          # суммарная объёмная подвижность Σ kr/μ
I_B = 14           # 14..16 bo, bw, bg
I_MOG = 17         # масса нефти (испарённой, Rv) в газовой фазе
N_PROPS = 18
# (компонент, фаза, строка подвижности)
FLUX_TERMS = ((OIL, OIL, I_MOO), (WAT, WAT, I_MWW), (GAS, GAS, I_MGG), (GAS, OIL, I_MGO))
FLUX_TERMS_WET = FLUX_TERMS + ((OIL, GAS, I_MOG),)
# зона нечувствительности переключения переменных: без «дребезга» ячеек на линии насыщения от шума округления
SWITCH_EPS_S = 1e-12   # по насыщенности
SWITCH_EPS_R = 1e-10   # относительная, по Rs / Rv
# ячейка без нефти и газа (Sw ≥ 1 − SO_EMPTY) ведётся по Sg, а не по Rs (как в OPM): масса газа So·bo·Rs
# при So = 0 не зависит от Rs, и пришедший газ (нагнетательная в водоносной ячейке) не может появиться
SO_EMPTY = 1e-6
# обратно на Rs — только при заметной нефти: иначе водоносные ячейки (Sw ≈ 1, Sg ≈ 0) дребезжат на пороге
SO_FILLED = 1e-4
# ячейка, сменившая переменные столько раз за шаг, до конца шага не переключается (иначе Ньютон зацикливается)
MAX_CELL_SWITCHES = 3

def surface_to_reservoir(pvt, p, qo, qw, qg, rs_max=None, rv_max=None):
    """
    Пластовые объёмы фаз по поверхностным (Qo, Qw, Qg) при давлении p (разгазирование и выпадение конденсата):
    Rs = min(Rs_max, Qg/Qo), Rv = min(Rv_max, Qo/Qg), D = 1 − Rs·Rv,
    Vo = (Qo − Rv·Qg)/(bo·D), Vg = (Qg − Rs·Qo)/(bg·D), Vw = Qw/bw.
    Rs_max, Rv_max по умолчанию — насыщенные при p.
    """
    rs_s, rv_s = pvt.rs_sat(p), pvt.rv_sat(p)
    rs_cap = rs_s if rs_max is None else torch.minimum(rs_s, rs_max)
    rv_cap = rv_s if rv_max is None else torch.minimum(rv_s, rv_max)
    rs = torch.where(qo > 0, torch.minimum(rs_cap, qg / qo.clamp_min(1e-300)), torch.zeros_like(p))
    rv = torch.where(qg > 0, torch.minimum(rv_cap, qo / qg.clamp_min(1e-300)), torch.zeros_like(p))
    bo, _ = pvt.oil(p, rs, rs >= rs_s - 1e-12)
    bw = pvt.water(p)[0]
    bg = pvt.gas(p, rv, rv >= rv_s - 1e-12)[0] if getattr(pvt, "wet", False) else pvt.gas(p)[0]
    D = (1.0 - rs * rv).clamp_min(1e-12)
    vo = ((qo - rv * qg) / (bo * D)).clamp_min(0.0)
    vg = ((qg - rs * qo) / (bg * D)).clamp_min(0.0)
    return vo, qw / bw, vg


class BlackOilWells(FIMWells):
    """Скважины трёхфазной модели: те же p_bh-неизвестные, режимы и дополнение Шура, что в FIMWells."""

    def __init__(self, well_manager, reservoir, fluid, dims, device):
        super().__init__(well_manager, reservoir, fluid, dims, device)
        self.rho_sc = torch.tensor([fluid.rho_oil_ref, fluid.rho_water_ref, fluid.rho_gas_ref],
                                   dtype=torch.float64, device=device)
        inj_ph, weights = [], []
        for w in self.wells:
            a = [0.0, 0.0, 0.0]
            if w.type == 'injector':
                ph = PHASE_IDX.get((w.injected_phase or 'water').lower())
                if ph is None or ph == OIL:
                    raise ValueError(f"Скважина {w.name}: закачка '{w.injected_phase}' не поддерживается (water|gas)")
                a[ph] = -1.0
                inj_ph.append(ph)
            else:
                ph = (w.surface_phase or 'liquid').lower()
                sel = {'oil': (OIL,), 'water': (WAT,), 'gas': (GAS,), 'liquid': (OIL, WAT)}.get(ph)
                if sel is None:
                    raise ValueError(f"Скважина {w.name}: surface_phase '{w.surface_phase}' не поддерживается")
                for s in sel:
                    a[s] = 1.0
                inj_ph.append(OIL)
            weights.append(a)
        self.inj_phase = torch.tensor(inj_ph, dtype=torch.long, device=device)
        self.rate_w = torch.tensor(weights, dtype=torch.float64, device=device) / self.rho_sc
        self.pvt = None
        self.wellbore_density = 'flash'   # 'flash' (OPM/Eclipse) | 'cell' (средняя по ячейкам)
        self.conn_head = torch.zeros_like(self.conn_dz)
        self.conn_open = torch.ones_like(self.conn_wi)   # 0 — соединения закрытой скважины (SHUT/STOP)
        self.conn_a = self.rate_w[self.conn_well]         # веса уравнения дебита по соединениям (на шаг)
        self.p_avg = None                                 # среднее пластовое давление (для RESV), Па
        self.rs_avg = self.rv_avg = None                  # средние Rs, Rv (для RESV)
        self.flux_terms = FLUX_TERMS
        self.groups = None                                # (GroupTree, [ограничения]) — групповые контроли
        self.group_controlled = set()
        self.reinit_bhp = set()                           # скважины, сменившие тип: p_bh — заново с начального

    def set_surface_phase(self, w_idx, phase):
        w = self.wells[w_idx]
        if w.type == 'injector':
            return
        sel = {'oil': (OIL,), 'water': (WAT,), 'gas': (GAS,), 'liquid': (OIL, WAT)}.get((phase or 'liquid').lower())
        if sel is None:
            raise ValueError(f"Скважина {w.name}: surface_phase '{phase}' не поддерживается")
        a = torch.zeros(3, dtype=torch.float64, device=self.device)
        a[list(sel)] = 1.0
        self.rate_w[w_idx] = a / self.rho_sc

    def set_type(self, w_idx, well_type, injected_phase=None, surface_phase=None):
        """Смена типа скважины (добыча ↔ закачка) или фазы закачки по расписанию."""
        w = self.wells[w_idx]
        if well_type == 'injector':
            ph = PHASE_IDX.get((injected_phase or 'water').lower())
            if ph is None or ph == OIL:
                raise ValueError(f"Скважина {w.name}: закачка '{injected_phase}' не поддерживается (water|gas)")
        elif well_type != 'producer':
            raise ValueError(f"Скважина {w.name}: тип {well_type}")
        w.type = well_type
        w.injected_phase = injected_phase if well_type == 'injector' else w.injected_phase
        self.is_inj[w_idx] = well_type == 'injector'
        if well_type == 'injector':
            self.inj_phase[w_idx] = ph
            a = torch.zeros(3, dtype=torch.float64, device=self.device)
            a[ph] = -1.0
            self.rate_w[w_idx] = a / self.rho_sc
        else:
            self.inj_phase[w_idx] = OIL
            w.surface_phase = surface_phase
            self.set_surface_phase(w_idx, surface_phase)
        self.mode[w_idx] = w.control_type
        self.reinit_bhp.add(w_idx)

    def set_open(self, w_idx, is_open):
        self.conn_open[self.conn_well == w_idx] = 1.0 if is_open else 0.0

    def _apply_groups(self, v, q, dqb):
        """Групповые цели → цели скважин (явно на шаг): guide rate = потенциал при предельном BHP."""
        from .groups import group_targets
        tree, constraints = self.groups
        if not constraints:
            return
        cw = self.conn_well
        rho = self.rho_sc
        zeros = torch.zeros(self.n_w, dtype=torch.float64, device=self.device)
        rw = self._resv_weights(v)
        lim = torch.tensor([self.bhp_limit(w) for w in self.wells], dtype=torch.float64, device=self.device)
        pb = self.p_bh if self.p_bh is not None else lim
        sign = torch.where(self.is_inj, -1.0, 1.0).double()

        def per_well(weights, p):
            q0 = zeros.clone().index_add_(0, cw, (weights * q).sum(1))
            D = zeros.clone().index_add_(0, cw, (weights * dqb).sum(1))
            return (sign * (q0 + D * p)).clamp_min(0.0).tolist()

        onehot = [torch.zeros(3, dtype=torch.float64, device=self.device) for _ in range(3)]
        for m in range(3):
            onehot[m][m] = 1.0 / rho[m]
        pot_ph = [per_well(onehot[m].expand_as(q), lim) for m in range(3)]
        cur_ph = [per_well(onehot[m].expand_as(q), pb) for m in range(3)]
        pot_res, cur_res = per_well(rw, lim), per_well(rw, pb)
        pot, cur, prod_res, info = {}, {}, {}, []
        names = {0: "oil", 1: "water", 2: "gas"}
        for i, w in enumerate(self.wells):
            pot[w.name] = {names[m]: pot_ph[m][i] for m in range(3)} | {"resv": pot_res[i]}
            cur[w.name] = {names[m]: cur_ph[m][i] for m in range(3)}
            prod_res[w.name] = cur_res[i]
            ctrl, val = self.step_targets[i]
            if w.type == 'injector':
                phase = (w.injected_phase or 'water').lower() if ctrl == 'rate' else None
            else:
                phase = 'resv' if i in self._resv_idx else \
                    {'rate': (w.surface_phase or 'liquid').lower()}.get(ctrl)
            grup = bool(getattr(w, 'grup', False))
            info.append(dict(name=w.name, type=w.type, open=ctrl != 'shut', grup=grup,
                             phase=None if grup else phase, value=None if grup else val,
                             inj_phase=(w.injected_phase or 'water').lower() if w.type == 'injector' else None))
        cons = [dict(c, target=c["target"] / DAY) for c in constraints]
        new = group_targets(tree, cons, info, pot, cur, prod_res)
        self.group_controlled = set(new)
        for name, (ph, val) in new.items():
            i = next(k for k, w in enumerate(self.wells) if w.name == name)
            m = cw == i
            if ph == 'resv':
                a = rw[m]
            elif self.wells[i].type == 'injector':
                a = -onehot[{'water': WAT, 'gas': GAS}[ph]].expand(int(m.sum()), 3)
            else:
                sel = {'oil': (OIL,), 'water': (WAT,), 'gas': (GAS,), 'liquid': (OIL, WAT)}[ph]
                a = sum(onehot[k] for k in sel).expand(int(m.sum()), 3)
            self.conn_a[m] = a
            self.step_targets[i] = ('rate', val)

    def _resv_weights(self, v):
        """
        Пересчёт масс компонентов в пластовый объём на соединениях: q_res = Σ a_c q_c. Как в Eclipse/OPM
        (RateConverter::calcCoeff) — при средних по углеводородному поровому объёму давлении, Rs и Rv
        (PVT-регион соединения); без средних — по состоянию ячеек соединений.
        """
        c = self.conn_cell
        rho = self.rho_sc
        if self.p_avg is not None and self.rs_avg is not None and self.pvt is not None:
            pvt = self.pvt.at(c)
            p = torch.full((c.numel(),), float(self.p_avg), dtype=torch.float64, device=self.device)
            rs_s = pvt.rs_sat(p)
            rs = torch.minimum(torch.full_like(p, float(self.rs_avg)), rs_s)
            bo = pvt.oil(p, rs, rs >= rs_s)[0]
            bw = pvt.water(p)[0]
            if getattr(pvt, "wet", False) and self.rv_avg is not None:
                rv_s = pvt.rv_sat(p)
                rv = torch.minimum(torch.full_like(p, float(self.rv_avg)), rv_s)
                bg = pvt.gas(p, rv, rv >= rv_s)[0]
            else:
                rv = torch.zeros_like(p)
                bg = pvt.gas(p)[0]
        else:
            bo, bw, bg = (v[I_B + m][c].clamp_min(1e-12) for m in range(3))
            rs = torch.where(v[I_MOO][c] > 0, v[I_MGO][c] / v[I_MOO][c].clamp_min(1e-300) * rho[OIL] / rho[GAS],
                             torch.zeros_like(bo))
            rv = torch.where(v[I_MGG][c] > 0, v[I_MOG][c] / v[I_MGG][c].clamp_min(1e-300) * rho[GAS] / rho[OIL],
                             torch.zeros_like(bo))
        D = (1.0 - rs * rv).clamp_min(1e-12)
        return torch.stack([(1.0 / bo - rs / bg) / (rho[OIL] * D), 1.0 / (rho[WAT] * bw),
                            (1.0 / bg - rv / bo) / (rho[GAS] * D)], dim=1)

    def _resv_target(self, w, p):
        """
        Пластовый объём исторических поверхностных дебитов (Qo, Qw, Qg), м³/с — как в Eclipse/OPM
        (SurfaceToReservoirVoidage): давление p и ограничения Rs, Rv — средние по углеводородному поровому
        объёму пласта; Rs = min(Qg/Qo, Rs_ср), Rv = min(Qo/Qg, Rv_ср).
        """
        qo, qw, qg = (x / DAY for x in w.hist_rates)
        pt = torch.tensor([float(p)], dtype=torch.float64, device=self.device)
        pvt = self.pvt.at(self.conn_cell[self.conn_well == self.wells.index(w)][:1])
        T = lambda a: torch.tensor([a], dtype=torch.float64, device=self.device)
        rs_max = None if self.rs_avg is None else T(self.rs_avg)
        rv_max = None if self.rv_avg is None else T(self.rv_avg)
        vo, vw, vg = surface_to_reservoir(pvt, pt, T(qo), T(qw), T(qg), rs_max, rv_max)
        return float(vo + vw + vg)

    def _conn_flows(self, v, d, p_bh, shut_backflow=True):
        """Массовые потоки компонентов по перфорациям q [n_conn,3], dq/dx [n_conn,3,3], dq/dp_bh [n_conn,3]."""
        c, cw, wi = self.conn_cell, self.conn_well, self.conn_wi * self.conn_open * self.conn_sched
        n_c = c.numel()
        p_wc = p_bh[cw] + self.conn_head
        q = torch.zeros((n_c, 3), dtype=torch.float64, device=self.device)
        dq = torch.zeros((n_c, 3, 3), dtype=torch.float64, device=self.device)
        dqb = torch.zeros((n_c, 3), dtype=torch.float64, device=self.device)
        for comp, ph, mi in self.flux_terms:
            M, dM = v[mi][c], d[mi][c]
            dpp = v[I_P + ph][c] - p_wc
            q[:, comp] += wi * M * dpp
            dq[:, comp, :] += wi.unsqueeze(1) * (dM * dpp.unsqueeze(1) + M.unsqueeze(1) * d[I_P + ph][c])
            dqb[:, comp] -= wi * M

        inj = self.is_inj[cw]
        if bool(inj.any()):
            ph = self.inj_phase[cw]
            E = F.one_hot(ph, 3).double()
            b = v[I_B:I_B + 3][:, c].gather(0, ph.unsqueeze(0)).squeeze(0)
            db = d[I_B:I_B + 3][:, c, :].gather(0, ph.view(1, -1, 1).expand(1, n_c, 3)).squeeze(0)
            lt, dlt = v[I_LT][c], d[I_LT][c]
            rho = self.rho_sc[ph]
            dpp = v[I_P][c] - p_wc
            m = wi * rho * b * lt
            dm = (wi * rho).unsqueeze(1) * (db * lt.unsqueeze(1) + b.unsqueeze(1) * dlt)
            qi = E * (m * dpp).unsqueeze(1)
            dqi = E.unsqueeze(2) * (dm * dpp.unsqueeze(1) + m.unsqueeze(1) * d[I_P][c]).unsqueeze(1)
            # перфорации нагнетательной с обратным потоком (пласт → ствол) закрыты, как при crossflow 'NO':
            # смешения в стволе нет, и обратный поток либо уходил бы на поверхность (потеря массы), либо нёс
            # закачиваемую фазу, которой в ячейке может не быть
            back = inj & (dpp > 0.0) if shut_backflow else torch.zeros_like(inj)
            qi = torch.where(back.unsqueeze(1), torch.zeros_like(qi), qi)
            dqi = torch.where(back.view(-1, 1, 1), torch.zeros_like(dqi), dqi)
            # dq/dp_bh не обнуляется: иначе у скважины на дебите со всеми перфорациями в обратном потоке
            # уравнение вырождается; Ньютон поднимает забой до начала закачки
            q = torch.where(inj.unsqueeze(1), qi, q)
            dq = torch.where(inj.view(-1, 1, 1), dqi, dq)
            dqb = torch.where(inj.unsqueeze(1), -E * m.unsqueeze(1), dqb)
        return q, dq, dqb

    def begin_step(self, v, d, t_days):
        if self.n_w == 0:
            return
        self.step_targets = self.targets(t_days)
        self.step_mode = []
        self.switches = 0
        c, cw = self.conn_cell, self.conn_well
        a = self.rate_w[cw].clone()
        resv_w = None
        self._resv_idx = set()
        for w_idx, w in enumerate(self.wells):
            ctrl, val = self.step_targets[w_idx]
            if ctrl != 'resv':
                continue
            if resv_w is None:
                resv_w = self._resv_weights(v)
            m = cw == w_idx
            a[m] = resv_w[m]
            if getattr(w, 'hist_rates', None) is not None:
                p_ref = self.p_avg if self.p_avg is not None else float(v[I_P][c[m]].mean())
                val = self._resv_target(w, p_ref)
            self.step_targets[w_idx] = ('rate', val)
            self._resv_idx.add(w_idx)
        self.conn_a = a
        lam = torch.stack([v[I_MOO], v[I_MWW], v[I_MGG]])[:, c] / (self.rho_sc.view(3, 1) * v[I_B:I_B + 3][:, c])
        rho = v[I_RHO:I_RHO + 3][:, c]
        mix = (lam * rho).sum(0) / lam.sum(0).clamp_min(1e-30)
        rho_inj = rho.gather(0, self.inj_phase[cw].unsqueeze(0)).squeeze(0)
        mix = torch.where(self.is_inj[cw], rho_inj, mix)
        zeros = torch.zeros(self.n_w, dtype=torch.float64, device=self.device)
        cnt = zeros.clone().index_add_(0, cw, torch.ones_like(mix))
        self.rho_wb = zeros.clone().index_add_(0, cw, mix) / cnt.clamp_min(1)
        if self.pvt is not None and self.wellbore_density == 'flash':
            self.conn_head = self._flash_heads(v, mix)
        else:
            self.conn_head = self.rho_wb[cw] * G * self.conn_dz

        # дебит линеен по p_bh при замороженной ячейке: q(pb) = q(0) + D·pb
        q, _, dqb = self._conn_flows(v, d, zeros, shut_backflow=False)
        if self.groups is not None:
            self._apply_groups(v, q, dqb)
        q0 = zeros.clone().index_add_(0, cw, (a * q).sum(1))
        D = zeros.clone().index_add_(0, cw, (a * dqb).sum(1))
        p_init = torch.zeros_like(zeros)
        for w_idx, w in enumerate(self.wells):
            ctrl, val = self.step_targets[w_idx]
            mode = 'bhp' if ctrl == 'bhp' else ('bhp_limit' if self.mode[w_idx] == 'bhp_limit' else 'rate')
            if ctrl == 'shut':
                mode = 'shut'
                p_init[w_idx] = float(v[I_P][c[cw == w_idx]].mean())
            elif mode != 'rate':
                p_init[w_idx] = val if ctrl == 'bhp' else self.bhp_limit(w)
            else:
                Dw = float(D[w_idx])
                pb = (val - float(q0[w_idx])) / Dw if abs(Dw) > 1e-30 else float(v[I_P][c[cw == w_idx]].mean())
                p_init[w_idx] = self._clamp_bhp(w, torch.tensor(pb, dtype=torch.float64, device=self.device))
            self.step_mode.append(mode)
        if self.p_bh is None:
            self.p_bh = p_init
        else:
            fixed = torch.tensor([m != 'rate' or k in self.reinit_bhp for k, m in enumerate(self.step_mode)],
                                 device=self.device)
            self.p_bh = torch.where(fixed, p_init, self.p_bh)
        self.reinit_bhp = set()

    def _flash_heads(self, v, mix_cell):
        """
        Гидростатика ствола по соединениям (как StandardWell в OPM): поверхностные притоки накапливаются
        снизу вверх, плотность смеси на соединении — при давлении ствола с разгазированием
        (Rs = min(Rs_sat(p), Qg/Qo)); участок между соединениями несёт смесь от нижележащих перфораций.
        Явно по состоянию начала шага. Возвращает p_conn − p_bh, Па.
        """
        c, cw, wi = self.conn_cell, self.conn_well, self.conn_wi * self.conn_sched
        rho_sc = self.rho_sc
        qo = wi * (v[I_MOO][c] + v[I_MOG][c]) / rho_sc[OIL]
        qw = wi * v[I_MWW][c] / rho_sc[WAT]
        qg = wi * (v[I_MGG][c] + v[I_MGO][c]) / rho_sc[GAS]
        p_cell = v[I_P][c]
        head_old = self.conn_head if self.p_bh is not None else None
        if head_old is not None:
            dd = (p_cell - self.p_bh[cw] - head_old).clamp_min(0.0)
            use = torch.zeros(self.n_w, dtype=torch.float64, device=self.device).index_add_(0, cw, dd)[cw] > 0
            dd = torch.where(use, dd, torch.ones_like(dd))
        else:
            dd = torch.ones_like(p_cell)
        qo, qw, qg = qo * dd, qw * dd, qg * dd
        p_bh = self.p_bh if self.p_bh is not None else torch.zeros(self.n_w, dtype=torch.float64, device=self.device)
        heads = torch.zeros_like(p_cell)
        dz_all = self.conn_dz
        for w_idx, w in enumerate(self.wells):
            idx = (cw == w_idx).nonzero().view(-1)
            if w.type == 'injector' or idx.numel() == 0:
                heads[idx] = mix_cell[idx] * G * dz_all[idx]
                continue
            # ствол — только открытые соединения (как в OPM); закрытым напор не нужен (WI = 0)
            op = idx[self.conn_sched[idx] > 0]
            if op.numel() == 0:
                heads[idx] = mix_cell[idx] * G * dz_all[idx]
                continue
            heads[idx] = mix_cell[idx] * G * dz_all[idx]
            idx = op
            order = idx[torch.argsort(dz_all[idx])]            # сверху вниз
            Qo = torch.flip(torch.cumsum(torch.flip(qo[order], [0]), 0), [0])
            Qw = torch.flip(torch.cumsum(torch.flip(qw[order], [0]), 0), [0])
            Qg = torch.flip(torch.cumsum(torch.flip(qg[order], [0]), 0), [0])
            mass = Qo * rho_sc[OIL] + Qw * rho_sc[WAT] + Qg * rho_sc[GAS]
            dz = dz_all[order]
            if head_old is None:
                h = mix_cell[order] * G * dz
                pb = (p_cell[order] - h).min()
            else:
                h = head_old[order]
                pb = p_bh[w_idx]
            for _ in range(2):
                p = (pb + h).clamp_min(1e5)
                vo, vw, vg = surface_to_reservoir(self.pvt.at(c[order]), p, Qo, Qw, Qg)
                vol = vo + vw + vg
                rho = torch.where((mass > 0) & (vol > 0), mass / vol.clamp_min(1e-300), mix_cell[order])
                seg = torch.diff(dz, prepend=torch.zeros(1, dtype=dz.dtype, device=dz.device))
                h = torch.cumsum(rho * G * seg, 0)
            heads[order] = h
        return heads

    def assemble(self, R, J_vals, diag_map, v, d, p_bh):
        if self.n_w == 0:
            return None
        if self.step_targets is None:
            raise RuntimeError("BlackOilWells.begin_step() не вызван")
        c, cw = self.conn_cell, self.conn_well
        q, dq, Cv = self._conn_flows(v, d, p_bh)
        R.view(-1, 3).index_add_(0, c, q)
        jd = diag_map[c]
        J_vals.index_add_(0, jd, dq)

        a = self.conn_a
        zeros = torch.zeros(self.n_w, dtype=torch.float64, device=self.device)
        q_surf = zeros.clone().index_add_(0, cw, (a * q).sum(1))
        dq_dpbh = zeros.clone().index_add_(0, cw, (a * Cv).sum(1))
        B_rate = (a.unsqueeze(2) * dq).sum(1)

        is_rate = torch.zeros(self.n_w, dtype=torch.bool, device=self.device)
        target = torch.zeros_like(zeros)
        for w_idx, w in enumerate(self.wells):
            ctrl, val = self.step_targets[w_idx]
            mode = self.step_mode[w_idx]
            if ctrl == 'rate' and mode == 'rate':
                is_rate[w_idx] = True
                target[w_idx] = val
            elif ctrl == 'shut':
                target[w_idx] = p_bh[w_idx]
            elif ctrl == 'bhp':
                target[w_idx] = val
            else:
                target[w_idx] = self.bhp_limit(w)
        r_w = torch.where(is_rate, q_surf - target, p_bh - target)
        D = torch.where(is_rate, dq_dpbh, torch.ones_like(zeros))
        degenerate = is_rate & (D.abs() < 1e-30)
        D = torch.where(degenerate, torch.ones_like(D), D)
        r_w = torch.where(degenerate, torch.zeros_like(r_w), r_w)
        conn_rate = is_rate[cw] & ~degenerate[cw]
        Bv = torch.where(conn_rate.unsqueeze(1), B_rate, torch.zeros_like(B_rate))
        return {"Cv": Cv, "Bv": Bv, "D": D, "r_w": r_w, "q_surf": q_surf, "is_rate": is_rate,
                "target": target, "q": q, "q_o": q[:, OIL], "q_w": q[:, WAT], "q_g": q[:, GAS], "diag_pos": jd}

    def records(self, sys, p_bh, p):
        recs = []
        for w_idx, w in enumerate(self.wells):
            m = self.conn_well == w_idx
            recs.append({"name": w.name, "type": w.type, "mode": self.step_mode[w_idx],
                         "p_bh_mpa": float(p_bh[w_idx]) / 1e6,
                         "q_surf_m3_day": float(sys["q_surf"][w_idx]) * DAY,
                         "q_kg_s": sys["q"][m].sum(0).tolist(), "eq_res": float(sys["r_w"][w_idx]),
                         "p_cells_mpa": [float(x) / 1e6 for x in p.view(-1)[self.conn_cell[m]]]})
        return recs


class BlackOilFIMSolver(ManualFIMSolver):
    """FIM black-oil: блоки Якобиана 3×3 (уравнения o, w, g × переменные P, Sw, X)."""

    nb = 3

    def __init__(self, reservoir, fluid, well_manager, params):
        self.reservoir = reservoir
        self.fluid = fluid
        self.well_manager = well_manager
        self.device = reservoir.device
        self.nx, self.ny, self.nz = reservoir.dimensions
        self.n_cells = self.nx * self.ny * self.nz
        self._init_params(params)
        self.drs_max_rel = float(params.get('newton_drs_max_rel', 0.2))
        self.relax_min = float(params.get('newton_relax_min', 0.5))
        self.relax_increment = float(params.get('newton_relax_increment', 0.1))
        self.relax_rel_tol = float(params.get('newton_relax_rel_tol', 0.2))
        self.debug.eq_names = ("oil", "water", "gas")

        print("Building sparse matrix structure (black-oil, блоки 3x3)...")
        self.builder = SparseMatrixBuilder(self.nx, self.ny, self.nz, block_size=3, device=self.device,
                                           pairs=self._conn_pairs())
        dev = self.device
        self.pore_volume = (reservoir.cell_volume * reservoir.porosity).reshape(-1).double().to(dev)
        c_rock = getattr(fluid, 'rock_compressibility', 0.0)
        self.c_rock = (c_rock.reshape(-1).double().to(dev) if torch.is_tensor(c_rock) else float(c_rock))
        p_ref = getattr(fluid, 'rock_ref_pressure', None)
        self.rock_eclipse = p_ref is not None
        if self.rock_eclipse:
            self.p_ref_rock = (p_ref.reshape(-1).double().to(dev) if torch.is_tensor(p_ref)
                               else torch.full((self.n_cells,), float(p_ref), dtype=torch.float64, device=dev))
        else:
            self.p_ref_rock = fluid.pressure.detach().reshape(-1).double().clone().to(dev)
        self.pvt = getattr(fluid, 'blackoil_pvt', None) or BlackOilPVT(fluid, dev)
        self.kr = getattr(fluid, 'blackoil_relperm', None) or ThreePhaseRelPerm(fluid, dev)
        drsdt = params.get('drsdt')
        self.drsdt = None if drsdt is None else float(drsdt)
        self._rs_cap = None
        self.rho_sc = torch.tensor(self.pvt.rho_sc, dtype=torch.float64, device=dev)
        vp = getattr(fluid, 'vappars', None)
        if vp is not None and float(vp[1]) != 0.0:
            raise NotImplementedError("VAPPARS: VAP2 ≠ 0 не поддерживается")
        self.vap1 = float(vp[0]) if vp is not None and float(vp[0]) > 0.0 else None

        self._build_connections()
        self.wells = BlackOilWells(well_manager, reservoir, fluid, reservoir.dimensions, dev)
        self.wells.pvt = self.pvt
        self.wet = bool(getattr(self.pvt, "wet", False))
        self.flux_terms = FLUX_TERMS_WET if self.wet else FLUX_TERMS
        self.wells.flux_terms = self.flux_terms
        self.wells.wellbore_density = str(params.get('wellbore_density', 'flash')).lower()
        for w in self.wells.wells:
            print(f"  Скважина {w.name}: перфораций {len(w.perforations)}, WI = "
                  + ", ".join(f"{x:.3e}" for x in w.connection_wi) + " м³")
        self._p_bh_cur = None

        p = fluid.pressure.detach().reshape(-1).double().to(dev)
        sg = fluid.s_g.detach().reshape(-1).double().to(dev)
        rs = getattr(fluid, 'rs', None)
        rs = self.pvt.rs_sat(p) if rs is None else rs.detach().reshape(-1).double().to(dev)
        self.sat = (sg > 0) | torch.full_like(sg, not self.pvt.has_pvt, dtype=torch.bool)
        rs = torch.where(self.sat, self.pvt.rs_sat(p), torch.minimum(rs, self.pvt.rs_sat(p)))
        self.x = torch.where(self.sat, sg, rs)
        sw0 = fluid.s_w.detach().reshape(-1).double().to(dev)
        rv = getattr(fluid, 'rv', None)
        rv_s = self.pvt.rv_sat(p)
        rv = rv_s if rv is None else torch.minimum(rv.detach().reshape(-1).double().to(dev), rv_s)
        self.ug = (sg > 0) & (1.0 - sw0 - sg <= 1e-12) & self.wet
        self.sat = self.sat & ~self.ug
        if self.pvt.has_pvt:
            empty = (1.0 - sw0 - sg <= SO_EMPTY) & ~self.ug & ~self.sat
            self.sat = self.sat | empty
            self.x = torch.where(empty, sg, self.x)
            rs = torch.where(empty, self.pvt.rs_sat(p), rs)
        self.x = torch.where(self.ug, rv, self.x)
        fluid.rs = rs.view(self.nx, self.ny, self.nz)
        fluid.rv = torch.where(self.ug, rv, rv_s).view(self.nx, self.ny, self.nz)
        self.so_max = (1.0 - sw0 - sg).clamp_min(0.0)   # VAPPARS: максимальная So ячейки (на начало шага)
        print(f"  Black-oil FIM: PVT={'таблица' if self.pvt.has_pvt else 'постоянные сжимаемости'}, "
              f"насыщенных ячеек {int(self.sat.sum())}/{self.n_cells}"
              + (f", газ без нефти (X = Rv) {int(self.ug.sum())}" if self.wet else ""))

    # ------------------------------------------------------------ свойства
    def _pv(self, p_flat):
        if not self.rock_eclipse:
            return super()._pv(p_flat)
        X = self.c_rock * (p_flat - self.p_ref_rock)
        return self.pore_volume * (1.0 + X + 0.5 * X * X)

    def rs_sat_eff(self, p):
        """Rs в насыщенной ячейке: Rs_sat(P), ограниченное сверху DRSDT (Rs_old + DRSDT·dt)."""
        rs = self.pvt.rs_sat(p)
        return rs if self._rs_cap is None else torch.minimum(rs, self._rs_cap)

    def rv_sat_eff(self, p, so):
        """
        Rv насыщенного газа: Rv_sat(P), при VAPPARS — ·(max(So, 1e-3)/So_max)^VAP1, если So < So_max и So_max > 0.01
        (как saturatedVaporizationFactor в OPM): остаточная нефть испаряется хуже.
        """
        rv = self.pvt.rv_sat(p)
        if self.vap1 is None:
            return rv
        som = self.so_max
        f = (so.clamp_min(1e-3) / som.clamp_min(1e-30)) ** self.vap1
        return rv * torch.where((som > 0.01) & (so < som), f, torch.ones_like(f))

    def _phase_state(self, p, sw, x, sat, ug):
        """Насыщенности, Rs/Rv, объёмные коэффициенты и вязкости фаз ячейки (без ОФП) и массы компонентов."""
        pvt = self.pvt
        rho_o, rho_w, rho_g = pvt.rho_sc
        cap = self._rs_cap
        wet = self.wet and ug is not None
        if wet:
            sg = torch.where(sat, x, torch.where(ug, 1.0 - sw, 0.0 * x))
        else:
            sg = torch.where(sat, x, 0.0 * x)
        rs_s = pvt.rs_sat(p)
        on_sat_line = sat if cap is None else sat & (rs_s.detach() <= cap)
        rs_sat = rs_s if cap is None else torch.minimum(rs_s, cap)
        rs = torch.where(sat, rs_sat, torch.where(ug, rs_sat, x) if wet else x)
        so = 1.0 - sw - sg
        bo, mu_o = pvt.oil(p, rs, on_sat_line | ug if wet else on_sat_line)
        bw, mu_w = pvt.water(p)
        if wet:
            rv = torch.where(ug, x, self.rv_sat_eff(p, so))
            # при VAPPARS насыщенный газ недонасыщен относительно PVTG — по Rv (ветви непрерывны)
            bg, mu_g = pvt.gas(p, rv, ~ug if self.vap1 is None else None)
        else:
            rv = None
            bg, mu_g = pvt.gas(p)
        pv = self._pv(p)
        if wet:
            acc_o = pv * so * bo * rho_o + pv * sg * bg * rv * rho_o
        else:
            acc_o = pv * so * bo * rho_o
        acc = (acc_o, pv * sw * bw * rho_w, pv * (sg * bg + so * bo * rs) * rho_g)
        return dict(sg=sg, rs=rs, so=so, rv=rv, bo=bo, mu_o=mu_o, bw=bw, mu_w=mu_w, bg=bg, mu_g=mu_g, acc=acc)

    def masses(self, p, sw, x, sat, ug=None):
        """Массы компонентов ячеек [3, N] (o, w, g) — то же, что строки I_ACC свойств, без ОФП и Pc."""
        return torch.stack(self._phase_state(p, sw, x, sat, ug)["acc"])

    def _cell_fn(self, sat, ug=None):
        kr = self.kr
        rho_o, rho_w, rho_g = self.pvt.rho_sc
        wet = self.wet and ug is not None

        def fn(p, sw, x):
            st = self._phase_state(p, sw, x, sat, ug)
            sg, rs, so, rv = st["sg"], st["rs"], st["so"], st["rv"]
            bo, mu_o, bw, mu_w, bg, mu_g = st["bo"], st["mu_o"], st["bw"], st["mu_w"], st["bg"], st["mu_g"]
            krw, kro, krg = kr(sw, sg)
            pcow, pcog = kr.pc(sw, sg)
            mo, mw, mg = kro * bo / mu_o, krw * bw / mu_w, krg * bg / mu_g
            if wet:
                rho_gr = (rho_g + rv * rho_o) * bg
                m_og = rho_o * rv * mg
            else:
                rho_gr, m_og = rho_g * bg, 0.0 * p
            return torch.stack([
                *st["acc"],
                p, p - pcow, p + pcog,
                (rho_o + rs * rho_g) * bo, rho_w * bw, rho_gr,
                rho_o * mo, rho_w * mw, rho_g * mg, rho_g * rs * mo,
                kro / mu_o + krw / mu_w + krg / mu_g,
                bo, bw, bg, m_og])
        return fn

    def cell_values(self, p, sw, x, sat, ug=None):
        return self._cell_fn(sat, ug)(p, sw, x)

    def cell_state(self, p, sw, x, sat, ug=None):
        """Свойства [N_PROPS, N] и производные [N_PROPS, N, 3] по (P, Sw, X)."""
        fn = self._cell_fn(sat, ug)
        one, zero = torch.ones_like(p), torch.zeros_like(p)
        ders = []
        val = None
        for k in range(3):
            tang = [zero, zero, zero]
            tang[k] = one
            val, dv = jvp(fn, (p, sw, x), tuple(tang))
            ders.append(dv)
        return val, torch.stack(ders, dim=-1)

    # ------------------------------------------------------------ сборка
    def assemble_bo(self, p, sw, x, sat, acc_old, dt, p_bh, state=None, ug=None):
        n, dev = self.n_cells, self.device
        v, d = state if state is not None else self.cell_state(p, sw, x, sat, ug)
        R = torch.zeros((n, 3), dtype=torch.float64, device=dev)
        J = torch.zeros((self.builder.n_nonzero, 3, 3), dtype=torch.float64, device=dev)
        dm = self.builder.diag_map
        R += (v[I_ACC:I_ACC + 3] - acc_old).T / dt
        J[dm] += d[I_ACC:I_ACC + 3].permute(1, 0, 2) / dt

        # вырожденная строка газа в недонасыщенной ячейке без нефти (So → 0): малая регуляризация
        so = 1.0 - sw - torch.where(sat, x, 0.0 * x)
        if ug is not None and self.wet:
            so = torch.where(ug, torch.zeros_like(so), so)
            # строка нефти газовой ячейки при Sg → 0 (Sw → 1) вырождается по Rv
            weak_o = ug & (1.0 - sw < 1e-6)
            if bool(weak_o.any()):
                J[dm[weak_o], OIL, 2] += 1e-8 * self.pore_volume[weak_o] * self.pvt.rho_sc[OIL] / dt
            weak = (~sat) & (~ug) & (so < 1e-6)
        else:
            weak = (~sat) & (so < 1e-6)
        if bool(weak.any()):
            J[dm[weak], GAS, 2] += 1e-8 * self.pore_volume[weak] * self.pvt.rho_sc[GAS] / dt

        L, Rr = self.face_L, self.face_R
        if L.numel():
            T, gdz = self.face_T, G * self.face_dZ
            pot = []
            th = self._face_thresholds(v)
            for ph in range(3):
                ip, ir = I_P + ph, I_RHO + ph
                rho_avg = 0.5 * (v[ir][L] + v[ir][Rr])
                dpot = (v[ip][L] - v[ip][Rr]) - rho_avg * gdz
                dpL = d[ip][L] - 0.5 * gdz.unsqueeze(1) * d[ir][L]
                dpR = -d[ip][Rr] - 0.5 * gdz.unsqueeze(1) * d[ir][Rr]
                up = dpot >= 0
                if th is not None:
                    # THPRES: поток через границу регионов — только сверх порога, Δφ_eff = sign·max(|Δφ| − θ, 0)
                    theta = torch.where(up, th[0], th[1])
                    over = (dpot.abs() > theta) | (theta == 0)
                    dpot = torch.where(over, dpot - torch.sign(dpot) * theta, torch.zeros_like(dpot))
                    gate = over.double().unsqueeze(1)
                    dpL, dpR = dpL * gate, dpR * gate
                pot.append((dpot, dpL, dpR, up))
            for comp, ph, mi in self.flux_terms:
                dpot, dpL, dpR, up = pot[ph]
                upd = up.double().unsqueeze(1)
                M = torch.where(up, v[mi][L], v[mi][Rr])
                flux = T * M * dpot
                R[:, comp].index_add_(0, L, flux)
                R[:, comp].index_add_(0, Rr, -flux)
                Tc = T.unsqueeze(1)
                dFL = Tc * (d[mi][L] * upd * dpot.unsqueeze(1) + M.unsqueeze(1) * dpL)
                dFR = Tc * (d[mi][Rr] * (1.0 - upd) * dpot.unsqueeze(1) + M.unsqueeze(1) * dpR)
                Jc = J[:, comp, :]
                Jc.index_add_(0, self.diag_L, dFL)
                Jc.index_add_(0, self.face_pos_LR, dFR)
                Jc.index_add_(0, self.diag_R, -dFR)
                Jc.index_add_(0, self.face_pos_RL, -dFL)

        wsys = self.wells.assemble(R, J, dm, v, d, p_bh) if self.wells.n_w else None
        return R.view(-1), J, wsys

    def _face_thresholds(self, v):
        """
        Пороговые давления THPRES по граням: (θ для потока L→R, θ для R→L) или None.
        Обратимые (по умолчанию) — одно значение для пары регионов в обе стороны; IRREVERS — по направлению.
        Порог по умолчанию — максимальная по фазам |Δφ| через границу пары в начальном состоянии (первая сборка).
        """
        spec = getattr(self.reservoir, "thpres", None)
        if spec is None:
            return None
        cached = getattr(self, "_th_cache", None)
        if cached is not None:
            return cached
        eql = self.reservoir.cell_eqlnum.to(self.device).long().view(-1)
        L, Rr = self.face_L, self.face_R
        rL, rR = eql[L], eql[Rr]
        cross = rL != rR
        n_r = int(eql.max()) + 1
        gdz = G * self.face_dZ
        dmax = torch.zeros_like(self.face_T)
        for ph, im in ((OIL, I_MOO), (WAT, I_MWW), (GAS, I_MGG)):
            ip, ir = I_P + ph, I_RHO + ph
            dpot = (v[ip][L] - v[ip][Rr]) - 0.5 * (v[ir][L] + v[ir][Rr]) * gdz
            mob = torch.where(dpot >= 0, v[im][L], v[im][Rr])
            dmax = torch.maximum(dmax, torch.where(mob > 0, dpot.abs(), torch.zeros_like(dpot)))
        table = {}
        for r1, r2, val in spec["records"]:
            if not (0 <= r1 < n_r and 0 <= r2 < n_r):
                raise ValueError(f"THPRES: регионы {r1 + 1}, {r2 + 1} вне EQLNUM 1..{n_r}")
            if val is None:
                m = cross & (((rL == r1) & (rR == r2)) | ((rL == r2) & (rR == r1)))
                val = float(dmax[m].max()) if bool(m.any()) else 0.0
            table[(r1, r2)] = val
            if not spec["irreversible"]:
                table[(r2, r1)] = val
        lut = torch.zeros(n_r, n_r, dtype=torch.float64, device=self.device)
        for (a, b), val in table.items():
            lut[a, b] = val
        th_lr = torch.where(cross, lut[rL, rR], torch.zeros_like(self.face_T))
        th_rl = torch.where(cross, lut[rR, rL], torch.zeros_like(self.face_T))
        self._th_cache = (th_lr, th_rl)
        return self._th_cache

    # ------------------------------------------------------------ критерии
    def convergence(self, R, dt, pv, wsys, p_bh, v=None):
        Rb = R.view(-1, 3)
        b_avg = (1.0 / v[I_B:I_B + 3]).mean(dim=1)
        scale = dt * b_avg / self.rho_sc
        cnv_cell = scale * Rb.abs() / pv.unsqueeze(1)
        cnv = cnv_cell.max(dim=0).values
        mb = scale * Rb.sum(dim=0).abs() / pv.sum()
        bad = cnv_cell.max(dim=1).values > self.tol_cnv
        # ослабление — после max_strict_iter итераций или раньше, если Ньютон застрял (невязка не убывает)
        prev = getattr(self, "_prev_norm", None)
        norm = max(max(c / self.tol_cnv, m / self.tol_mb) for c, m in zip(cnv.tolist(), mb.tolist()))
        stagnant = prev is not None and norm > 0.99 * prev
        relaxed = ((getattr(self, "_newton_iter", 0) >= self.max_strict_iter or stagnant) and bool(bad.any())
                   and float(pv[bad].sum() / pv.sum()) <= self.relaxed_max_pv_fraction)
        tol_cnv = self.tol_cnv_relaxed if relaxed else self.tol_cnv
        tol_mb = self.tol_mb_relaxed if relaxed else self.tol_mb
        well_ok, well_err = True, 0.0
        if wsys is not None:
            rate = wsys["is_rate"]
            rel = wsys["r_w"].abs() / torch.clamp(wsys["target"].abs(), min=1e-12)
            rate_err = torch.where(rate, rel, torch.zeros_like(rel))
            bhp_err = torch.where(rate, torch.zeros_like(rel), wsys["r_w"].abs())
            well_err = float(rate_err.max()) if bool(rate.any()) else 0.0
            well_ok = bool((rate_err <= self.tol_well_rate).all() and (bhp_err <= self.tol_well_bhp).all())
        cnv_l, mb_l = cnv.tolist(), mb.tolist()
        ok = max(cnv_l) <= tol_cnv and max(mb_l) <= tol_mb and well_ok
        return ok, {"cnv": cnv_l, "mb": mb_l, "well_rate_rel_err": well_err, "cnv_relaxed": relaxed}

    # ------------------------------------------------------------ шаг
    def _sg_rs(self, p, x, sat, ug=None, sw=None):
        rs_s = self.rs_sat_eff(p)
        if ug is None or not self.wet:
            return torch.where(sat, x, torch.zeros_like(x)), torch.where(sat, rs_s, x)
        sg = torch.where(sat, x, torch.where(ug, 1.0 - sw, torch.zeros_like(x)))
        return sg, torch.where(sat | ug, rs_s, x)

    def _rv(self, p, x, ug, so):
        return torch.where(ug, x, self.rv_sat_eff(p, so))

    def step(self, dt, current_time=None):
        dbg = self.debug
        log = (lambda *a, **k: None) if (dbg.enabled and dbg.console) else print
        t_days = (current_time or 0.0) / 86400.0
        shp = (self.nx, self.ny, self.nz)

        p_old = self.fluid.pressure.detach().reshape(-1).double().clone()
        sw_old = self.fluid.s_w.detach().reshape(-1).double().clone()
        sat_old, x_old, ug_old = self.sat.clone(), self.x.clone(), self.ug.clone()
        if self.drsdt is not None and self.pvt.has_pvt:
            rs_old = self.fluid.rs.reshape(-1).double()
            self._rs_cap = rs_old + self.drsdt * dt
        sg_old, _ = self._sg_rs(p_old, x_old, sat_old, ug_old, sw_old)
        acc_old = self.masses(p_old, sw_old, x_old, sat_old, ug_old)
        mass_total = acc_old.sum(dim=1).tolist() if dbg.enabled else None

        p, sw, x, sat, ug = p_old.clone(), sw_old.clone(), x_old.clone(), sat_old.clone(), ug_old.clone()
        state = self.cell_state(p, sw, x, sat, ug)
        if self.wells.n_w:
            hc = self.pore_volume * (1.0 - sw.view(-1))
            hc_sum = hc.sum().clamp_min(1e-30)
            self.wells.p_avg = float((hc * p.view(-1)).sum() / hc_sum)
            if self.pvt.has_pvt:
                self.wells.rs_avg = float((hc * self.fluid.rs.reshape(-1).double()).sum() / hc_sum)
                rv = getattr(self.fluid, "rv", None)
                self.wells.rv_avg = float((hc * rv.reshape(-1).double()).sum() / hc_sum) if self.wet and rv is not None \
                    else None
            self.wells.begin_step(state[0], state[1], t_days)
            p_bh = self.wells.p_bh.clone()
        else:
            p_bh = torch.zeros(0, dtype=torch.float64, device=self.device)
        self._p_bh_cur = p_bh
        dbg.begin_step(dt, current_time, p_old, sw_old)
        initial_cnv = None
        n_switch = 0
        cnv_hist, omega = [], 1.0
        prev_norm, lin_scale = None, 1.0
        n_flips = torch.zeros(p.numel(), dtype=torch.int32, device=p.device)

        def fail(reason, R=None, J=None):
            st = self._last_linear_stats if reason == "linear_solver_failed" else {}
            log(f"    ✖ {reason}" + (f": невязка {st.get('true_rel_res', float('nan')):.2e} при допуске "
                                     f"{st.get('tol', float('nan')):.1e}" if st else ""))
            dbg.end_step(False, reason, p.view(shp), sw.view(shp), R, J, self.builder)
            self._p_bh_cur = None
            return False

        for k in range(self.max_newton):
            dbg.begin_iter(k)
            timer = dbg.timer()
            with timer('assemble'):
                R, J, wsys = self.assemble_bo(p, sw, x, sat, acc_old, dt, p_bh, state, ug)
                if wsys is not None and k > 0:
                    msgs = self.wells.update_controls(p_bh, wsys["q_surf"])
                    for m in msgs:
                        log(f"    ⇄ {m}")
                    if msgs:
                        if dbg.enabled:
                            dbg.iter_rec["control_switch"] = msgs
                        p_bh = self._reset_bhp_targets(p_bh)
                        self._p_bh_cur = p_bh
                        R, J, wsys = self.assemble_bo(p, sw, x, sat, acc_old, dt, p_bh, state, ug)
            pv = self._pv(p)
            self._newton_iter, self._prev_norm = k, prev_norm
            ok, crit = self.convergence(R, dt, pv, wsys, p_bh, state[0])
            dbg.residual(R, dt, mass_total)
            if dbg.enabled:
                dbg.iter_rec["criteria"] = crit
                dbg.iter_rec["n_saturated"] = int(sat.sum())
                if wsys is not None and dbg.level >= 2:
                    dbg.wells(self.wells.records(wsys, p_bh, p))
            log(f"  Iter {k}: CNV o/w/g = " + "/".join(f"{c:.2e}" for c in crit['cnv'])
                + f", MB = {max(crit['mb']):.2e}, wells = {crit['well_rate_rel_err']:.1e}"
                + (f", переключений фаз {n_switch}" if n_switch else ""))

            if not torch.isfinite(R).all():
                dbg.converged_iter()
                return fail("nan_residual", R, J)
            if initial_cnv is None:
                initial_cnv = max(crit["cnv"]) + 1e-300

            if ok and k >= self.min_newton:
                log("  Converged!")
                sg, rs = self._sg_rs(p, x, sat, ug, sw)
                so, so_old = 1.0 - sw - sg, 1.0 - sw_old - sg_old
                self.last_newton_iters = k
                self.last_max_dsw = float(torch.stack([(sw - sw_old).abs().max(), (sg - sg_old).abs().max(),
                                                       (so - so_old).abs().max()]).max())
                self.last_max_dp = float((p - p_old).abs().max())
                dbg.converged_iter()
                dbg.end_step(True, "converged", p.view(shp), sw.view(shp))
                f = self.fluid
                dtp = f.pressure.dtype
                f.pressure.copy_(p.view(shp).to(dtp))
                f.s_w.copy_(sw.view(shp).to(f.s_w.dtype))
                f.s_g = sg.view(shp).to(dtp).clone()
                f.s_o = so.view(shp).to(dtp).clone()
                f.rs = rs.view(shp).clone()
                f.rv = self._rv(p, x, ug, so).view(shp).clone()
                self.so_max = torch.maximum(self.so_max, so.clamp_min(0.0))
                self.sat, self.x, self.ug = sat, x, ug
                hyst = getattr(self.kr, "hyst", None)
                if hyst is not None:
                    hyst.commit(sw, sg)
                if self.wells.n_w:
                    self.wells.commit(p_bh)
                    self._store_well_results(wsys, p_bh, dt)
                self._p_bh_cur = None
                return True

            dbg.jacobian(J, self.builder.diag_map)
            # допуск линейного решателя — по норме всей системы, а CNV — максимум по ячейкам: локальная невязка
            # (например, в водоносной ячейке) может не уменьшаться при «решённой» системе; при застое Ньютона
            # (нормированная невязка не упала вдвое) допуск ужесточается в 100 раз
            norm = max(max(c / self.tol_cnv, m / self.tol_mb) for c, m in zip(crit["cnv"], crit["mb"]))
            if prev_norm is not None and norm > 0.5 * prev_norm:
                lin_scale *= 0.01
            prev_norm = norm
            rel = max(crit["cnv"]) / initial_cnv * lin_scale
            with timer('linear'):
                sol = self.solve_linear(J, R, wsys, newton_rel=rel,
                                        rebuild_amg=(k == 0 or self.amg_rebuild == 'iteration'))
            dbg.linear(self._last_linear_stats)
            if sol is None:
                dbg.end_iter(timer.t if dbg.enabled else {})
                return fail("linear_solver_failed", R, J)
            dx, dpw = sol
            cnv_hist.append([max(c / self.tol_cnv, m / self.tol_mb) for c, m in zip(crit["cnv"], crit["mb"])])
            if self._oscillating(cnv_hist):
                omega = max(omega - self.relax_increment, self.relax_min)
                log(f"    ~ колебания Ньютона: релаксация ω = {omega:.2f}")
            if omega < 1.0:
                dx, dpw = dx * omega, dpw * omega
            sat_prev, ug_prev = sat, ug
            p, sw, x, sat, ug, n_sw = self._apply_update(p, sw, x, sat, ug, dx, dbg,
                                                         frozen=n_flips >= MAX_CELL_SWITCHES)
            n_flips += (sat != sat_prev) | (ug != ug_prev)
            n_switch += n_sw
            if dpw.numel():
                dpw_lim = torch.clamp(self.dbhp_max_rel * p_bh.abs(), min=self.dp_max_abs)
                p_bh = (p_bh + torch.maximum(torch.minimum(dpw, dpw_lim), -dpw_lim)).clamp_min(1e3)
            self._p_bh_cur = p_bh
            state = self.cell_state(p, sw, x, sat, ug)
            if not torch.isfinite(state[0]).all():
                dbg.end_iter(timer.t if dbg.enabled else {})
                return fail("nan_state")
            dbg.line_search([], 1.0, True)
            dbg.end_iter(timer.t if dbg.enabled else {})

        R, J, _ = self.assemble_bo(p, sw, x, sat, acc_old, dt, p_bh, state, ug)
        return fail("max_newton_iter", R, J)

    def _oscillating(self, hist):
        """
        Колебания Ньютона (как detectOscillations в OPM): по ≥2 уравнениям невязка вернулась к значению
        двух итераций назад (|F0 − F2| < tol·F0), но отличается от предыдущей (|F0 − F1| > tol·F0).
        hist — невязки уравнений, нормированные на допуск: max(CNV/tol_cnv, MB/tol_mb).
        """
        if len(hist) < 3:
            return False
        tol = self.relax_rel_tol
        n = 0
        for f0, f1, f2 in zip(hist[-1], hist[-2], hist[-3]):
            if f0 <= 1.0:
                continue
            if abs(f0 - f2) < tol * f0 and abs(f0 - f1) > tol * f0:
                n += 1
        return n >= min(2, len(hist[-1]))

    @staticmethod
    def _clamp_x(x, sw, sat, es):
        """Sg насыщенной ячейки — в полосе [−es, 1 − Sw + es] без обрезки (обрезка теряет массу), Rs/Rv ≥ 0."""
        return torch.where(sat, torch.minimum(torch.maximum(x, -es + 0.0 * x), 1.0 - sw + es), x.clamp_min(0.0))

    def _conservative_switch(self, p, sw, x, sat_old, ug_old, sat, ug, changed):
        """
        Первичная переменная ячейки в новом состоянии (sat, ug) так, чтобы масса компонента, по которому
        произошло переключение, совпала с массой в исходном состоянии (sat_old, ug_old) — по той же функции
        свойств ячейки: при появлении/исчезновении газа сохраняется газ, нефти — нефть.
        Sg в насыщенной ячейке входит в массы линейно (точно за один шаг), Rs/Rv — секущими.
        """
        target_m = self.masses(p, sw, x, sat_old, ug_old)
        a = 1.0 - sw
        to_sat = changed & sat
        if ug is None:
            comp_gas = torch.ones_like(sat)
        else:
            comp_gas = (sat_old & ~ug) | (~sat_old & ~ug_old)
        k = torch.where(comp_gas, GAS, OIL)

        def mass(xx):
            m = self.masses(p, sw, xx, sat, ug)
            return torch.gather(m, 0, k.view(1, -1)).view(-1)

        tgt = torch.gather(target_m, 0, k.view(1, -1)).view(-1)
        x_new = x.clone()
        if bool(to_sat.any()):
            z0, z1 = torch.zeros_like(x), a
            f0, f1 = mass(torch.where(to_sat, z0, x)), mass(torch.where(to_sat, z1, x))
            den = torch.where((f1 - f0).abs() > 1e-300, f1 - f0, torch.ones_like(f0))
            x_new = torch.where(to_sat, (a * (tgt - f0) / den).clamp(min=0.0).minimum(a), x_new)
        to_us = changed & ~sat
        if bool(to_us.any()):
            rs_s = self.rs_sat_eff(p)
            rv_s = self.rv_sat_eff(p, torch.zeros_like(p)) if self.wet else torch.zeros_like(p)
            hi = rs_s if ug is None else torch.where(ug, rv_s, rs_s)
            xa, xb = torch.zeros_like(x), hi
            fa = mass(torch.where(to_us, xa, x_new)) - tgt
            fb = mass(torch.where(to_us, xb, x_new)) - tgt
            for _ in range(6):
                den = torch.where((fb - fa).abs() > 1e-300, fb - fa, torch.ones_like(fa))
                xc = (xb - fb * (xb - xa) / den).clamp(min=0.0).minimum(hi)
                fc = mass(torch.where(to_us, xc, x_new)) - tgt
                xa, fa, xb, fb = xb, fb, xc, fc
            x_new = torch.where(to_us, xb, x_new)
        return x_new

    def _apply_update(self, p, sw, x, sat, ug, dx, dbg, frozen=None):
        """Ограничение поправок по ячейке (Appleyard) и переключение переменных Rs ↔ Sg ↔ Rv."""
        dp, dsw, dxx = dx[:, 0], dx[:, 1], dx[:, 2]
        dp_lim = torch.clamp(self.dp_max_rel * p.abs(), min=self.dp_max_abs)
        dp = torch.maximum(torch.minimum(dp, dp_lim), -dp_lim)
        dsg = torch.where(sat, dxx, torch.zeros_like(dxx))
        ds_max = torch.stack([dsw.abs(), dsg.abs(), (dsw + dsg).abs()]).max(dim=0).values
        fs = torch.clamp(self.ds_max / ds_max.clamp_min(1e-300), max=1.0)
        dsw, dsg = dsw * fs, dsg * fs
        drs_lim = self.drs_max_rel * max(self.pvt.rs_max, 1e-12)
        drs = torch.where(sat, torch.zeros_like(dxx), dxx.clamp(-drs_lim, drs_lim))
        if self.wet:
            drv_lim = self.drs_max_rel * max(self.pvt.rv_max, 1e-12)
            drs = torch.where(ug, dxx.clamp(-drv_lim, drv_lim), drs)
        if dbg.enabled:
            dbg.update(dp, torch.maximum(dsw.abs(), dsg.abs()), 1.0)
            dbg.iter_rec["update"].update(n_chopped_p=int((dx[:, 0].abs() > dp_lim).sum()),
                                          n_chopped_s=int((fs < 1).sum()))

        p = (p + dp).clamp_min(1e3)
        sw = (sw + dsw).clamp(0.0, 1.0)
        x = x + torch.where(sat, dsg, drs)
        n_switch = 0
        free = torch.ones_like(sat) if frozen is None else ~frozen
        if self.wet:
            rs_s, rv_s = self.rs_sat_eff(p), self.rv_sat_eff(p, torch.zeros_like(p))   # rv_s — для ячеек без нефти
            uo = ~sat & ~ug
            es, er = SWITCH_EPS_S, SWITCH_EPS_R
            no_oil = 1.0 - sw <= SO_EMPTY
            sat_to_uo = free & sat & (x < -es) & (1.0 - sw > SO_FILLED)
            sat_to_ug = free & sat & (x > 1.0 - sw + es)
            uo_to_sat = free & uo & ((x > rs_s * (1.0 + er) + 1e-300) | no_oil)
            ug_to_sat = free & ug & (x > rv_s * (1.0 + er) + 1e-300)
            ug_n = (ug & ~ug_to_sat) | sat_to_ug
            sat_n = (sat & ~sat_to_uo & ~sat_to_ug) | uo_to_sat | ug_to_sat
            changed = sat_to_uo | sat_to_ug | uo_to_sat | ug_to_sat
            if bool(changed.any()):
                x = self._conservative_switch(p, sw, x, sat, ug, sat_n, ug_n, changed)
            sat, ug = sat_n, ug_n
            n_switch = int(sat_to_uo.sum() + sat_to_ug.sum() + uo_to_sat.sum() + ug_to_sat.sum())
            x = self._clamp_x(x, sw, sat, es)
            return p, sw, x, sat, ug, n_switch
        if self.pvt.has_pvt:
            rs_s = self.rs_sat_eff(p)
            es, er = SWITCH_EPS_S, SWITCH_EPS_R
            no_oil = 1.0 - sw <= SO_EMPTY
            to_us = free & sat & (x < -es) & (1.0 - sw > SO_FILLED)
            to_sat = free & (~sat) & ((x > rs_s * (1.0 + er) + 1e-300) | no_oil)
            sat_n = (sat & ~to_us) | to_sat
            if bool((to_us | to_sat).any()):
                x = self._conservative_switch(p, sw, x, sat, None, sat_n, None, to_us | to_sat)
            sat = sat_n
            n_switch = int(to_us.sum() + to_sat.sum())
            x = self._clamp_x(x, sw, sat, es)
            return p, sw, x, sat, ug, n_switch
        x = torch.where(sat, torch.minimum(x.clamp_min(0.0), 1.0 - sw), x.clamp_min(0.0))
        return p, sw, x, sat, ug, n_switch

    def _store_well_results(self, wsys, p_bh, dt):
        rho = self.pvt.rho_sc
        for w_idx, w in enumerate(self.wells.wells):
            m = self.wells.conn_well == w_idx
            q = wsys["q"][m].sum(dim=0)
            q_o, q_w, q_g = (float(q[i]) / rho[i] * DAY for i in range(3))
            w.bhp_mpa = float(p_bh[w_idx]) / 1e6
            w.last_surface_rates = {"oil": q_o, "water": q_w, "gas": q_g, "liquid": q_o + q_w}
            # добыча без перетоков (перфорации с обратным потоком в пласт не вычитаются, как в Eclipse)
            qp = wsys["q"][m].clamp_min(0.0).sum(dim=0)
            w.last_surface_prod = {ph: float(qp[i]) / rho[i] * DAY for i, ph in enumerate(("oil", "water", "gas"))}
            # накопленные поверхностные объёмы, м³ (знак как у дебитов: + отбор); отдельно — за периоды
            # добычи (cum_prod) и закачки (cum_inj, + закачка) у скважин, меняющих тип
            zero = {"oil": 0.0, "water": 0.0, "gas": 0.0}
            cum = getattr(w, "cum_surface", None) or dict(zero)
            cp = getattr(w, "cum_prod", None) or dict(zero)
            ci = getattr(w, "cum_inj", None) or dict(zero)
            for ph, val in (("oil", q_o), ("water", q_w), ("gas", q_g)):
                cum[ph] += val * dt / DAY
                if w.type == 'producer':
                    cp[ph] += val * dt / DAY
                else:
                    ci[ph] -= val * dt / DAY
            w.cum_surface, w.cum_prod, w.cum_inj = cum, cp, ci


def needs_blackoil(fluid, well_manager):
    """Нужна ли трёхфазная модель: PVT-таблица, свободный газ или газовые скважины."""
    if getattr(fluid, 'pvt', None) is not None or getattr(fluid, 'blackoil_pvt', None) is not None:
        return True
    if float(fluid.s_g.max()) > 0:
        return True
    for w in well_manager.wells:
        if w.type == 'injector' and (w.injected_phase or '').lower() == 'gas':
            return True
        if w.type == 'producer' and (w.surface_phase or '').lower() == 'gas':
            return True
    return False
