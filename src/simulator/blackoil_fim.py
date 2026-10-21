"""
Полностью неявный трёхфазный black-oil (FIM): нефть, вода, газ с растворённым газом Rs.

Неизвестные в ячейке: P (давление нефти), Sw и переменная X с переключением (как в OPM Flow):
  насыщенная ячейка (есть свободный газ):     X = Sg,  Rs = Rs_sat(P)
  недонасыщенная нефть (свободного газа нет): X = Rs,  Sg = 0
Уравнения — массовые балансы компонентов [кг/с]:
  нефть: d/dt(φ So bo ρo_sc) + div(ρo_sc bo kro/μo ∇Φo)
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
N_PROPS = 17
# (компонент, фаза, строка подвижности)
FLUX_TERMS = ((OIL, OIL, I_MOO), (WAT, WAT, I_MWW), (GAS, GAS, I_MGG), (GAS, OIL, I_MGO))

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

    def _conn_flows(self, v, d, p_bh):
        """Массовые потоки компонентов по перфорациям q [n_conn,3], dq/dx [n_conn,3,3], dq/dp_bh [n_conn,3]."""
        c, cw, wi = self.conn_cell, self.conn_well, self.conn_wi
        n_c = c.numel()
        p_wc = p_bh[cw] + self.rho_wb[cw] * G * self.conn_dz
        q = torch.zeros((n_c, 3), dtype=torch.float64, device=self.device)
        dq = torch.zeros((n_c, 3, 3), dtype=torch.float64, device=self.device)
        dqb = torch.zeros((n_c, 3), dtype=torch.float64, device=self.device)
        for comp, ph, mi in FLUX_TERMS:
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
        lam = torch.stack([v[I_MOO], v[I_MWW], v[I_MGG]])[:, c] / (self.rho_sc.view(3, 1) * v[I_B:I_B + 3][:, c])
        rho = v[I_RHO:I_RHO + 3][:, c]
        mix = (lam * rho).sum(0) / lam.sum(0).clamp_min(1e-30)
        rho_inj = rho.gather(0, self.inj_phase[cw].unsqueeze(0)).squeeze(0)
        mix = torch.where(self.is_inj[cw], rho_inj, mix)
        zeros = torch.zeros(self.n_w, dtype=torch.float64, device=self.device)
        cnt = zeros.clone().index_add_(0, cw, torch.ones_like(mix))
        self.rho_wb = zeros.clone().index_add_(0, cw, mix) / cnt.clamp_min(1)

        # дебит линеен по p_bh при замороженной ячейке: q(pb) = q(0) + D·pb
        q, _, dqb = self._conn_flows(v, d, zeros)
        a = self.rate_w[cw]
        q0 = zeros.clone().index_add_(0, cw, (a * q).sum(1))
        D = zeros.clone().index_add_(0, cw, (a * dqb).sum(1))
        p_init = torch.zeros_like(zeros)
        for w_idx, w in enumerate(self.wells):
            ctrl, val = self.step_targets[w_idx]
            mode = 'bhp' if ctrl == 'bhp' else ('bhp_limit' if self.mode[w_idx] == 'bhp_limit' else 'rate')
            if mode != 'rate':
                p_init[w_idx] = val if ctrl == 'bhp' else self.bhp_limit(w)
            else:
                Dw = float(D[w_idx])
                pb = (val - float(q0[w_idx])) / Dw if abs(Dw) > 1e-30 else float(v[I_P][c[cw == w_idx]].mean())
                p_init[w_idx] = self._clamp_bhp(w, torch.tensor(pb, dtype=torch.float64, device=self.device))
            self.step_mode.append(mode)
        if self.p_bh is None:
            self.p_bh = p_init
        else:
            fixed = torch.tensor([m != 'rate' for m in self.step_mode], device=self.device)
            self.p_bh = torch.where(fixed, p_init, self.p_bh)

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

        a = self.rate_w[cw]
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
        self.debug.eq_names = ("oil", "water", "gas")

        print("Building sparse matrix structure (black-oil, блоки 3x3)...")
        self.builder = SparseMatrixBuilder(self.nx, self.ny, self.nz, block_size=3, device=self.device)
        dev = self.device
        self.pore_volume = (reservoir.cell_volume * reservoir.porosity).reshape(-1).double().to(dev)
        self.c_rock = float(getattr(fluid, 'rock_compressibility', 0.0))
        p_ref = getattr(fluid, 'rock_ref_pressure', None)
        self.rock_eclipse = p_ref is not None
        if self.rock_eclipse:
            self.p_ref_rock = torch.full((self.n_cells,), float(p_ref), dtype=torch.float64, device=dev)
        else:
            self.p_ref_rock = fluid.pressure.detach().reshape(-1).double().clone().to(dev)
        self.pvt = getattr(fluid, 'blackoil_pvt', None) or BlackOilPVT(fluid, dev)
        self.kr = getattr(fluid, 'blackoil_relperm', None) or ThreePhaseRelPerm(fluid, dev)
        drsdt = params.get('drsdt')
        self.drsdt = None if drsdt is None else float(drsdt)
        self._rs_cap = None
        self.rho_sc = torch.tensor(self.pvt.rho_sc, dtype=torch.float64, device=dev)

        self._build_connections()
        self.wells = BlackOilWells(well_manager, reservoir, fluid, reservoir.dimensions, dev)
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
        fluid.rs = rs.view(self.nx, self.ny, self.nz)
        print(f"  Black-oil FIM: PVT={'таблица' if self.pvt.has_pvt else 'постоянные сжимаемости'}, "
              f"насыщенных ячеек {int(self.sat.sum())}/{self.n_cells}")

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

    def _cell_fn(self, sat):
        pvt, kr = self.pvt, self.kr
        rho_o, rho_w, rho_g = pvt.rho_sc
        cap = self._rs_cap

        def fn(p, sw, x):
            sg = torch.where(sat, x, 0.0 * x)
            rs_s = pvt.rs_sat(p)
            on_sat_line = sat if cap is None else sat & (rs_s.detach() <= cap)
            rs = torch.where(sat, rs_s if cap is None else torch.minimum(rs_s, cap), x)
            so = 1.0 - sw - sg
            bo, mu_o = pvt.oil(p, rs, on_sat_line)
            bw, mu_w = pvt.water(p)
            bg, mu_g = pvt.gas(p)
            krw, kro, krg = kr(sw, sg)
            pcow, pcog = kr.pc(sw, sg)
            pv = self._pv(p)
            mo, mw, mg = kro * bo / mu_o, krw * bw / mu_w, krg * bg / mu_g
            return torch.stack([
                pv * so * bo * rho_o, pv * sw * bw * rho_w, pv * (sg * bg + so * bo * rs) * rho_g,
                p, p - pcow, p + pcog,
                (rho_o + rs * rho_g) * bo, rho_w * bw, rho_g * bg,
                rho_o * mo, rho_w * mw, rho_g * mg, rho_g * rs * mo,
                kro / mu_o + krw / mu_w + krg / mu_g,
                bo, bw, bg])
        return fn

    def cell_values(self, p, sw, x, sat):
        return self._cell_fn(sat)(p, sw, x)

    def cell_state(self, p, sw, x, sat):
        """Свойства [N_PROPS, N] и производные [N_PROPS, N, 3] по (P, Sw, X)."""
        fn = self._cell_fn(sat)
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
    def assemble_bo(self, p, sw, x, sat, acc_old, dt, p_bh, state=None):
        n, dev = self.n_cells, self.device
        v, d = state if state is not None else self.cell_state(p, sw, x, sat)
        R = torch.zeros((n, 3), dtype=torch.float64, device=dev)
        J = torch.zeros((self.builder.n_nonzero, 3, 3), dtype=torch.float64, device=dev)
        dm = self.builder.diag_map
        R += (v[I_ACC:I_ACC + 3] - acc_old).T / dt
        J[dm] += d[I_ACC:I_ACC + 3].permute(1, 0, 2) / dt

        # вырожденная строка газа в недонасыщенной ячейке без нефти (So → 0): малая регуляризация
        so = 1.0 - sw - torch.where(sat, x, 0.0 * x)
        weak = (~sat) & (so < 1e-6)
        if bool(weak.any()):
            J[dm[weak], GAS, 2] += 1e-8 * self.pore_volume[weak] * self.pvt.rho_sc[GAS] / dt

        L, Rr = self.face_L, self.face_R
        if L.numel():
            T, gdz = self.face_T, G * self.face_dZ
            pot = []
            for ph in range(3):
                ip, ir = I_P + ph, I_RHO + ph
                rho_avg = 0.5 * (v[ir][L] + v[ir][Rr])
                dpot = (v[ip][L] - v[ip][Rr]) - rho_avg * gdz
                dpL = d[ip][L] - 0.5 * gdz.unsqueeze(1) * d[ir][L]
                dpR = -d[ip][Rr] - 0.5 * gdz.unsqueeze(1) * d[ir][Rr]
                up = dpot >= 0
                pot.append((dpot, dpL, dpR, up))
            for comp, ph, mi in FLUX_TERMS:
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

    # ------------------------------------------------------------ критерии
    def convergence(self, R, dt, pv, wsys, p_bh, v=None):
        Rb = R.view(-1, 3)
        b_avg = (1.0 / v[I_B:I_B + 3]).mean(dim=1)
        scale = dt * b_avg / self.rho_sc
        cnv = (scale * Rb.abs() / pv.unsqueeze(1)).max(dim=0).values
        mb = scale * Rb.sum(dim=0).abs() / pv.sum()
        well_ok, well_err = True, 0.0
        if wsys is not None:
            rate = wsys["is_rate"]
            rel = wsys["r_w"].abs() / torch.clamp(wsys["target"].abs(), min=1e-12)
            rate_err = torch.where(rate, rel, torch.zeros_like(rel))
            bhp_err = torch.where(rate, torch.zeros_like(rel), wsys["r_w"].abs())
            well_err = float(rate_err.max()) if bool(rate.any()) else 0.0
            well_ok = bool((rate_err <= self.tol_well_rate).all() and (bhp_err <= self.tol_well_bhp).all())
        cnv_l, mb_l = cnv.tolist(), mb.tolist()
        ok = max(cnv_l) <= self.tol_cnv and max(mb_l) <= self.tol_mb and well_ok
        return ok, {"cnv": cnv_l, "mb": mb_l, "well_rate_rel_err": well_err}

    # ------------------------------------------------------------ шаг
    def _sg_rs(self, p, x, sat):
        rs_s = self.rs_sat_eff(p)
        return torch.where(sat, x, torch.zeros_like(x)), torch.where(sat, rs_s, x)

    def step(self, dt, current_time=None):
        dbg = self.debug
        log = (lambda *a, **k: None) if (dbg.enabled and dbg.console) else print
        t_days = (current_time or 0.0) / 86400.0
        shp = (self.nx, self.ny, self.nz)

        p_old = self.fluid.pressure.detach().reshape(-1).double().clone()
        sw_old = self.fluid.s_w.detach().reshape(-1).double().clone()
        sat_old, x_old = self.sat.clone(), self.x.clone()
        if self.drsdt is not None and self.pvt.has_pvt:
            rs_old = self.fluid.rs.reshape(-1).double()
            self._rs_cap = rs_old + self.drsdt * dt
        sg_old, _ = self._sg_rs(p_old, x_old, sat_old)
        acc_old = self.cell_values(p_old, sw_old, x_old, sat_old)[I_ACC:I_ACC + 3]
        mass_total = acc_old.sum(dim=1).tolist() if dbg.enabled else None

        p, sw, x, sat = p_old.clone(), sw_old.clone(), x_old.clone(), sat_old.clone()
        state = self.cell_state(p, sw, x, sat)
        if self.wells.n_w:
            self.wells.begin_step(state[0], state[1], t_days)
            p_bh = self.wells.p_bh.clone()
        else:
            p_bh = torch.zeros(0, dtype=torch.float64, device=self.device)
        self._p_bh_cur = p_bh
        dbg.begin_step(dt, current_time, p_old, sw_old)
        initial_cnv = None
        n_switch = 0

        def fail(reason, R=None, J=None):
            dbg.end_step(False, reason, p.view(shp), sw.view(shp), R, J, self.builder)
            self._p_bh_cur = None
            return False

        for k in range(self.max_newton):
            dbg.begin_iter(k)
            timer = dbg.timer()
            with timer('assemble'):
                R, J, wsys = self.assemble_bo(p, sw, x, sat, acc_old, dt, p_bh, state)
                if wsys is not None and k > 0:
                    msgs = self.wells.update_controls(p_bh, wsys["q_surf"])
                    for m in msgs:
                        log(f"    ⇄ {m}")
                    if msgs:
                        if dbg.enabled:
                            dbg.iter_rec["control_switch"] = msgs
                        p_bh = self._reset_bhp_targets(p_bh)
                        self._p_bh_cur = p_bh
                        R, J, wsys = self.assemble_bo(p, sw, x, sat, acc_old, dt, p_bh, state)
            pv = self._pv(p)
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
                sg, rs = self._sg_rs(p, x, sat)
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
                self.sat, self.x = sat, x
                if self.wells.n_w:
                    self.wells.commit(p_bh)
                    self._store_well_results(wsys, p_bh, dt)
                self._p_bh_cur = None
                return True

            dbg.jacobian(J, self.builder.diag_map)
            rel = max(crit["cnv"]) / initial_cnv
            with timer('linear'):
                sol = self.solve_linear(J, R, wsys, newton_rel=rel,
                                        rebuild_amg=(k == 0 or self.amg_rebuild == 'iteration'))
            dbg.linear(self._last_linear_stats)
            if sol is None:
                dbg.end_iter(timer.t if dbg.enabled else {})
                return fail("linear_solver_failed", R, J)
            dx, dpw = sol
            p, sw, x, sat, n_sw = self._apply_update(p, sw, x, sat, dx, dbg)
            n_switch += n_sw
            if dpw.numel():
                dpw_lim = torch.clamp(self.dbhp_max_rel * p_bh.abs(), min=self.dp_max_abs)
                p_bh = (p_bh + torch.maximum(torch.minimum(dpw, dpw_lim), -dpw_lim)).clamp_min(1e3)
            self._p_bh_cur = p_bh
            state = self.cell_state(p, sw, x, sat)
            if not torch.isfinite(state[0]).all():
                dbg.end_iter(timer.t if dbg.enabled else {})
                return fail("nan_state")
            dbg.line_search([], 1.0, True)
            dbg.end_iter(timer.t if dbg.enabled else {})

        R, J, _ = self.assemble_bo(p, sw, x, sat, acc_old, dt, p_bh, state)
        return fail("max_newton_iter", R, J)

    def _apply_update(self, p, sw, x, sat, dx, dbg):
        """Ограничение поправок по ячейке (Appleyard) и переключение переменных Sg ↔ Rs."""
        dp, dsw, dxx = dx[:, 0], dx[:, 1], dx[:, 2]
        dp_lim = torch.clamp(self.dp_max_rel * p.abs(), min=self.dp_max_abs)
        dp = torch.maximum(torch.minimum(dp, dp_lim), -dp_lim)
        dsg = torch.where(sat, dxx, torch.zeros_like(dxx))
        ds_max = torch.stack([dsw.abs(), dsg.abs(), (dsw + dsg).abs()]).max(dim=0).values
        fs = torch.clamp(self.ds_max / ds_max.clamp_min(1e-300), max=1.0)
        dsw, dsg = dsw * fs, dsg * fs
        drs_lim = self.drs_max_rel * max(self.pvt.rs_max, 1e-12)
        drs = torch.where(sat, torch.zeros_like(dxx), dxx.clamp(-drs_lim, drs_lim))
        if dbg.enabled:
            dbg.update(dp, torch.maximum(dsw.abs(), dsg.abs()), 1.0)
            dbg.iter_rec["update"].update(n_chopped_p=int((dx[:, 0].abs() > dp_lim).sum()),
                                          n_chopped_s=int((fs < 1).sum()))

        p = (p + dp).clamp_min(1e3)
        sw = (sw + dsw).clamp(0.0, 1.0)
        x = x + torch.where(sat, dsg, drs)
        n_switch = 0
        if self.pvt.has_pvt:
            rs_s = self.rs_sat_eff(p)
            to_us = sat & (x < 0)
            to_sat = (~sat) & (x > rs_s)
            x = torch.where(to_us, rs_s, x)
            x = torch.where(to_sat, torch.zeros_like(x), x)
            sat = (sat & ~to_us) | to_sat
            n_switch = int(to_us.sum() + to_sat.sum())
        x = torch.where(sat, torch.minimum(x.clamp_min(0.0), 1.0 - sw), x.clamp_min(0.0))
        return p, sw, x, sat, n_switch

    def _store_well_results(self, wsys, p_bh, dt):
        rho = self.pvt.rho_sc
        for w_idx, w in enumerate(self.wells.wells):
            m = self.wells.conn_well == w_idx
            q = wsys["q"][m].sum(dim=0)
            q_o, q_w, q_g = (float(q[i]) / rho[i] * DAY for i in range(3))
            w.bhp_mpa = float(p_bh[w_idx]) / 1e6
            w.last_surface_rates = {"oil": q_o, "water": q_w, "gas": q_g, "liquid": q_o + q_w}
            # накопленные поверхностные объёмы, м³ (знак как у дебитов: + отбор)
            cum = getattr(w, "cum_surface", None) or {"oil": 0.0, "water": 0.0, "gas": 0.0}
            for ph, val in (("oil", q_o), ("water", q_w), ("gas", q_g)):
                cum[ph] += val * dt / DAY
            w.cum_surface = cum


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
