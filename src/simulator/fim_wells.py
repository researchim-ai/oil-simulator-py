"""
Модель скважин для полностью неявного решателя (FIM).

Каждая скважина имеет собственную неизвестную — забойное давление на опорной глубине (p_bh).
Система Ньютона на итерации:

    [ J  C ] [dx ]     [ R  ]
    [ B  D ] [dpw] = - [ r_w]

где C = dR/dp_bh, B = dr_w/dx, D = dr_w/dp_bh. Скважинные уравнения исключаются через
дополнение Шура (J - C D^-1 B) — как в OPM Flow. Матрица J в разреженной структуре не меняется:
вклад «своей» перфорации кладётся в диагональный блок (для предобуславливателя), а полная
межперфорационная связь применяется в операторе.

Знак дебитов: q > 0 — отбор из пласта (как в невязке R), q < 0 — закачка.
"""
import math

import torch

G = 9.80665
DAY = 86400.0


def peaceman_wi(kx, ky, dx, dy, h, rw, skin=0.0):
    """Индекс скважины Писмена для вертикального ствола в анизотропной ячейке, м^3."""
    kx = max(float(kx), 1e-30)
    ky = max(float(ky), 1e-30)
    a, b = math.sqrt(ky / kx), math.sqrt(kx / ky)
    r0 = 0.28 * math.sqrt(a * dx * dx + b * dy * dy) / ((ky / kx) ** 0.25 + (kx / ky) ** 0.25)
    denom = math.log(r0 / rw) + skin
    if denom <= 0:
        raise ValueError(f"Peaceman: ln(r0/rw)+skin={denom:.3f} <= 0 (r0={r0:.3f} м, rw={rw} м)")
    return 2.0 * math.pi * math.sqrt(kx * ky) * h / denom


class FIMWells:
    def __init__(self, well_manager, reservoir, fluid, dims, device):
        self.wells = list(well_manager.wells)
        self.n_w = len(self.wells)
        self.device = device
        self.fluid = fluid
        nx, ny, nz = dims
        dx = reservoir.dx_vector.cpu().double()
        dy = reservoir.dy_vector.cpu().double()
        dz = reservoir.dz_vector.cpu().double()
        zc = reservoir.z_centers.cpu().double()
        kx = reservoir.permeability_x.cpu().double()
        ky = reservoir.permeability_y.cpu().double()

        cells, owner, wis, depths = [], [], [], []
        self.ref_depth = []
        for w_idx, w in enumerate(self.wells):
            perfs = w.perforations
            w_depths = []
            for (i, j, k) in perfs:
                if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
                    raise ValueError(f"Скважина {w.name}: перфорация ({i},{j},{k}) вне сетки {dims}")
                wi = w.wi_override if w.wi_override is not None else peaceman_wi(
                    kx[i, j, k], ky[i, j, k], float(dx[i]), float(dy[j]), float(dz[k]), w.radius, w.skin)
                cells.append((i * ny + j) * nz + k)
                owner.append(w_idx)
                wis.append(wi)
                w_depths.append(float(zc[k]))
            ref = min(w_depths) if w.ref_depth is None else float(w.ref_depth)
            self.ref_depth.append(ref)
            depths.extend(d - ref for d in w_depths)
            w.connection_wi = wis[-len(perfs):]

        self.conn_cell = torch.tensor(cells, dtype=torch.long, device=device)
        self.conn_well = torch.tensor(owner, dtype=torch.long, device=device)
        self.conn_wi = torch.tensor(wis, dtype=torch.float64, device=device)
        self.conn_dz = torch.tensor(depths, dtype=torch.float64, device=device)
        self.is_inj = torch.tensor([w.type == 'injector' for w in self.wells], dtype=torch.bool, device=device)

        # состояние (фиксируется только на успешном шаге)
        self.p_bh = None
        self.step_targets = None
        self.mode = [w.control_type for w in self.wells]
        self.rho_wb = torch.zeros(self.n_w, dtype=torch.float64, device=device)

    # ----------------------------------------------------------- управление
    def targets(self, t_days):
        """Возвращает (тип, значение в СИ) для каждой скважины с учётом опционального ramping."""
        out = []
        for w in self.wells:
            ctrl, val = w.get_current_control(t_days)
            out.append((ctrl, val * 1e6 if ctrl == 'bhp' else val / DAY))
        return out

    def rate_phases(self, w):
        """Веса фаз (нефть, вода) в уравнении дебита для поверхностного объёма."""
        if w.type == 'injector':
            return (0.0, -1.0)
        ph = (w.surface_phase or 'liquid').lower()
        if ph == 'oil':
            return (1.0, 0.0)
        if ph == 'water':
            return (0.0, 1.0)
        return (1.0, 1.0)

    def begin_step(self, p, sw, props, t_days):
        """Начало шага: плотность в стволе (явно), начальное p_bh, исходные режимы."""
        if self.n_w == 0:
            return
        self.step_targets = self.targets(t_days)
        self.step_mode = []
        self.switches = 0
        pc = p.view(-1)[self.conn_cell]
        lam_o, lam_w = props['lam_o'][self.conn_cell], props['lam_w'][self.conn_cell]
        rho_o, rho_w = props['rho_o'][self.conn_cell], props['rho_w'][self.conn_cell]
        mob_o, mob_w = lam_o / rho_o, lam_w / rho_w
        mix = (mob_o * rho_o + mob_w * rho_w) / (mob_o + mob_w + 1e-30)
        rho_mix = torch.zeros(self.n_w, dtype=torch.float64, device=self.device).index_add_(0, self.conn_well, mix)
        cnt = torch.zeros(self.n_w, dtype=torch.float64, device=self.device).index_add_(0, self.conn_well, torch.ones_like(mix))
        rho_mix = rho_mix / cnt.clamp_min(1)
        self.rho_wb = torch.where(self.is_inj, torch.full_like(rho_mix, self.fluid.rho_water_ref), rho_mix)

        p_init = torch.zeros(self.n_w, dtype=torch.float64, device=self.device)
        p_ref_cell = pc - self.rho_wb[self.conn_well] * G * self.conn_dz
        lam_t = (mob_o + mob_w) * self.conn_wi
        sum_lt = torch.zeros_like(p_init).index_add_(0, self.conn_well, lam_t)
        sum_lp = torch.zeros_like(p_init).index_add_(0, self.conn_well, lam_t * p_ref_cell)
        p_avg = sum_lp / sum_lt.clamp_min(1e-30)
        for w_idx, w in enumerate(self.wells):
            ctrl, val = self.step_targets[w_idx]
            mode = 'bhp' if ctrl == 'bhp' else ('bhp_limit' if self.mode[w_idx] == 'bhp_limit' else 'rate')
            if ctrl == 'bhp' or mode == 'bhp_limit':
                p_init[w_idx] = val if ctrl == 'bhp' else self.bhp_limit(w)
            else:
                q = val if w.type == 'producer' else -val
                p_init[w_idx] = p_avg[w_idx] - q / max(sum_lt[w_idx].item(), 1e-30)
                p_init[w_idx] = self._clamp_bhp(w, p_init[w_idx])
            self.step_mode.append(mode)
        if self.p_bh is None:
            self.p_bh = p_init
        else:
            # предыдущее p_bh — лучшее начальное приближение; для BHP-режимов — уставка
            fixed = torch.tensor([m != 'rate' for m in self.step_mode], device=self.device)
            self.p_bh = torch.where(fixed, p_init, self.p_bh)

    def commit(self, p_bh):
        self.p_bh = p_bh.clone()
        self.mode = list(self.step_mode)
        for w_idx, w in enumerate(self.wells):
            w.last_mode = self.step_mode[w_idx]

    def _clamp_bhp(self, w, pb):
        lo = (w.bhp_min * 1e6) if (w.type == 'producer' and w.bhp_min is not None) else 1e5
        hi = (w.bhp_max * 1e6) if (w.type == 'injector' and w.bhp_max is not None) else 1e9
        return torch.clamp(pb, lo, hi)

    def bhp_limit(self, w):
        if w.type == 'producer':
            return (w.bhp_min if w.bhp_min is not None else 0.101325) * 1e6
        return (w.bhp_max if w.bhp_max is not None else 1e3) * 1e6

    def update_controls(self, p_bh, q_surf):
        """Переключение rate↔BHP по ограничениям; возвращает список сообщений."""
        msgs = []
        if self.switches >= 4 * max(1, self.n_w):
            return msgs
        for w_idx, w in enumerate(self.wells):
            ctrl, val = self.step_targets[w_idx]
            if ctrl != 'rate':
                continue
            lim = self.bhp_limit(w)
            pb = float(p_bh[w_idx])
            mode = self.step_mode[w_idx]
            if mode == 'rate':
                violated = pb < lim if w.type == 'producer' else pb > lim
                if violated:
                    self.step_mode[w_idx] = 'bhp_limit'
                    self.switches += 1
                    msgs.append(f"{w.name}: rate→BHP ({pb/1e6:.3f} МПа за пределом {lim/1e6:.3f})")
            else:
                q = float(q_surf[w_idx])
                if q > val * (1 + 1e-9):
                    self.step_mode[w_idx] = 'rate'
                    self.switches += 1
                    msgs.append(f"{w.name}: BHP→rate (дебит {q*DAY:.2f} > уставки {val*DAY:.2f} м3/сут)")
        return msgs

    # ------------------------------------------------------------- сборка
    def assemble(self, R, J_vals, diag_map, p, sw, props, p_bh):
        """
        Добавляет перфорационные потоки в R и J (диагональные блоки), возвращает блоки Шура:
        Cv [n_conn,2] = dR/dp_bh, Bv [n_conn,2] = dr_w/d(p,sw), D [n_w], r_w [n_w], q_surf [n_w], а также
        дебиты по перфорациям для журнала.
        """
        if self.n_w == 0:
            return None
        c = self.conn_cell
        wi = self.conn_wi
        pc = p.view(-1)[c]
        p_wc = p_bh[self.conn_well] + self.rho_wb[self.conn_well] * G * self.conn_dz
        dp = pc - p_wc

        lam_o, lam_w = props['lam_o'][c], props['lam_w'][c]
        dlo_dp, dlo_ds = props['dlam_o_dp'][c], props['dlam_o_ds'][c]
        dlw_dp, dlw_ds = props['dlam_w_dp'][c], props['dlam_w_ds'][c]
        rho_w, drho_w = props['rho_w'][c], props['drho_w_dp']
        if self.step_targets is None:
            raise RuntimeError("FIMWells.begin_step() не вызван")
        mu_o, mu_w = self.fluid.mu_oil, self.fluid.mu_water
        kro, krw = props['kro'][c], props['krw'][c]
        dkro, dkrw = props['dkro_ds'][c], props['dkrw_ds'][c]

        inj = self.is_inj[self.conn_well]
        # закачка воды: вода входит с полной подвижностью ячейки
        mt = kro / mu_o + krw / mu_w
        dmt_ds = dkro / mu_o + dkrw / mu_w
        lam_inj = mt * rho_w
        dlam_inj_dp = mt * drho_w
        dlam_inj_ds = dmt_ds * rho_w

        Lo = torch.where(inj, torch.zeros_like(lam_o), lam_o)
        Lw = torch.where(inj, lam_inj, lam_w)
        dLo_dp = torch.where(inj, torch.zeros_like(dlo_dp), dlo_dp)
        dLo_ds = torch.where(inj, torch.zeros_like(dlo_ds), dlo_ds)
        dLw_dp = torch.where(inj, dlam_inj_dp, dlw_dp)
        dLw_ds = torch.where(inj, dlam_inj_ds, dlw_ds)

        q_o = wi * Lo * dp
        q_w = wi * Lw * dp
        R.view(-1, 2)[:, 0].index_add_(0, c, q_o)
        R.view(-1, 2)[:, 1].index_add_(0, c, q_w)

        jd = diag_map[c]
        J_vals[:, 0, 0].index_add_(0, jd, wi * (dLo_dp * dp + Lo))
        J_vals[:, 0, 1].index_add_(0, jd, wi * dLo_ds * dp)
        J_vals[:, 1, 0].index_add_(0, jd, wi * (dLw_dp * dp + Lw))
        J_vals[:, 1, 1].index_add_(0, jd, wi * dLw_ds * dp)

        Cv = torch.stack([-wi * Lo, -wi * Lw], dim=1)

        rho_o_ref, rho_w_ref = self.fluid.rho_oil_ref, self.fluid.rho_water_ref
        phase_w = torch.tensor([self.rate_phases(w) for w in self.wells], dtype=torch.float64, device=self.device)
        a_o = phase_w[self.conn_well, 0] / rho_o_ref
        a_w = phase_w[self.conn_well, 1] / rho_w_ref
        q_surf_conn = a_o * q_o + a_w * q_w
        q_surf = torch.zeros(self.n_w, dtype=torch.float64, device=self.device).index_add_(0, self.conn_well, q_surf_conn)
        dq_dpbh = torch.zeros_like(q_surf).index_add_(0, self.conn_well, a_o * Cv[:, 0] + a_w * Cv[:, 1])
        B_rate = torch.stack([a_o * wi * (dLo_dp * dp + Lo) + a_w * wi * (dLw_dp * dp + Lw),
                              a_o * wi * dLo_ds * dp + a_w * wi * dLw_ds * dp], dim=1)

        r_w = torch.zeros_like(q_surf)
        D = torch.ones_like(q_surf)
        is_rate = torch.zeros(self.n_w, dtype=torch.bool, device=self.device)
        target = torch.zeros_like(q_surf)
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
        D = torch.where(is_rate, dq_dpbh, torch.ones_like(D))
        # вырожденный дебит (нулевая подвижность) — уравнение не определяет p_bh
        degenerate = is_rate & (D.abs() < 1e-30)
        D = torch.where(degenerate, torch.ones_like(D), D)
        r_w = torch.where(degenerate, torch.zeros_like(r_w), r_w)
        conn_rate = is_rate[self.conn_well] & ~degenerate[self.conn_well]
        Bv = torch.where(conn_rate.unsqueeze(1), B_rate, torch.zeros_like(B_rate))

        return {"Cv": Cv, "Bv": Bv, "D": D, "r_w": r_w, "q_surf": q_surf, "is_rate": is_rate,
                "target": target, "q_o": q_o, "q_w": q_w, "diag_pos": jd}

    def add_local_schur(self, sys, J_vals):
        """Вклад перфорации в собственный диагональный блок: -C_c D^-1 B_c (для предобуславливателя)."""
        loc = sys["Cv"].unsqueeze(2) * sys["Bv"].unsqueeze(1) / sys["D"][self.conn_well].view(-1, 1, 1)
        J_vals.index_add_(0, sys["diag_pos"], -loc)

    # ------------------------------------------------------------ оператор
    def schur_correction(self, sys, x_blk, local_included=True):
        """Возвращает -C D^-1 B x  (без локальной части, если она уже в J), форма [N,2]."""
        c = self.conn_cell
        bx_conn = (sys["Bv"] * x_blk[c]).sum(dim=1)
        bx = torch.zeros(self.n_w, dtype=x_blk.dtype, device=x_blk.device).index_add_(0, self.conn_well, bx_conn)
        coef = bx[self.conn_well]
        if local_included:
            coef = coef - bx_conn
        corr = -sys["Cv"] * (coef / sys["D"][self.conn_well]).unsqueeze(1)
        out = torch.zeros_like(x_blk)
        out.index_add_(0, c, corr)
        return out

    def schur_rhs(self, sys, n_cells):
        """C D^-1 r_w в форме [N,2] — добавка к -R."""
        corr = sys["Cv"] * (sys["r_w"] / sys["D"])[self.conn_well].unsqueeze(1)
        out = torch.zeros((n_cells, sys["Cv"].shape[1]), dtype=torch.float64, device=self.device)
        out.index_add_(0, self.conn_cell, corr)
        return out

    def recover_dpw(self, sys, dx_blk):
        bx_conn = (sys["Bv"] * dx_blk[self.conn_cell]).sum(dim=1)
        bx = torch.zeros(self.n_w, dtype=torch.float64, device=self.device).index_add_(0, self.conn_well, bx_conn)
        return (-sys["r_w"] - bx) / sys["D"]

    def records(self, sys, p_bh, p):
        recs = []
        for w_idx, w in enumerate(self.wells):
            m = self.conn_well == w_idx
            recs.append({"name": w.name, "type": w.type, "mode": self.step_mode[w_idx],
                         "p_bh_mpa": float(p_bh[w_idx]) / 1e6,
                         "q_surf_m3_day": float(sys["q_surf"][w_idx]) * DAY,
                         "q_oil_kg_s": float(sys["q_o"][m].sum()), "q_wat_kg_s": float(sys["q_w"][m].sum()),
                         "eq_res": float(sys["r_w"][w_idx]),
                         "p_cells_mpa": [float(v) / 1e6 for v in p.view(-1)[self.conn_cell[m]]]})
        return recs
