"""
Полностью неявный (FIM) решатель нефть–вода с ручной сборкой Якобиана.

Неизвестные: давление P и водонасыщенность Sw в каждой ячейке + забойное давление каждой скважины.
Дискретизация: TPFA по списку граней (готово к NNC / corner-point), upwind по потенциалу фазы,
гравитация, сжимаемость флюидов и породы.

Сходимость Ньютона — по промышленным критериям (как в Eclipse/OPM):
  CNV_α = dt · max_c |R_α,c| / (ρ_α,ref · PV_c)       ≤ tol_cnv   (локальная, по ячейкам)
  MB_α  = dt · |Σ_c R_α,c| / (ρ_α,ref · Σ_c PV_c)    ≤ tol_mb    (глобальный материальный баланс)
  скважины: |Q - Q_target| ≤ tol_well_rate · |Q_target|, |p_bh - p_target| ≤ tol_well_bhp
"""
import contextlib
import io
import time

import torch

from .reservoir import Reservoir
from .fluid import Fluid
from .well import WellManager
from .sparse_utils import SparseMatrixBuilder
from .solver_debug import SolverDebugger
from .fim_wells import FIMWells, G

try:
    from ..solver.fgmres import fgmres
except (ImportError, ValueError):
    from solver.fgmres import fgmres
from .cpr_gpu import CPRPreconditionerGPU


class ManualFIMSolver:
    """
    Fully Implicit Solver с ручной сборкой разреженного блочного Якобиана (BSR 2x2) на GPU.
    """
    def __init__(self, reservoir: Reservoir, fluid: Fluid, well_manager: WellManager, params: dict):
        self.reservoir = reservoir
        self.fluid = fluid
        self.well_manager = well_manager
        self.device = reservoir.device

        self.nx, self.ny, self.nz = reservoir.dimensions
        self.n_cells = self.nx * self.ny * self.nz
        self._init_params(params)

        print("Building sparse matrix structure...")
        self.builder = SparseMatrixBuilder(self.nx, self.ny, self.nz, block_size=2, device=self.device,
                                           pairs=self._conn_pairs())
        print(f"Structure built. NNZ blocks: {self.builder.n_nonzero}")

        # --- Свойства породы ---
        self.pore_volume = (reservoir.cell_volume * reservoir.porosity).reshape(-1).double().to(self.device)
        self.c_rock = float(getattr(fluid, 'rock_compressibility', 0.0))
        self.p_ref_rock = fluid.pressure.detach().reshape(-1).double().clone().to(self.device)

        # --- Относительные проницаемости (Corey) ---
        self.sw_irr = float(getattr(fluid, 'sw_cr', 0.2))
        self.so_r = float(getattr(fluid, 'so_r', 0.2))
        self.nw = float(getattr(fluid, 'nw', 2.0))
        self.no = float(getattr(fluid, 'no', 2.0))

        self._build_connections()
        self.wells = FIMWells(well_manager, reservoir, fluid, reservoir.dimensions, self.device)
        for w in self.wells.wells:
            print(f"  Скважина {w.name}: перфораций {len(w.perforations)}, WI = "
                  + ", ".join(f"{v:.3e}" for v in w.connection_wi) + " м³")
        self._p_bh_cur = None

    def _init_params(self, params):
        # Воспроизводимость: ядра torch с накоплением (scatter/index_add, разреженное умножение в AMG) иначе
        # недетерминированы, и расхождения ~1e-14 усиливаются Ньютоном до различий в результатах (~1e-5)
        if bool(params.get('deterministic', True)):
            torch.use_deterministic_algorithms(True, warn_only=True)
        # --- Нелинейный решатель ---
        self.max_newton = int(params.get('newton_max_iter', 15))
        self.min_newton = int(params.get('newton_min_iter', 0))
        self.tol_cnv = float(params.get('newton_tol_cnv', 1e-3))
        self.tol_mb = float(params.get('newton_tol_mb', 1e-7))
        # как в OPM: если ячейки с CNV > tol_cnv занимают ≤ relaxed_max_pv_fraction порового объёма,
        # CNV и MB проверяются с ослабленными допусками (скважины — строго); только после max_strict_iter итераций
        self.max_strict_iter = int(params.get('newton_max_strict_iter', 8))
        self.tol_cnv_relaxed = float(params.get('newton_tol_cnv_relaxed', 1.0))
        self.tol_mb_relaxed = float(params.get('newton_tol_mb_relaxed', 1e-6))
        self.relaxed_max_pv_fraction = float(params.get('newton_relaxed_max_pv_fraction', 0.03))
        self.tol_well_rate = float(params.get('newton_tol_well_rate', 1e-5))
        self.tol_well_bhp = float(params.get('newton_tol_well_bhp', 10.0))  # Па
        self.ds_max = float(params.get('newton_ds_max', 0.2))
        self.dp_max_rel = float(params.get('newton_dp_max_rel', 0.3))
        self.dp_max_abs = float(params.get('newton_dp_max_abs_mpa', 2.0)) * 1e6
        self.dbhp_max_rel = float(params.get('newton_dbhp_max_rel', 1.0))
        if 'newton_tol_mass' in params or 'newton_tolerance' in params:
            print("  ⚠ newton_tol_mass/newton_tolerance устарели: используются CNV/MB "
                  f"(tol_cnv={self.tol_cnv:g}, tol_mb={self.tol_mb:g})")

        # --- Линейный решатель ---
        self.lin_restart = int(params.get('linear_restart', 50))
        self.lin_max_restarts = int(params.get('linear_max_restarts', 4))
        self.lin_tol_max = float(params.get('linear_tol_max', 1e-2))
        self.lin_tol_min = float(params.get('linear_tol_min', 1e-6))
        self.lin_accept = float(params.get('linear_accept_rel_res', 0.1))
        self.amg_rebuild = params.get('amg_rebuild', 'step')  # 'step' | 'iteration'
        self.cpr_smoother_sweeps = int(params.get('cpr_smoother_sweeps', 1))
        self.linear_verbose = bool(params.get('linear_verbose', False))
        self.amg_verbose = bool(params.get('amg_verbose', False))
        self._amg = None
        self.last_newton_iters = 0
        self.last_max_dsw = 0.0
        self.last_max_dp = 0.0

        self.debug = SolverDebugger(params.get('debug', {}), self.reservoir.dimensions, self.device)
        self._last_linear_stats = {}

    # ------------------------------------------------------------------ геометрия
    def _conn_pairs(self):
        conn = getattr(self.reservoir, 'connections', None)
        return None if conn is None else conn[:2]

    def _build_connections(self):
        """TPFA: T = A / (d_L/(2k_L) + d_R/(2k_R)) по каждой грани; список граней L→R."""
        res = self.reservoir
        nx, ny, nz = self.nx, self.ny, self.nz
        dev = self.device
        dx = res.dx_vector.to(dev).double().view(nx, 1, 1)
        dy = res.dy_vector.to(dev).double().view(1, ny, 1)
        dz = res.dz_vector.to(dev).double().view(1, 1, nz)
        kx = res.permeability_x.to(dev).double()
        ky = res.permeability_y.to(dev).double()
        kz = res.permeability_z.to(dev).double()
        cell_depth = getattr(res, 'cell_depth', None)
        zc = (cell_depth.to(dev).double() if cell_depth is not None
              else res.z_centers.to(dev).double().view(1, 1, nz).expand(nx, ny, nz))
        idx = torch.arange(self.n_cells, device=dev).reshape(nx, ny, nz)
        tiny = 1e-300
        face_trans = getattr(res, 'face_trans', None)

        conn = getattr(res, 'connections', None)
        if conn is not None:
            # произвольные связи (corner-point с разломами/NNC/PINCH): (L, R, T)
            L, Rr = conn[0].to(dev).long(), conn[1].to(dev).long()
            self.face_L, self.face_R = L, Rr
            self.face_T = conn[2].to(dev).double()
            zc_flat = zc.reshape(-1)
            self.face_dZ = zc_flat[L] - zc_flat[Rr]
            self.face_pos_LR = self.builder.position(L, Rr)
            self.face_pos_RL = self.builder.position(Rr, L)
            self.diag_L = self.builder.diag_map[L]
            self.diag_R = self.builder.diag_map[Rr]
            return

        faces = []
        if face_trans is not None:
            maps = ((self.builder.x_pos_map, self.builder.x_neg_map), (self.builder.y_pos_map, self.builder.y_neg_map),
                    (self.builder.z_pos_map, self.builder.z_neg_map))
            for axis, (t, (mp, mn)) in enumerate(zip(face_trans, maps)):
                if (nx, ny, nz)[axis] < 2:
                    continue
                lo = [slice(None)] * 3
                hi = [slice(None)] * 3
                lo[axis], hi[axis] = slice(None, -1), slice(1, None)
                faces.append((idx[tuple(lo)], idx[tuple(hi)], t.to(dev).double(), mp, mn))
        elif nx > 1:
            area = (dy * dz).expand(nx - 1, ny, nz)
            hl = dx[:-1] / (2 * kx[:-1] + tiny)
            hr = dx[1:] / (2 * kx[1:] + tiny)
            faces.append((idx[:-1], idx[1:], area / (hl + hr), self.builder.x_pos_map, self.builder.x_neg_map))
        if ny > 1 and face_trans is None:
            area = (dx * dz).expand(nx, ny - 1, nz)
            hl = dy[:, :-1] / (2 * ky[:, :-1] + tiny)
            hr = dy[:, 1:] / (2 * ky[:, 1:] + tiny)
            faces.append((idx[:, :-1], idx[:, 1:], area / (hl + hr), self.builder.y_pos_map, self.builder.y_neg_map))
        if nz > 1 and face_trans is None:
            area = (dx * dy).expand(nx, ny, nz - 1)
            hl = dz[:, :, :-1] / (2 * kz[:, :, :-1] + tiny)
            hr = dz[:, :, 1:] / (2 * kz[:, :, 1:] + tiny)
            faces.append((idx[:, :, :-1], idx[:, :, 1:], area / (hl + hr), self.builder.z_pos_map, self.builder.z_neg_map))

        zc_flat = zc.reshape(-1)
        L = torch.cat([f[0].reshape(-1) for f in faces]) if faces else torch.zeros(0, dtype=torch.long, device=dev)
        Rr = torch.cat([f[1].reshape(-1) for f in faces]) if faces else torch.zeros(0, dtype=torch.long, device=dev)
        self.face_L, self.face_R = L, Rr
        self.face_T = torch.cat([f[2].reshape(-1) for f in faces]) if faces else torch.zeros(0, dtype=torch.float64, device=dev)
        self.face_dZ = zc_flat[L] - zc_flat[Rr]
        # позиции внедиагональных блоков: (строка L, столбец R) и (строка R, столбец L)
        self.face_pos_LR = torch.cat([f[3][f[0].reshape(-1)] for f in faces]) if faces else L
        self.face_pos_RL = torch.cat([f[4][f[1].reshape(-1)] for f in faces]) if faces else L
        self.diag_L = self.builder.diag_map[L]
        self.diag_R = self.builder.diag_map[Rr]
        # совместимость: проводимости по осям
        shapes = [(nx - 1, ny, nz), (nx, ny - 1, nz), (nx, ny, nz - 1)]
        off = 0
        for name, shp in zip(("Tx", "Ty", "Tz"), shapes):
            n = shp[0] * shp[1] * shp[2] if min(shp) > 0 else 0
            setattr(self, name, self.face_T[off:off + n].view(shp) if n else torch.zeros(0, device=dev))
            off += n

    # ------------------------------------------------------------------ свойства
    def cell_props(self, p, sw):
        """Плотности, Corey-проницаемости, массовые подвижности λ = kr·ρ/μ и их производные (плоские)."""
        f = self.fluid
        p = p.reshape(-1)
        sw = sw.reshape(-1)
        c_w, c_o = f.water_compressibility, f.oil_compressibility
        rho_w = f.rho_water_ref * (1.0 + c_w * (p - 1e5))
        rho_o = f.rho_oil_ref * (1.0 + c_o * (p - 1e5))
        drho_w = f.rho_water_ref * c_w
        drho_o = f.rho_oil_ref * c_o

        denom = 1.0 - self.sw_irr - self.so_r
        swn = torch.clamp((sw - self.sw_irr) / denom, 0.0, 1.0)
        # односторонняя производная на границах: фронт начинается ровно с Sw = Swc
        active = ((sw >= self.sw_irr) & (sw <= 1.0 - self.so_r)).double()
        krw = swn ** self.nw
        kro = (1.0 - swn) ** self.no
        dkrw = self.nw * swn ** (self.nw - 1) / denom * active
        dkro = -self.no * (1.0 - swn) ** (self.no - 1) / denom * active

        mu_w, mu_o = f.mu_water, f.mu_oil
        return {
            "rho_w": rho_w, "rho_o": rho_o, "drho_w_dp": drho_w, "drho_o_dp": drho_o,
            "krw": krw, "kro": kro, "dkrw_ds": dkrw, "dkro_ds": dkro,
            "lam_w": krw * rho_w / mu_w, "lam_o": kro * rho_o / mu_o,
            "dlam_w_dp": krw * drho_w / mu_w, "dlam_o_dp": kro * drho_o / mu_o,
            "dlam_w_ds": dkrw * rho_w / mu_w, "dlam_o_ds": dkro * rho_o / mu_o,
        }

    def _pv(self, p_flat):
        return self.pore_volume * (1.0 + self.c_rock * (p_flat - self.p_ref_rock))

    def calc_masses(self, p, sw):
        f = self.fluid
        pf, sf = p.reshape(-1), sw.reshape(-1)
        pv = self._pv(pf)
        rho_w = f.rho_water_ref * (1.0 + f.water_compressibility * (pf - 1e5))
        rho_o = f.rho_oil_ref * (1.0 + f.oil_compressibility * (pf - 1e5))
        shp = (self.nx, self.ny, self.nz)
        return (pv * (1 - sf) * rho_o).view(shp), (pv * sf * rho_w).view(shp)

    # ------------------------------------------------------------------ сборка
    def assemble_system(self, p, sw, m_o_old, m_w_old, dt, current_time=None, p_bh=None, props=None):
        """
        Невязка R [кг/с] (2N) и блоки Якобиана dR/d(P,Sw) при фиксированных p_bh.
        Скважинные блоки Шура — в self._last_well_sys.
        """
        n = self.n_cells
        dev = self.device
        pf, sf = p.reshape(-1).double(), sw.reshape(-1).double()
        if props is None:
            props = self.cell_props(pf, sf)
        R = torch.zeros(n * 2, device=dev, dtype=torch.float64)
        Rb = R.view(-1, 2)
        J = torch.zeros((self.builder.n_nonzero, 2, 2), device=dev, dtype=torch.float64)

        # --- накопление ---
        f = self.fluid
        pv = self._pv(pf)
        dpv = self.pore_volume * self.c_rock
        rho_o, rho_w = props["rho_o"], props["rho_w"]
        m_o = pv * (1 - sf) * rho_o
        m_w = pv * sf * rho_w
        Rb[:, 0] += (m_o - m_o_old.reshape(-1)) / dt
        Rb[:, 1] += (m_w - m_w_old.reshape(-1)) / dt
        d = self.builder.diag_map
        J[d, 0, 0] += ((1 - sf) * (pv * props["drho_o_dp"] + rho_o * dpv)) / dt
        J[d, 0, 1] += (-pv * rho_o) / dt
        J[d, 1, 0] += (sf * (pv * props["drho_w_dp"] + rho_w * dpv)) / dt
        J[d, 1, 1] += (pv * rho_w) / dt

        # --- потоки по граням ---
        L, Rr = self.face_L, self.face_R
        if L.numel():
            T, gdz = self.face_T, G * self.face_dZ
            pL, pR = pf[L], pf[Rr]
            for eq, (rho, drho, lam, dl_dp, dl_ds) in enumerate((
                    (rho_o, props["drho_o_dp"], props["lam_o"], props["dlam_o_dp"], props["dlam_o_ds"]),
                    (rho_w, props["drho_w_dp"], props["lam_w"], props["dlam_w_dp"], props["dlam_w_ds"]))):
                rho_avg = 0.5 * (rho[L] + rho[Rr])
                dpot = (pL - pR) - rho_avg * gdz
                up = (dpot >= 0).double()
                dn = 1.0 - up
                lam_f = lam[L] * up + lam[Rr] * dn
                flux = T * lam_f * dpot
                Rb[:, eq].index_add_(0, L, flux)
                Rb[:, eq].index_add_(0, Rr, -flux)

                ddpot_dpL = 1.0 - 0.5 * gdz * drho
                ddpot_dpR = -1.0 - 0.5 * gdz * drho
                dF_dpL = T * (dl_dp[L] * up * dpot + lam_f * ddpot_dpL)
                dF_dsL = T * (dl_ds[L] * up * dpot)
                dF_dpR = T * (dl_dp[Rr] * dn * dpot + lam_f * ddpot_dpR)
                dF_dsR = T * (dl_ds[Rr] * dn * dpot)

                J[:, eq, 0].index_add_(0, self.diag_L, dF_dpL)
                J[:, eq, 1].index_add_(0, self.diag_L, dF_dsL)
                J[:, eq, 0].index_add_(0, self.face_pos_LR, dF_dpR)
                J[:, eq, 1].index_add_(0, self.face_pos_LR, dF_dsR)
                J[:, eq, 0].index_add_(0, self.diag_R, -dF_dpR)
                J[:, eq, 1].index_add_(0, self.diag_R, -dF_dsR)
                J[:, eq, 0].index_add_(0, self.face_pos_RL, -dF_dpL)
                J[:, eq, 1].index_add_(0, self.face_pos_RL, -dF_dsL)

        # --- скважины ---
        self._last_well_sys = None
        if self.wells.n_w:
            if self.wells.step_targets is None:
                t_days = (current_time or 0.0) / 86400.0
                self.wells.begin_step(pf, sf, props, t_days)
            if p_bh is None:
                p_bh = self._p_bh_cur if self._p_bh_cur is not None else self.wells.p_bh
            self._last_well_sys = self.wells.assemble(R, J, d, pf, sf, props, p_bh)
        return R, J

    # ------------------------------------------------------------------ критерии
    def convergence(self, R, dt, pv, wsys, p_bh):
        f = self.fluid
        Rb = R.view(-1, 2)
        rho_ref = torch.tensor([f.rho_oil_ref, f.rho_water_ref], dtype=torch.float64, device=R.device)
        cnv = (dt * Rb.abs() / (rho_ref * pv.unsqueeze(1))).max(dim=0).values
        mb = dt * Rb.sum(dim=0).abs() / (rho_ref * pv.sum())
        well_ok, well_err = True, 0.0
        if wsys is not None:
            rate = wsys["is_rate"]
            tgt = wsys["target"].abs()
            rel = wsys["r_w"].abs() / torch.clamp(tgt, min=1e-12)
            rate_err = torch.where(rate, rel, torch.zeros_like(rel))
            bhp_err = torch.where(rate, torch.zeros_like(rel), wsys["r_w"].abs())
            well_err = float(rate_err.max()) if rate.any() else 0.0
            well_ok = bool((rate_err <= self.tol_well_rate).all() and (bhp_err <= self.tol_well_bhp).all())
        cnv_l, mb_l = cnv.tolist(), mb.tolist()
        ok = max(cnv_l) <= self.tol_cnv and max(mb_l) <= self.tol_mb and well_ok
        return ok, {"cnv": cnv_l, "mb": mb_l, "well_rate_rel_err": well_err}

    # ------------------------------------------------------------------ шаг
    def step(self, dt, current_time=None):
        """Шаг по времени dt [с]. Состояние флюида и скважин фиксируется только при сходимости."""
        dbg = self.debug
        log = (lambda *a, **k: None) if (dbg.enabled and dbg.console) else print
        t_days = (current_time or 0.0) / 86400.0

        p_old = self.fluid.pressure.detach().reshape(-1).double().clone()
        sw_old = self.fluid.s_w.detach().reshape(-1).double().clone()
        p_curr, sw_curr = p_old.clone(), sw_old.clone()
        m_o_old, m_w_old = self.calc_masses(p_old, sw_old)
        m_o_old, m_w_old = m_o_old.reshape(-1), m_w_old.reshape(-1)
        mass_total = (m_o_old.sum().item(), m_w_old.sum().item()) if dbg.enabled else None

        props = self.cell_props(p_curr, sw_curr)
        if self.wells.n_w:
            self.wells.begin_step(p_curr, sw_curr, props, t_days)
            p_bh = self.wells.p_bh.clone()
        else:
            p_bh = torch.zeros(0, dtype=torch.float64, device=self.device)
        self._p_bh_cur = p_bh
        dbg.begin_step(dt, current_time, p_old, sw_old)
        shp = (self.nx, self.ny, self.nz)
        initial_cnv = None

        for k in range(self.max_newton):
            dbg.begin_iter(k)
            timer = dbg.timer()
            with timer('assemble'):
                R, J = self.assemble_system(p_curr, sw_curr, m_o_old, m_w_old, dt, current_time, p_bh, props)
                wsys = self._last_well_sys
                if wsys is not None and k > 0:
                    msgs = self.wells.update_controls(p_bh, wsys["q_surf"])
                    for m in msgs:
                        log(f"    ⇄ {m}")
                    if msgs:
                        if dbg.enabled:
                            dbg.iter_rec["control_switch"] = msgs
                        p_bh = self._reset_bhp_targets(p_bh)
                        self._p_bh_cur = p_bh
                        R, J = self.assemble_system(p_curr, sw_curr, m_o_old, m_w_old, dt, current_time, p_bh, props)
                        wsys = self._last_well_sys
            pv = self._pv(p_curr)
            ok, crit = self.convergence(R, dt, pv, wsys, p_bh)
            dbg.residual(R, dt, mass_total)
            if dbg.enabled:
                dbg.iter_rec["criteria"] = crit
                if wsys is not None and dbg.level >= 2:
                    dbg.wells(self.wells.records(wsys, p_bh, p_curr))
            log(f"  Iter {k}: CNV o/w = {crit['cnv'][0]:.2e}/{crit['cnv'][1]:.2e}, "
                f"MB = {max(crit['mb']):.2e}, wells = {crit['well_rate_rel_err']:.1e}")

            if not torch.isfinite(R).all():
                dbg.converged_iter()
                dbg.end_step(False, "nan_residual", p_curr.view(shp), sw_curr.view(shp), R, J, self.builder)
                return False
            if initial_cnv is None:
                initial_cnv = max(crit["cnv"]) + 1e-300

            if ok and k >= self.min_newton:
                log("  Converged!")
                self.last_newton_iters = k
                self.last_max_dsw = float((sw_curr - sw_old).abs().max())
                self.last_max_dp = float((p_curr - p_old).abs().max())
                dbg.converged_iter()
                dbg.end_step(True, "converged", p_curr.view(shp), sw_curr.view(shp))
                self.fluid.pressure.copy_(p_curr.view(shp).to(self.fluid.pressure.dtype))
                self.fluid.s_w.copy_(sw_curr.view(shp).to(self.fluid.s_w.dtype))
                self.fluid.s_o.copy_((1.0 - sw_curr).view(shp).to(self.fluid.s_o.dtype))
                if self.wells.n_w:
                    self.wells.commit(p_bh)
                    self._store_well_results(wsys, p_bh)
                self._p_bh_cur = None
                return True

            dbg.jacobian(J, self.builder.diag_map)
            if dbg.level >= 3:
                with timer('fd_check'):
                    dbg.fd_check(self, p_curr.view(shp), sw_curr.view(shp), m_o_old, m_w_old, dt, current_time, J)
                dbg.maybe_dump_iter(p_curr.view(shp), sw_curr.view(shp), R, J, self.builder, k)

            rel = max(crit["cnv"]) / initial_cnv
            with timer('linear'):
                sol = self.solve_linear(J, R, wsys, newton_rel=rel, rebuild_amg=(k == 0 or self.amg_rebuild == 'iteration'))
            dbg.linear(self._last_linear_stats)
            if sol is None:
                dbg.end_iter(timer.t if dbg.enabled else {})
                Rf, Jf = self.assemble_system(p_curr, sw_curr, m_o_old, m_w_old, dt, current_time, p_bh)
                dbg.end_step(False, "linear_solver_failed", p_curr.view(shp), sw_curr.view(shp), Rf, Jf, self.builder)
                return False
            dx, dpw = sol

            # --- Appleyard-подобное ограничение по ячейкам ---
            dp = dx[:, 0]
            ds = dx[:, 1]
            dp_lim = torch.clamp(self.dp_max_rel * p_curr.abs(), min=self.dp_max_abs)
            n_chop_p = int((dp.abs() > dp_lim).sum()) if dbg.enabled else 0
            n_chop_s = int((ds.abs() > self.ds_max).sum()) if dbg.enabled else 0
            dbg.update(dp, ds, 1.0)
            if dbg.enabled:
                dbg.iter_rec["update"].update(n_chopped_p=n_chop_p, n_chopped_sw=n_chop_s)
            dp = torch.maximum(torch.minimum(dp, dp_lim), -dp_lim)
            ds = ds.clamp(-self.ds_max, self.ds_max)
            p_curr = p_curr + dp
            sw_curr = (sw_curr + ds).clamp(0.0, 1.0)
            if dpw.numel():
                dpw_lim = torch.clamp(self.dbhp_max_rel * p_bh.abs(), min=self.dp_max_abs)
                p_bh = p_bh + torch.maximum(torch.minimum(dpw, dpw_lim), -dpw_lim)
                p_bh = p_bh.clamp_min(1e3)
                p_cell_floor = (p_curr < 0).any()
                if bool(p_cell_floor):
                    p_curr = p_curr.clamp_min(1e3)
                if dbg.enabled:
                    dbg.iter_rec["update"]["max_dp_bh_bar"] = float(dpw.abs().max()) / 1e5
            self._p_bh_cur = p_bh
            props = self.cell_props(p_curr, sw_curr)
            dbg.line_search([], 1.0, True)
            dbg.end_iter(timer.t if dbg.enabled else {})

        Rf, Jf = self.assemble_system(p_curr, sw_curr, m_o_old, m_w_old, dt, current_time, p_bh)
        dbg.end_step(False, "max_newton_iter", p_curr.view(shp), sw_curr.view(shp), Rf, Jf, self.builder)
        self._p_bh_cur = None
        return False

    def _reset_bhp_targets(self, p_bh):
        p_bh = p_bh.clone()
        for w_idx, w in enumerate(self.wells.wells):
            if self.wells.step_mode[w_idx] == 'bhp_limit':
                p_bh[w_idx] = self.wells.bhp_limit(w)
        return p_bh

    def _store_well_results(self, wsys, p_bh):
        for w_idx, w in enumerate(self.wells.wells):
            m = self.wells.conn_well == w_idx
            q_o = float(wsys["q_o"][m].sum()) / self.fluid.rho_oil_ref * 86400.0
            q_w = float(wsys["q_w"][m].sum()) / self.fluid.rho_water_ref * 86400.0
            w.bhp_mpa = float(p_bh[w_idx]) / 1e6
            w.last_surface_rates = {"oil": q_o, "water": q_w, "liquid": q_o + q_w}

    # ------------------------------------------------------------------ СЛАУ
    def solve_linear(self, J, R, wsys, newton_rel=1.0, rebuild_amg=True):
        """
        Решает систему Ньютона с исключёнными скважинами (дополнение Шура) методом FGMRES + CPR.
        Возвращает (dx [N,2], dp_bh [n_w]) или None при отказе.
        """
        n = self.n_cells
        nb = J.shape[1]
        stats = {"backend": "fgmres_cpr", "n_wells": self.wells.n_w}
        wells_on = wsys is not None
        if wells_on:
            self.wells.add_local_schur(wsys, J)
        rhs = -R.view(-1, nb).clone()
        if wells_on:
            rhs += self.wells.schur_rhs(wsys, n)

        dg = J[self.builder.diag_map]
        # масштаб строки — по диагонали; если она вырождена относительно строки блока (нефть в ячейке без
        # нефти: ∂M_o/∂p = 0, ∂M_o/∂Sw ≠ 0), — по максимуму строки блока, иначе 1/0 переполняет нормы Крылова
        d = torch.diagonal(dg, dim1=1, dim2=2).abs()
        row_max = dg.abs().amax(dim=2)
        ref = torch.where(d > 1e-8 * row_max, d, row_max)
        scale = torch.where(ref > 0, 1.0 / ref.clamp_min(1e-300), torch.ones_like(ref))
        rows = torch.repeat_interleave(torch.arange(n, device=J.device),
                                       self.builder.indptr[1:] - self.builder.indptr[:-1])
        J.mul_(scale[rows].unsqueeze(2))
        rhs = rhs * scale

        # изолированные ячейки (строка — только диагональный блок, без скважин: неактивные ячейки corner-point
        # сетки) решаются точно блоком 3×3, Крылов и AMG — только по связанной части
        sub = self._linear_subsystem()
        dx_full = None
        if sub is not None:
            keep, iso, vpos, indptr_s, indices_s, rows_s = sub
            dx_full = torch.zeros_like(rhs)
            dx_full[iso] = torch.linalg.solve(J[self.builder.diag_map[iso]], rhs[iso].unsqueeze(-1)).squeeze(-1)
            Js, rhs_s, n_s = J[vpos], rhs[keep], int(keep.numel())
        else:
            Js, rhs_s, n_s = J, rhs, n
            indptr_s, indices_s, rows_s = self.builder.indptr, self.builder.indices, rows
        b = rhs_s.reshape(-1)

        def matvec(x):
            xb = x.view(-1, nb)
            y = torch.zeros_like(xb)
            y.index_add_(0, rows_s, torch.bmm(Js, xb[indices_s].unsqueeze(-1)).squeeze(-1))
            if wells_on:
                if sub is None:
                    y += self.wells.schur_correction(wsys, xb, local_included=True) * scale
                else:
                    xf = torch.zeros((n, nb), dtype=xb.dtype, device=xb.device)
                    xf[keep] = xb
                    y += (self.wells.schur_correction(wsys, xf, local_included=True) * scale)[keep]
            return y.view(-1)

        t0 = time.perf_counter()
        amg_in = None if (rebuild_amg or self._amg is None) else self._amg
        cpr = CPRPreconditionerGPU(Js, indptr_s, indices_s, n_s, self.device,
                                   amg=amg_in, smoother_sweeps=self.cpr_smoother_sweeps,
                                   amg_verbose=self.amg_verbose)
        if cpr.amg is None and amg_in is not None:
            cpr = CPRPreconditionerGPU(Js, indptr_s, indices_s, n_s, self.device,
                                       amg=None, smoother_sweeps=self.cpr_smoother_sweeps,
                                       amg_verbose=self.amg_verbose)
        self._amg = cpr.amg
        stats.update(amg_levels=cpr.amg_levels, amg_rebuilt=cpr.amg_rebuilt, amg_setup_s=cpr.amg_build_s,
                     amg_error=cpr.amg_error, precond_setup_s=time.perf_counter() - t0)
        if self.debug.level >= 2 and cpr.amg_log:
            self.debug.event("captured_output", tag="cpr_amg", text=cpr.amg_log)

        tol = max(self.lin_tol_min, min(self.lin_tol_max, 0.1 * newton_rel))
        t1 = time.perf_counter()
        x, info = fgmres(matvec, b, M_mv=cpr.matvec, max_iter=self.lin_max_restarts, restart=self.lin_restart,
                         tol=tol, verbose=self.linear_verbose, stats=stats)
        stats["solve_s"] = time.perf_counter() - t1
        stats["tol"] = tol
        true_rel = ((matvec(x) - b).norm() / b.norm().clamp_min(1e-300)).item()
        stats["true_rel_res"] = true_rel
        self._last_linear_stats = stats
        if not (true_rel == true_rel) or true_rel > self.lin_accept:
            return None
        dx = x.view(-1, nb)
        if sub is not None:
            dx_full[keep] = dx
            dx = dx_full
        dpw = self.wells.recover_dpw(wsys, dx) if wells_on else torch.zeros(0, dtype=torch.float64, device=self.device)
        return dx, dpw

    def _linear_subsystem(self):
        """
        Связная часть линейной системы без изолированных ячеек: (keep, iso, позиции блоков, indptr, indices,
        строки блоков) или None, если изолированных ячеек нет. Структура матрицы постоянна — считается один раз.
        """
        if getattr(self, "_lin_sub_ready", False):
            return self._lin_sub
        self._lin_sub_ready, self._lin_sub = True, None
        if not getattr(self, "linear_drop_isolated", True):
            return None
        indptr, indices = self.builder.indptr, self.builder.indices
        n = self.n_cells
        dev = indptr.device
        iso = (indptr[1:] - indptr[:-1]) == 1
        wc = getattr(self.wells, "conn_cell", None)
        if wc is not None and wc.numel():
            iso[wc] = False
        if int(iso.sum()) == 0:
            return None
        keep = (~iso).nonzero().view(-1)
        new = torch.full((n,), -1, dtype=torch.long, device=dev)
        new[keep] = torch.arange(keep.numel(), device=dev)
        cnt = indptr[1:][keep] - indptr[:-1][keep]
        indptr_s = torch.zeros(keep.numel() + 1, dtype=indptr.dtype, device=dev)
        indptr_s[1:] = torch.cumsum(cnt, 0)
        rows_s = torch.repeat_interleave(torch.arange(keep.numel(), device=dev), cnt)
        start = torch.repeat_interleave(indptr[:-1][keep], cnt)
        vpos = start + torch.arange(rows_s.numel(), device=dev) - torch.repeat_interleave(indptr_s[:-1], cnt)
        indices_s = new[indices[vpos]].to(indices.dtype)
        if bool((indices_s < 0).any()):
            return None
        self._lin_sub = (keep, iso.nonzero().view(-1), vpos, indptr_s, indices_s, rows_s)
        return self._lin_sub
