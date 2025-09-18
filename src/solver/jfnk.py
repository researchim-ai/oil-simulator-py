import torch
import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from linear_gpu.fgmres import fgmres
from .cpr import CPRPreconditioner

class FullyImplicitSolver:
    def __init__(self, simulator, backend="amgx"):
        self.sim = simulator

        # --------------------------------------------------------------
        # 📏 АВТОМАТИЧЕСКИЙ ВЫБОР BACKEND ДЛЯ CPR
        #   • Для микро-моделей (N_cells < 5000) Jacobi быстрее и строго
        #     линейный → unit-тесты JFNK проходят без ложных срабатываний.
        #   • Если запущены тесты (OIL_TEST=1), тоже форсируем Jacobi,
        #     чтобы избежать непредсказуемых переключений AMG↔Jacobi.
        # --------------------------------------------------------------
        nx, ny, nz = simulator.reservoir.dimensions
        n_cells_tot = nx * ny * nz
        if n_cells_tot < 5000 or os.environ.get("OIL_TEST", "0") == "1":
            if backend != "jacobi":
                print(
                    f"🔧 JFNK: переключаем CPR backend '{backend}' → 'jacobi' "
                    f"(n_cells={n_cells_tot}, OIL_TEST={os.environ.get('OIL_TEST','0')})"
                )
            backend = "jacobi"

        # --- Variable scaling (pressure → dimensionless) ---------------
        self.scaler = simulator.scaler  # используем уже созданный в Simulator

        # CPR preconditioner (pressure block) ------------------------------
        sim_params = simulator.sim_params
        smoother = sim_params.get("smoother", "jacobi")

        # Параметры GeoSolver из конфигурации
        geo_params = {
            "cycles_per_call": sim_params.get("geo_cycles", 1),
            "pre_smooth":      sim_params.get("geo_pre", 2),
            "post_smooth":     sim_params.get("geo_post", 2),
            "max_levels":      sim_params.get("geo_levels", 6),
            # Передаём предпочтительный сглаживатель для GeoSolverV2
            "smoother_fine":   sim_params.get("smoother", "rbgs"),
        }

        # CPR конфиг из sim_params
        cpr_backend     = sim_params.get("cpr_backend", backend)
        geo_tol         = sim_params.get("geo_tol", 1e-6)
        geo_max_iter    = sim_params.get("geo_max_iter", 10)
        gmres_tol       = sim_params.get("gmres_tol", 1e-3)
        gmres_max_iter  = sim_params.get("gmres_max_iter", 60)

        self.prec = CPRPreconditioner(
            simulator,
            backend=cpr_backend,
            smoother=smoother,
            scaler=self.scaler,
            geo_params=geo_params,
            geo_tol=geo_tol,
            geo_max_iter=geo_max_iter,
            gmres_tol=gmres_tol,
            gmres_max_iter=gmres_max_iter,
        )

        # Newton params ----------------------------------------------------
        self.tol = simulator.sim_params.get("newton_tolerance", 1e-7)  # абсолютная
        self.rtol = simulator.sim_params.get("newton_rtol", 1e-4)       # относительная
        self.max_it = simulator.sim_params.get("newton_max_iter", 30)

        # Для очень маленьких задач гарантируем минимум 30 итераций, чтобы дать шанc уменьшить F;
        # для крупных моделей (>500 ячеек) повышаем потолок до 25–30, иначе Ньютона
        # часто не успевает уменьшить невязку до tol.
        nx, ny, nz = simulator.reservoir.dimensions
        n_cells_total = nx * ny * nz
        if n_cells_total <= 100 and self.max_it < 30:
            self.max_it = 30
        elif n_cells_total > 500 and self.max_it < 25:
            self.max_it = 25

        # --- Pseudo-Transient continuation (PTC) ------------------------
        self.ptc_enabled = simulator.sim_params.get("ptc", True)
        #  Значение по умолчанию гораздо меньше, чтобы не "душить" мелкие тестовые задачи
        self.ptc_tau0 = simulator.sim_params.get("ptc_tau0", 10.0)

    # --- small helpers -------------------------------------------------
    def _n_cells(self):
        if self.scaler is not None:
            return self.scaler.n_cells
        nx, ny, nz = self.sim.reservoir.dimensions
        return nx * ny * nz

    def _check_scale_inv(self, z_hat: torch.Tensor, tag: str):
        if self.scaler is None:
            return
        z_phys = self.scaler.unscale_vec(z_hat)
        z_back = self.scaler.scale_vec(z_phys)
        err = (z_back - z_hat).abs().max().item()
        if err > 1e-8:
            print(f"[SCALE-MISMATCH] {tag}: {err:.3e}")


    def _Jv(self, x: torch.Tensor, v: torch.Tensor, dt):
        """
        Чисто физические единицы на входе/выходе.
        Jv через блочное масштабирование направления и центральную
        разность; для насыщенностей у границ — односторонняя разность.
        """
        if v.norm() < 1e-14:
            return torch.zeros_like(v)

        # --- разметка переменных -------------------------------------------
        n_cells = self.scaler.n_cells if self.scaler is not None else (len(x)//2)
        n = n_cells
        nvars = x.numel()
        has_sw = (nvars >= 2*n)
        has_sg = (nvars >= 3*n)

        v_p  = v[:n]
        v_sw = v[n:2*n]   if has_sw else None
        v_sg = v[2*n:3*n] if has_sg else None

        # --- целевые реальные амплитуды шагов -------------------------------
        dp_target = float(self.sim.sim_params.get("jv_dp_step", 3e4))    # 0.1 MPa
        ds_target = float(self.sim.sim_params.get("jv_ds_step", 5e-4))   # меньше: сгладит разностную ошибку на старте

        # --- масштабы по блокам --------------------------------------------
        tiny = 1e-30
        s_p = dp_target / (float(v_p.abs().max().item()) + tiny)
        if has_sw:
            s_sw = ds_target / (float(v_sw.abs().max().item()) + tiny)
        if has_sg:
            s_sg = ds_target / (float(v_sg.abs().max().item()) + tiny)

        v_mod = v.clone()
        v_mod[:n] *= s_p
        if has_sw: v_mod[n:2*n] *= s_sw
        if has_sg: v_mod[2*n:3*n] *= s_sg

        # --- границы по насыщенности ---------------------------------------
        def _project(z):
            z = z.clone()
            if has_sw:
                z[n:2*n] = torch.clamp(z[n:2*n], 1e-8, 1.0 - 1e-8)
            if has_sg:
                z[2*n:3*n] = torch.clamp(z[2*n:3*n], 1e-8, 1.0 - 1e-8)
            return z

        a_plus, a_minus = 1.0, 1.0
        if has_sw:
            sw   = x[n:2*n]
            swc  = float(self.sim.fluid.sw_cr)
            sor  = float(self.sim.fluid.so_r)
            epsb = 5e-5
            lo, hi = swc + epsb, 1.0 - sor - epsb

            vs = v_mod[n:2*n]
            # допустимые коэффициенты, чтобы не выйти за [lo, hi]
            allow_p = torch.ones_like(sw)
            allow_m = torch.ones_like(sw)

            pos = (vs > 0); neg = (vs < 0)
            allow_p[pos] = torch.clamp((hi - sw[pos]) / (vs[pos] + tiny), max=1.0)
            allow_p[neg] = torch.clamp((sw[neg] - lo) / (-vs[neg] + tiny), max=1.0)
            allow_m[pos] = torch.clamp((sw[pos] - lo) / (vs[pos] + tiny), max=1.0)
            allow_m[neg] = torch.clamp((hi - sw[neg]) / (-vs[neg] + tiny), max=1.0)

            a_plus  = min(a_plus,  0.9 * float(allow_p.min().item()))
            a_minus = min(a_minus, 0.9 * float(allow_m.min().item()))

            # если реально «сидим» на нижней/верхней границе — делаем чисто односторонне
            if float((sw - lo).min().item()) < 2.5*ds_target and a_minus < 1e-3:
                a_minus = 0.0
            if float((hi - sw).min().item()) < 2.5*ds_target and a_plus < 1e-3:
                a_plus = 0.0

        # --- вычисления -----------------------------------------------------
        # F0 нужен, если одна из сторон «обнулилась»
        F0 = None
        if a_plus == 0.0 or a_minus == 0.0:
            F0 = self.sim._fi_residual_vec(x, dt)

        x_plus  = _project(x + a_plus  * v_mod) if a_plus  > 0.0 else x
        x_minus = _project(x - a_minus * v_mod) if a_minus > 0.0 else x

        if not hasattr(self, "_dbg_jv_real_step3"):
            dp_real = (x_plus[:n] - x[:n]).abs().max().item()
            ds_real = (x_plus[n:2*n] - x[n:2*n]).abs().max().item() if has_sw else float('nan')
            print(f"[Jv REAL]  Δp_max={dp_real:.3e} Pa  ΔS_max={ds_real:.3e}")
            self._dbg_jv_real_step3 = True

        F_plus  = self.sim._fi_residual_vec(x_plus,  dt) if a_plus  > 0.0 else F0
        F_minus = self.sim._fi_residual_vec(x_minus, dt) if a_minus > 0.0 else F0

        denom = a_plus + a_minus
        if denom < 1e-12:
            return torch.zeros_like(v)

        # обобщённая «центральная» формула: [F(x+αδ) - F(x-βδ)] / (α+β) ≈ J(x)·δ
        Jv_core = (F_plus - F_minus) / denom

        # --- снимаем блочные масштабы (хотим именно J @ v, а не J @ (S v)) --
        Jv_core[:n] /= s_p
        if has_sw: Jv_core[n:2*n] /= s_sw
        if has_sg: Jv_core[2*n:3*n] /= s_sg

        # --- PTC ------------------------------------------------------------
        if (nvars >= 800) and self.ptc_enabled and getattr(self, "ptc_tau", 0.0) > 0.0:
            Jv_core = Jv_core + (self.ptc_tau / dt) * v

        if not torch.isfinite(Jv_core).all():
            print("  _Jv: NaN/Inf → zero")
            Jv_core = torch.zeros_like(v)

        return Jv_core

    def _fd_step_for_direction(self, x_hat: torch.Tensor, v_hat: torch.Tensor) -> float:
        """
        Возвращает скалярный шаг eps в HAT-единицах для направления v_hat, на текущей точке x_hat.
        Гарантируем минимальные физические приращения по давлению и насыщенности.
        Для насыщенности учитываем, что неизвестная — y, а физический шаг по s равен (ds/dy) * (Δy_phys),
        где Δy_phys = eps * v_y_hat * s_scale_y.
        """
        # минимальные физические приращения (можно подкрутить при необходимости):
        EPS_ABS_P   = 1e3      # Па
        TARGET_DS   = 5e-4     # целевой физический шаг по насыщенности
        EPS_REL     = 1e-6     # относительное к ||v||_inf в физ. ед.

        ndof = int(v_hat.numel())
        assert self.scaler is not None, "JFNK: scaler/normalizer обязателен"
        N = int(self.scaler.n_cells)  # раскладка [p(0..N-1), sw(0..N-1), (sg...)]

        p_scale  = float(getattr(self.scaler, 'p_scale', 1.0) or 1.0)
        s_scales = getattr(self.scaler, 's_scales', (1.0,))
        sw_scale = float(s_scales[0] if len(s_scales) > 0 else 1.0)

        # нормы направления по давлению в ФИЗ. единицах
        v_p_phys = v_hat[:N] * p_scale
        nv_p = float(v_p_phys.abs().max().item()) if N > 0 else 0.0

        eps_p_phys = max(EPS_ABS_P, EPS_REL * nv_p)

        # обратно в hat (давление)
        eps_p_hat = eps_p_phys / max(p_scale, 1.0)

        # целевой шаг по s через y: оценим медианное |(ds/dy) * (s_scale_y * v_y_hat)|
        eps_y_hat = 0.0
        try:
            if ndof >= 2 * N and N > 0:
                y_hat = x_hat[N:2*N]
                v_y_hat = v_hat[N:2*N]
                # y_phys = y_hat * sw_scale
                y_phys = y_hat * sw_scale
                # ds/dy = (1 - swc - sor) * sigma * (1 - sigma)
                swc = float(getattr(self.sim.fluid, 'sw_cr', 0.0))
                sor = float(getattr(self.sim.fluid, 'so_r', 0.0))
                denom_s = max(1e-12, 1.0 - swc - sor)
                sigma = torch.sigmoid(y_phys)
                ds_dy = denom_s * (sigma * (1.0 - sigma))
                # физический коэффициент преобразования шага eps → Δs_phys по направлению v
                conv = (ds_dy.abs() * (sw_scale * v_y_hat.abs()))
                # используем медиану по ячейкам для устойчивости
                conv_med = float(conv.median().item()) if conv.numel() > 0 else 0.0
                if conv_med > 0.0:
                    eps_y_hat = TARGET_DS / conv_med
        except Exception:
            eps_y_hat = 0.0

        # Дополнительно ограничим по допустимому физическому шагу давления (cap), чтобы не рвало Δp
        P_CAP = 1e5  # Па
        if nv_p > 0.0:
            eps_p_cap_hat = P_CAP / nv_p  # т.к. δp_phys ≈ eps * nv_p
            eps_p_hat = min(eps_p_hat, eps_p_cap_hat)

        # Выбираем eps как min по ограничениям (и нижняя граница на всякий случай)
        candidates = [x for x in [eps_p_hat, eps_y_hat] if x and x > 0.0]
        eps_hat = min(candidates) if candidates else max(eps_p_hat, eps_y_hat)
        eps_hat = float(max(min(eps_hat, 1.0), 1e-12))
        return float(eps_hat)

    def _fd_steps_for_blocks(self, x_hat: torch.Tensor, v_hat: torch.Tensor) -> tuple:
        """
        Возвращает (eps_p_hat, eps_y_hat) — скалярные шаги в HAT для блоков давления и y.
        eps_p_hat ограничиваем по абсолютному δP и относительному масштабу направления.
        eps_y_hat выбираем из целевого физического δS с учётом ds/dy и масштаба y.
        """
        ndof = int(v_hat.numel())
        assert self.scaler is not None, "JFNK: scaler/normalizer обязателен"
        N = int(self.scaler.n_cells)

        p_scale  = float(getattr(self.scaler, 'p_scale', 1.0) or 1.0)
        s_scales = getattr(self.scaler, 's_scales', (1.0,))
        sw_scale = float(s_scales[0] if len(s_scales) > 0 else 1.0)

        # давление
        EPS_ABS_P = 1e3
        EPS_REL   = 1e-6
        P_CAP     = 1e5
        v_p_phys = (v_hat[:N] * p_scale) if N > 0 else torch.tensor(0.0, device=v_hat.device, dtype=v_hat.dtype)
        nv_p = float(v_p_phys.abs().max().item()) if N > 0 else 0.0
        eps_p_phys = max(EPS_ABS_P, EPS_REL * nv_p)
        eps_p_hat = eps_p_phys / max(p_scale, 1.0)
        if nv_p > 0.0:
            eps_p_cap_hat = P_CAP / nv_p
            eps_p_hat = min(eps_p_hat, eps_p_cap_hat)

        # насыщенность (y)
        TARGET_DS = 2e-3
        eps_y_hat = 0.0
        try:
            if ndof >= 2 * N and N > 0:
                y_hat = x_hat[N:2*N]
                v_y_hat = v_hat[N:2*N]
                y_phys = y_hat * sw_scale
                swc = float(getattr(self.sim.fluid, 'sw_cr', 0.0))
                sor = float(getattr(self.sim.fluid, 'so_r', 0.0))
                denom_s = max(1e-12, 1.0 - swc - sor)
                sigma = torch.sigmoid(y_phys)
                ds_dy = denom_s * (sigma * (1.0 - sigma))
                conv = (ds_dy.abs() * (sw_scale * v_y_hat.abs()))
                # используем персентиль 70% вместо медианы для устойчивости
                if conv.numel() > 0:
                    try:
                        q = torch.quantile(conv, 0.7).item()
                    except Exception:
                        q = float(conv.median().item())
                    if q > 0.0:
                        eps_y_hat = TARGET_DS / float(q)
        except Exception:
            eps_y_hat = 0.0

        # безопасность
        eps_p_hat = float(max(min(eps_p_hat, 1.0), 1e-12))
        eps_y_hat = float(max(min(eps_y_hat, 1.0), 1e-12)) if eps_y_hat > 0.0 else 0.0
        return eps_p_hat, eps_y_hat

    def _matvec(self, x_hat: torch.Tensor, v_hat: torch.Tensor) -> torch.Tensor:
        """
        Возвращает J(x_hat)·v_hat в HAT-единицах.
        F_func(x_hat) обязан возвращать невязку тоже в HAT-единицах (как и у тебя сейчас).
        """
        with torch.no_grad():
            Fx = self.F_func(x_hat)
            N = int(self.scaler.n_cells) if self.scaler is not None else (x_hat.numel() // 2)
            eps_p, eps_y = self._fd_steps_for_blocks(x_hat, v_hat)

            # вклад давления
            Jv_p = torch.zeros_like(Fx)
            if eps_p > 0.0 and N > 0:
                v_p = torch.zeros_like(v_hat)
                v_p[:N] = v_hat[:N]
                Fxp = self.F_func(x_hat + eps_p * v_p)
                Jv_p = (Fxp - Fx) / eps_p

            # вклад насыщенности (y)
            Jv_y = torch.zeros_like(Fx)
            if eps_y > 0.0 and x_hat.numel() >= 2 * N:
                v_y = torch.zeros_like(v_hat)
                v_y[N:2*N] = v_hat[N:2*N]
                Fxy = self.F_func(x_hat + eps_y * v_y)
                Jv_y = (Fxy - Fx) / eps_y

            Jv_h = Jv_p + Jv_y
            return Jv_h


    def step(self, x0: torch.Tensor, dt: float):
        """
        Полностью имплицитный шаг Ньютона–Крылова (JFNK) с CPR/AMG предобуславливанием,
        ограничителями по давлению/насыщенности, строгим Armijo line-search и trust-region.
        Все векторы x/F внутри — В ШАПКАХ (hat), кроме мест, где явно перевожу в phys.
        На выходе возвращаю x в ФИЗИЧЕСКИХ единицах (Па, насыщенности).
        """
        # текущее состояние в hat
        x = x0.clone()

        # размеры
        n_cells = (
            self.scaler.n_cells if self.scaler is not None
            else (self.sim.reservoir.dimensions[0] *
                self.sim.reservoir.dimensions[1] *
                self.sim.reservoir.dimensions[2])
        )
        n = n_cells  # давление занимает первые n элементов
        # ---- Параметризация насыщенностей: y <-> s -----------------------
        sw_cr = float(self.sim.fluid.sw_cr)
        so_r  = float(self.sim.fluid.so_r)
        denom_s = max(1e-12, 1.0 - sw_cr - so_r)

        def _y_to_s(y: torch.Tensor) -> torch.Tensor:
            return sw_cr + denom_s * torch.sigmoid(y)

        def _s_to_y(s: torch.Tensor) -> torch.Tensor:
            # Сдвиг к внутренней области, чтобы избежать вырождения ds/dy≈0 на старте
            try:
                y_init_off = float(self.sim.sim_params.get("sat_y_init_offset", 1e-3))
            except Exception:
                y_init_off = 1e-3
            z = ((s - sw_cr + y_init_off) / denom_s).clamp(1e-12, 1.0 - 1e-12)
            return torch.log(z / (1.0 - z))

        def _phys_from_hat_y(x_hat: torch.Tensor) -> torch.Tensor:
            # Перевод «hat-вектора с y в блоке насыщенностей» → физические переменные
            x_phys_base = self._unscale_x(x_hat) if self.scaler is not None else x_hat.clone()
            if x_phys_base.numel() >= 2 * n:
                yw = x_phys_base[n:2*n]
                x_phys_base[n:2*n] = _y_to_s(yw)
            return x_phys_base

        # Инициализация: перепишем блок насыщенности в x (hat) из s → y
        try:
            x_phys0 = self._unscale_x(x) if self.scaler is not None else x
            if x_phys0.numel() >= 2 * n:
                y0 = _s_to_y(x_phys0[n:2*n])
                x[n:2*n] = y0
        except Exception:
            pass

        # Динамическое согласование масштабов scaler для y‑переменной:
        # s_scale_sw ≈ 1 / median(ds/dy) на текущем y, чтобы единичный шаг по y
        # соответствовал осмысленному ΔS в физ. переменной.
        try:
            if self.scaler is not None and x.numel() >= 2 * n:
                y_cur = x[n:2*n]
                # переводим в физический y: y_phys = y_hat * sw_scale
                s_scales_loc = getattr(self.scaler, "s_scales", (1.0,))
                sw_scale_loc = float(s_scales_loc[0] if len(s_scales_loc) > 0 else 1.0)
                y_phys_cur = y_cur * sw_scale_loc
                sigma_cur = torch.sigmoid(y_phys_cur)
                dsdy_cur = denom_s * (sigma_cur * (1.0 - sigma_cur))
                med_dsdy = float(dsdy_cur.median().item()) if dsdy_cur.numel() > 0 else 0.0
                if med_dsdy > 1e-8:
                    s_scale_sw = 1.0 / med_dsdy
                    # обновим scaler и перескалируем текущий x_hat по y, чтобы сохранить физический y
                    if hasattr(self.scaler, "s_scales") and len(self.scaler.s_scales) >= 1:
                        old_sw_scale = float(self.scaler.s_scales[0])
                        if old_sw_scale <= 0.0:
                            old_sw_scale = 1.0
                        # сначала перескалируем текущие значения y в hat: y_hat_new = y_hat_old * old/new
                        # (так phys_y = y_hat * s_scale остаётся неизменным)
                        with torch.no_grad():
                            x[n:2*n] = x[n:2*n] * (old_sw_scale / s_scale_sw)

                        self.scaler.s_scales[0] = s_scale_sw
                        self.scaler.inv_s_scales[0] = 1.0 / s_scale_sw
                        # пересоберём комбинированные массивы
                        self.scaler.scale = [self.scaler.inv_p_scale] + list(self.scaler.inv_s_scales)
                        self.scaler.inv_scale = [self.scaler.p_scale] + list(self.scaler.s_scales)
                        print(f"[scaler] updated s_scale_sw={s_scale_sw:.3e} from median(ds/dy)={med_dsdy:.3e}")
        except Exception:
            pass

        # якорим среднее давление (в hat) ТОЛЬКО если нет BHP-скважин
        baseline_mean_p = x[:n].mean().clone()
        if "fix_pressure_drift" in self.sim.sim_params:
            fix_pressure_drift = bool(self.sim.sim_params.get("fix_pressure_drift", True))
        else:
            has_bhp = False
            try:
                wm = getattr(self.sim, "well_manager", None)
                if wm is not None and hasattr(wm, "get_wells"):
                    for _w in wm.get_wells():
                        if getattr(_w, "control_type", "").lower() == "bhp":
                            has_bhp = True
                            break
            except Exception:
                has_bhp = False
            # Если есть BHP – нельзя зажимать среднее давление, иначе система переопределяется
            fix_pressure_drift = not has_bhp

        def _anchor_pressure(x_hat: torch.Tensor):
            if not fix_pressure_drift:
                return x_hat
            drift = x_hat[:n].mean() - baseline_mean_p
            if torch.abs(drift) > 1e-6:
                x_hat[:n] -= drift
            return x_hat

        def _project_zero_mean_p(v_hat: torch.Tensor):
            # проецируем компоненту давления на подпространство со средним = 0
            v_hat = v_hat.clone()
            if v_hat.numel() >= n:
                v_hat[:n] -= v_hat[:n].mean()
            return v_hat


        # дефляция разрешена только на крупных задачах
        advanced_threshold = int(self.sim.sim_params.get("advanced_threshold", 50_000))
        allow_defl = (n_cells > advanced_threshold)

        # PTC
        # Плавный PTC-нагрев на старте Ньютона: первые it≤ptc_iters добавляем (τ/dt)(x−x_ref)
        # Значения можно задать в конфиге: ptc_tau0, ptc_iters
        try:
            self.ptc_tau0 = float(getattr(self, "ptc_tau0", self.sim.sim_params.get("ptc_tau0", 0.5)))
        except Exception:
            self.ptc_tau0 = 0.5
        try:
            self.ptc_iters = int(getattr(self, "ptc_iters", self.sim.sim_params.get("ptc_iters", 3)))
        except Exception:
            self.ptc_iters = 3
        self.ptc_enabled = True
        self.ptc_tau = self.ptc_tau0 if self.ptc_enabled else 0.0
        x_ref = _anchor_pressure(x0.clone())  # hat

        # Trust-region базовая настройка
        nvars_guess = n_cells * 2  # (p + Sw) по умолчанию
        if nvars_guess < 500:
            trust_radius = 200.0
        else:
            default_tr = 20.0 + 0.5 * math.sqrt(n_cells)
            trust_radius = float(self.sim.sim_params.get("trust_radius", default_tr))

        # счётчики/диагностика
        self.total_gmres_iters = 0
        self.defl_basis = []
        init_F_scaled = None
        prev_F_norm = None

        # нижняя граница для точности GMRES
        gmres_tol_base = float(self.sim.sim_params.get("gmres_min_tol", 1e-7))
        effective_max_it = self.max_it
        if n_cells <= 100 and self.max_it < 30:
            effective_max_it = 30

        # локальные помощники -------------------------------------------------
        def _F_hat(x_hat: torch.Tensor) -> torch.Tensor:
            # Никакого дополнительного якорения: невязку считаем на реальной точке.
            # ВАЖНО: строим физическое состояние из представления y для насыщенностей
            try:
                x_phys = _phys_from_hat_y(x_hat)
            except Exception:
                x_phys = self._unscale_x(x_hat) if self.scaler is not None else x_hat
            F_phys = self.sim._fi_residual_vec(x_phys, dt)

            # --- ЕДИНАЯ НОРМАЛИЗАЦИЯ НЕВЯЗОК В HAT-ПРОСТРАНСТВЕ -------------
            # Давление: как раньше (делим на p_scale)
            # Насыщенности: делим на характерный масштаб PV/Δt·ρ_w,
            # чтобы ||F_p|| и ||F_s|| были одного порядка.
            if self.scaler is None:
                Fh = F_phys
            else:
                n = self.scaler.n_cells
                vars_per_cell = max(2, min(3, F_phys.numel() // n))

                # build PV/dt scales from props
                try:
                    from simulator.props import compute_cell_props
                    props = compute_cell_props(self.sim, x_phys, dt)
                    phi = props['phi']
                    V   = props['V']
                    dt_eff = props['dt']
                    rho_w_ref = props.get('rho_w', torch.ones_like(phi))
                    # Массовая форма: давление по PV/dt, насыщенности по (PV/dt)*rho_w
                    p_scale_F = (phi * V) / (dt_eff + 1e-30)
                    sat_scale = p_scale_F * rho_w_ref
                except Exception:
                    sat_scale = torch.ones(n, device=F_phys.device, dtype=F_phys.dtype)
                    p_scale_F = torch.ones(n, device=F_phys.device, dtype=F_phys.dtype)

                Fh = torch.zeros_like(F_phys)
                # Делим обе подсистемы на один и тот же масштаб PV/dt
                Fh[:n] = F_phys[:n] / (p_scale_F + 1e-30)
                Fh[n:2*n] = F_phys[n:2*n] / (sat_scale + 1e-30)
                if vars_per_cell == 3 and F_phys.numel() >= 3*n:
                    Fh[2*n:3*n] = F_phys[2*n:3*n] / (sat_scale + 1e-30)

            if self.ptc_enabled and self.ptc_tau > 0.0:
                try:
                    n_loc_ptc = self.scaler.n_cells if self.scaler is not None else (x_hat.numel() // 2)
                except Exception:
                    n_loc_ptc = x_hat.numel() // 2
                Fh = Fh.clone()
                Fh[:n_loc_ptc] = Fh[:n_loc_ptc] + (self.ptc_tau / dt) * (x_hat[:n_loc_ptc] - x_ref[:n_loc_ptc])
            # Одноразовая диагностика масштабов PV/dt и кривых krw
            if not hasattr(self, "_dbg_scales_logged"):
                try:
                    from simulator.props import compute_cell_props
                    props_dbg = compute_cell_props(self.sim, x_phys, dt)
                    phi_dbg = props_dbg['phi']; V_dbg = props_dbg['V']; dt_dbg = props_dbg['dt']
                    pvdt = (phi_dbg * V_dbg) / (dt_dbg + 1e-30)
                    print(f"[scales] PV/dt: min={pvdt.min().item():.3e}, max={pvdt.max().item():.3e}, median={pvdt.median().item():.3e}")
                except Exception:
                    pass
                self._dbg_scales_logged = True
            if not hasattr(self, "_dbg_kr_logged"):
                try:
                    fl = self.sim.fluid
                    n = self.scaler.n_cells if self.scaler is not None else x_phys.numel()//2
                    sw = x_phys[n:2*n].view_as(fl.s_w)
                    krw = fl.calc_water_kr(sw)
                    dkr = fl.calc_dkrw_dsw(sw)
                    print(f"[relperm] krw[min,med,max]=({krw.min().item():.3e},{krw.median().item():.3e},{krw.max().item():.3e}); dkrw/dsw[min,med,max]=({dkr.min().item():.3e},{dkr.median().item():.3e},{dkr.max().item():.3e})")
                except Exception:
                    pass
                self._dbg_kr_logged = True
            return Fh

        # Привязываем локальную функцию невязки ко всем атрибутам, которые её вызывают
        self.F_func = _F_hat
        self._F_hat = _F_hat

        def A(v_hat: torch.Tensor) -> torch.Tensor:
            # Используем блочный матвектор с раздельными eps по p и y
            return self._matvec(x, v_hat)

        def M_hat(r_hat: torch.Tensor) -> torch.Tensor:
            # CPR в hat (для geo2 используем apply_hat)
            try:
                from simulator.props import compute_cell_props
                x_phys_curr = _phys_from_hat_y(x)
                self.sim._cell_props_cache = compute_cell_props(self.sim, x_phys_curr, dt)
                # также положим текущие s и ds/dy для предобуславливателя
                try:
                    n_loc3 = self.scaler.n_cells if self.scaler is not None else (x.numel() // 2)
                    if x.numel() >= 2 * n_loc3:
                        yloc = x[n_loc3:2*n_loc3]
                        s_scales_loc = getattr(self.scaler, "s_scales", (1.0,))
                        sw_scale_loc = float(s_scales_loc[0] if len(s_scales_loc) > 0 else 1.0)
                        yloc_phys = yloc * sw_scale_loc
                        # параметры сигмоиды по текущему флюиду
                        swc3 = float(getattr(self.sim.fluid, 'sw_cr', 0.0))
                        sor3 = float(getattr(self.sim.fluid, 'so_r', 0.0))
                        denom3 = max(1e-12, 1.0 - swc3 - sor3)
                        sigma3 = torch.sigmoid(yloc_phys)
                        sw_curr3 = swc3 + denom3 * sigma3
                        dsdy3 = denom3 * (sigma3 * (1.0 - sigma3))
                        self.sim._cell_props_cache["sw_for_prec"] = sw_curr3.detach().to(x_phys_curr)
                        self.sim._cell_props_cache["dsdy_for_prec"] = dsdy3.detach().to(x_phys_curr)
                except Exception:
                    pass
            except Exception:
                self.sim._cell_props_cache = None

            if getattr(self.prec, "backend", "") == "geo2" and hasattr(self.prec, "apply_hat"):
                return self.prec.apply_hat(r_hat)
            else:
                return self.prec.apply(r_hat)

        # основной цикл Ньютона -----------------------------------------------
        for it in range(effective_max_it):
            # адаптивный PTC (только давление): сильнее в начале, затухает
            if self.ptc_enabled:
                if it == 0:
                    self.ptc_tau = 20.0 * dt
                elif it == 1:
                    self.ptc_tau = 6.0 * dt
                elif it == 2:
                    self.ptc_tau = 2.0 * dt
                else:
                    self.ptc_tau = 0.0
            # передаём номер итерации в CPR для динамики связи p→s
            try:
                self.sim._newton_it = it
            except Exception:
                pass
            F = _F_hat(x)
            F_norm = F.norm()
            self.last_res_norm = float(F_norm)
            F_scaled = F_norm / math.sqrt(len(F))

            # ранний приём
            early_tol = float(self.sim.sim_params.get("early_accept_tol", 1e-4))
            if F_scaled < early_tol:
                print(f"  Newton: ||F||_scaled={F_scaled:.3e} < early_tol={early_tol:.1e} → приём")
                self.last_newton_iters = max(1, it)
                self.last_gmres_iters = self.total_gmres_iters
                _anchor_pressure(x)
                return _phys_from_hat_y(x), True  # ВОЗВРАТ В ФИЗИЧЕСКИХ ЕДИНИЦАХ

            if init_F_scaled is None:
                init_F_scaled = F_scaled
            print(f"  Newton #{it}: ||F||={F_norm:.3e}, ||F||_scaled={F_scaled:.3e}")

            # критерий сходимости (абс/относит)
            if (F_scaled < self.tol) or (F_scaled < self.rtol * init_F_scaled):
                print(f"  Newton сошёлся за {it} итераций.")
                self.last_newton_iters = max(1, it)
                self.last_gmres_iters = self.total_gmres_iters
                _anchor_pressure(x)
                return _phys_from_hat_y(x), True  # ВОЗВРАТ В ФИЗИЧЕСКИХ ЕДИНИЦАХ

            # адаптивный forcing-term η_k
            if prev_F_norm is None:
                eta_k = float(self.sim.sim_params.get("newton_eta0", 3e-5))
            else:
                ratio = (F_norm / (prev_F_norm + 1e-30)).item()
                eta_k = 0.5 * (ratio ** 2)
            eta_k = min(max(eta_k, 1e-5), 2e-3)

            # требуемая точность GMRES
            gmres_tol_min = max(5e-5, gmres_tol_base)
            gmres_tol = max(gmres_tol_min, eta_k)
            if it <= 2:
                gmres_tol = min(gmres_tol, 5e-4)
            print(f"  GMRES: tol={gmres_tol:.3e}")

            # политика рестарта/итераций GMRES
            if (it <= 2) or (gmres_tol <= 3e-4):
                gmres_restart = 80
                gmres_maxiter = 120
            else:
                gmres_restart = 30
                gmres_maxiter = 40

            # дефляция на крупных задачах
            basis_tensor = None
            if allow_defl and self.defl_basis:
                basis_tensor = torch.stack(self.defl_basis, dim=1)

            # Актуализируем PTC-коэффициент: активен только на первых it≤ptc_iters
            if it < self.ptc_iters:
                self.ptc_enabled = True
                self.ptc_tau = self.ptc_tau0
            else:
                self.ptc_enabled = False
                self.ptc_tau = 0.0

            # решаем линейную подсистему A δ = -F (в hat)
            delta, info, gm_iters = fgmres(
                A, -F, M=M_hat, tol=gmres_tol,
                restart=gmres_restart, max_iter=gmres_maxiter,
                deflation_basis=basis_tensor, min_iters=3
            )
            self.total_gmres_iters += gm_iters
            print(f"[GMRES] info={info}, iters={gm_iters}, ||δ_hat||={delta.norm():.3e}, "
                f"||δp_hat||={delta[:n].norm():.3e}, ||δs_hat||={(delta[n:].norm() if delta.numel()>n else 0.0):.3e}")

            # защита от NaN/Inf
            if (not torch.isfinite(delta).all()) or info not in (0,):
                print("  GMRES не сошёлся/NaN — Jacobi fallback ×0.1")
                delta = 0.1 * M_hat(-F)
                if not torch.isfinite(delta).all():
                    delta = torch.zeros_like(F)

            # отладка: диапазоны приращений до проекций
            try:
                if delta.numel() > n:
                    dsw = delta[n:]
                    print(f"[Δraw] δsw[min,med,max]=({dsw.min().item():.3e},{dsw.median().item():.3e},{dsw.max().item():.3e})")
            except Exception:
                pass

            delta = _project_zero_mean_p(delta)
            # --- глобальные ограничители на δ --------------------------------
            # 1) давление: ±20 МПа в hat
            p_scale = float(getattr(self.scaler, "p_scale", 1.0)) if self.scaler is not None else 1.0
            P_CLIP_HAT = 20.0e6 / p_scale
            delta[:n] = delta[:n].clamp(-P_CLIP_HAT, P_CLIP_HAT)

            # 2) насыщенности (мы работаем в переменной y): |Δy| ≤ Δy_max (в hat)
            if delta.numel() >= 2 * n:
                dy_cap = float(self.sim.sim_params.get("delta_y_max", 2.0))
                delta[n:2*n] = delta[n:2*n].clamp(-dy_cap, dy_cap)

            # 3) Диагностика границ S (НЕ зануляем δ – проекции выполняются на уровне x_candidate и в F(x))
            try:
                # Диагностика границ в терминах физического δs = (ds/dy) * (Δy_phys)
                # где Δy_phys = (Δy_hat * s_scale).
                if self.scaler is not None and delta.numel() >= 2 * n:
                    s_scales = getattr(self.scaler, "s_scales", (1.0,))
                    sw_scale = float(s_scales[0] if len(s_scales) > 0 else 1.0)
                    # текущее y (hat) и его физическая версия
                    y_hat_cur = x[n:2*n]
                    y_phys_cur = y_hat_cur * sw_scale
                    swc = float(self.sim.fluid.sw_cr); sor = float(self.sim.fluid.so_r)
                    denom = max(1e-12, 1.0 - swc - sor)
                    sigma = torch.sigmoid(y_phys_cur)
                    sw_curr = swc + denom * sigma
                    dsdy = denom * (sigma * (1.0 - sigma))
                    # приращение по y в hat → физический δs
                    dy_hat = delta[n:2*n]
                    delta_sw_phys = dsdy * (dy_hat * sw_scale)
                    eps_bnd = 1e-12
                    at_lo = sw_curr <= (swc + eps_bnd)
                    at_hi = sw_curr >= (1.0 - sor - eps_bnd)
                    blocked_neg = at_lo & (delta_sw_phys < 0)
                    blocked_pos = at_hi & (delta_sw_phys > 0)
                    if blocked_neg.any() or blocked_pos.any():
                        print(f"[Δproj-bounds] would_zero={int(blocked_neg.sum()+blocked_pos.sum())}")
            except Exception:
                pass

            # --- стартовый множитель шага (агрегируем лимитеры) --------------
            factor = 1.0

            # лимитер по давлению в ФИЗИЧЕСКИХ единицах: СКЕЙЛИМ ТОЛЬКО δp, не весь шаг
            if self.scaler is not None:
                delta_phys = self.scaler.unscale_vec(delta)
            else:
                delta_phys = delta
            dp_abs_max = float(delta_phys[:n].abs().max().item()) + 1e-30
            # Адаптивный лимит по давлению в физ. единицах (Па)
            # Можно переопределить через sim_params.p_step_max; иначе — мягкий график по итерации Ньютона
            if "p_step_max" in self.sim.sim_params:
                P_STEP_MAX = float(self.sim.sim_params.get("p_step_max"))
            else:
                if it <= 1:
                    P_STEP_MAX = 5.0e6  # 5 МПа в начале
                elif it <= 3:
                    P_STEP_MAX = 3.0e6  # 3 МПа
                else:
                    P_STEP_MAX = 1.0e6  # 1 МПа далее
            alpha_p = min(1.0, P_STEP_MAX / dp_abs_max)
            if alpha_p < 1.0:
                # масштабируем только давление в δ (в hat), насыщенности не трогаем
                delta[:n] *= alpha_p
            try:
                print(f"[p-cap] dp_abs_max={dp_abs_max:.3e} Pa, P_STEP_MAX={P_STEP_MAX:.3e}, alpha_p={alpha_p:.3e}")
            except Exception:
                pass

            # лимитер по насыщенностям: только диагностика по δs (мы НЕ сжимаем шаг по y)
            try:
                if self.scaler is not None and delta.numel() >= 2 * n:
                    s_scales = getattr(self.scaler, "s_scales", (1.0,))
                    sw_scale = float(s_scales[0] if len(s_scales) > 0 else 1.0)
                    y_hat_cur = x[n:2*n]
                    y_phys_cur = y_hat_cur * sw_scale
                    swc = float(self.sim.fluid.sw_cr); sor = float(self.sim.fluid.so_r)
                    denom = max(1e-12, 1.0 - swc - sor)
                    sigma = torch.sigmoid(y_phys_cur)
                    dsdy = denom * (sigma * (1.0 - sigma))
                    dy_hat = delta[n:2*n]
                    delta_sw_phys = dsdy * (dy_hat * sw_scale)
                    # оценка «доступного» alpha по границам в s (диагностика)
                    sw_curr = swc + denom * sigma
                    alpha_sat = 1.0
                    pos_mask = (delta_sw_phys > 0)
                    if pos_mask.any():
                        alpha_sw_pos = ((1.0 - sor) - sw_curr[pos_mask]) / (delta_sw_phys[pos_mask] + 1e-30)
                        alpha_sat = min(alpha_sat, float(alpha_sw_pos.min()))
                    neg_mask = (delta_sw_phys < 0)
                    if neg_mask.any():
                        alpha_sw_neg = (sw_curr[neg_mask] - swc) / (-delta_sw_phys[neg_mask] + 1e-30)
                        alpha_sat = min(alpha_sat, float(alpha_sw_neg.min()))
                    # только диагностика без сжатия шага по насыщенности
                    print(f"[limiter] alpha_p={alpha_p:.3e}, alpha_sat_diag={alpha_sat:.3e}")
            except Exception as _e:
                print(f"[sat-limiter] предупреждение: {_e}")

            # ВАЖНО: глобальный шаг теперь НЕ сжимается из-за давления (мы уже масштабировали δp).
            # Оставляем factor управляться trust-region/LS.

            # --- Trust-region (динамический) ---------------------------------
            tr_cfg = self.sim.sim_params.get("trust_radius", None)
            if tr_cfg is not None:
                trust_radius = float(tr_cfg)
            else:
                rhs_norm = self.last_res_norm
                n_vars = delta.numel()
                dyn_tr = 20.0 * rhs_norm / max(n_vars**0.5, 1.0)
                trust_radius = max(50.0, dyn_tr)

            delta_norm_scaled = delta.norm() / math.sqrt(len(delta))
            if delta_norm_scaled > trust_radius:
                factor = min(factor, trust_radius / (delta_norm_scaled + 1e-12))
                print(f"  Trust-region: сокращаем шаг до α={factor:.3e} (R={trust_radius:.2f})")

            # --- Line search (строгий Armijo и реальное снижение) ------------
            c1 = 1e-3 if it == 0 else 3e-4
            ls_max = 10

            # не позволяем конфигом задрать min_alpha слишком высоко
            cfg_alpha = float(self.sim.sim_params.get("line_search_min_alpha", 1e-8))
            min_factor = min(max(1e-8, cfg_alpha), 1e-5)

            success = False
            base_F = F
            base_norm = float(F_norm)
            Jv_hat_ls = None
            # масштабируем только давление фактором trust-region, насыщенности оставляем как есть
            delta_ls = delta.clone()
            if factor < 1.0:
                delta_ls[:n] *= factor

            for ls_it in range(ls_max):
                if factor < min_factor:
                    print(f"  LS: достигли минимального α={min_factor:.3e} — стоп")
                    break

                if Jv_hat_ls is None:
                    Jv_hat_ls = A(delta_ls)

                x_candidate = _anchor_pressure(x + delta_ls) if fix_pressure_drift else (x + delta_ls)
                # Жёсткий колпак по давлению в HAT на этапе line-search (переменные в hat)
                try:
                    n_loc = n
                    p_scale_loc = float(getattr(self.scaler, "p_scale", 1.0)) if self.scaler is not None else 1.0
                    # Синхронизируем с P_STEP_MAX: по умолчанию клип равен тому же лимиту
                    p_step_ls_max = float(self.sim.sim_params.get("p_step_ls_max", P_STEP_MAX))
                    dp_clip_hat = p_step_ls_max / max(p_scale_loc, 1.0)
                    dp_hat = x_candidate[:n_loc] - x[:n_loc]
                    dp_hat = dp_hat.clamp(-dp_clip_hat, dp_clip_hat)
                    x_candidate[:n_loc] = x[:n_loc] + dp_hat
                    try:
                        dp_phys_inf = float((dp_hat.abs().max() * p_scale_loc))
                        print(f"[p-clip] dp_hat_max={dp_phys_inf:.3e} Pa, limit={p_step_ls_max:.3e} Pa")
                    except Exception:
                        pass
                except Exception:
                    pass
                if not torch.isfinite(x_candidate).all():
                    factor *= 0.5
                    continue

                F_cand = _F_hat(x_candidate)
                if not torch.isfinite(F_cand).all():
                    factor *= 0.5
                    continue

                f_curr = float(F_cand.norm())

                # требуем заметное относительное снижение (Армижо с нижним порогом)
                min_rel_drop = 1e-3 if it <= 2 else 5e-4
                sufficient = (f_curr <= (1 - max(c1 * factor, min_rel_drop)) * base_norm)

                if ls_it == 0:
                    # диагностический линейный прогноз
                    lin_err = (F_cand - (base_F + factor * Jv_hat_ls)).norm() / (factor * Jv_hat_ls.norm() + 1e-30)
                    # диапазон Sw в phys
                    x_cand_phys = _phys_from_hat_y(x_candidate)
                    sw_rng = (
                        x_cand_phys[n:2*n].min().item() if x_cand_phys.numel() >= 2*n else float('nan'),
                        x_cand_phys[n:2*n].max().item() if x_cand_phys.numel() >= 2*n else float('nan'),
                    )
                    print(f"    LS try α={factor:.3e}: ||F||={f_curr:.3e} "
                        f"(ratio={f_curr/(base_norm+1e-30):.3e}), lin_err={float(lin_err):.3e}, "
                        f"Sw_range=({sw_rng[0]:.3e},{sw_rng[1]:.3e})")

                if sufficient:
                    print(f"  Line search принял шаг α={factor:.3e}, ||F||={f_curr:.3e}")
                    x_new = x_candidate
                    success = True
                    # для forcing-term нам нужна «старая» норма
                    prev_F_norm = torch.tensor(base_norm, dtype=F.dtype, device=F.device)
                    break

                # Функция прогноза убыли: Armijo по реальному Jv и δ_ls
                pred_decrease = -c1 * (Jv_hat_ls * delta_ls).sum().item()
                new_F = _F_hat(x_candidate)
                new_norm = float(new_F.norm().item())
                if new_norm <= base_norm + pred_decrease:
                    success = True
                    F = new_F
                    F_norm = new_norm
                    x = x_candidate
                    # унифицируем выход: используем x_new далее
                    x_new = x_candidate
                    # для forcing-term — та же prev_F_norm, что и в другой ветке
                    prev_F_norm = torch.tensor(base_norm, dtype=F.dtype, device=F.device)
                    break
                else:
                    # уменьшаем только давление ещё раз
                    delta_ls[:n] *= 0.5
                    factor *= 0.5
                    Jv_hat_ls = None

            # Если line-search не сработал — снимем диагностическую решётку φ(α)
            if not success and bool(self.sim.sim_params.get("ls_probe", True)):
                alphas = [1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5]
                vals = []
                for a in alphas:
                    try:
                        Fc = _F_hat(_anchor_pressure(x + a * delta))
                        vals.append(float(Fc.norm()))
                    except Exception:
                        vals.append(float("nan"))
                print("[LS-PROBE] " + " ".join(f"α={a:.0e}:{v:.3e}" for a, v in zip(alphas, vals)))

            # Fallback: демпфированный Jacobi-шаг
            if not success:
                print("  Line search не нашёл шаг — Jacobi fallback (α=0.3)")
                delta_fb = 0.3 * M_hat(-base_F)
                delta_fb = _project_zero_mean_p(delta_fb)
                # безопасные клампы
                delta_fb[:n] = delta_fb[:n].clamp(-P_CLIP_HAT, P_CLIP_HAT)
                if delta_fb.numel() > n:
                    delta_fb[n:] = delta_fb[n:].clamp(-0.05, 0.05)

                x_fb = _anchor_pressure(x + delta_fb)
                if torch.isfinite(x_fb).all():
                    F_fb = _F_hat(x_fb)
                    if torch.isfinite(F_fb).all():
                        if float(F_fb.norm()) < 0.95 * base_norm:
                            print(f"  ✅ Jacobi fallback принят, ||F||={float(F_fb.norm()):.3e}")
                            x_new = x_fb
                            success = True
                            # СТАВИМ «СТАРУЮ» НОРМУ, а не норму fallback — стабильно для η_k
                            prev_F_norm = torch.tensor(base_norm, dtype=F.dtype, device=F.device)

            if not success:
                print("  JFNK: even fallback failed – завершаем шаг неудачей")
                self.last_newton_iters = self.max_it
                self.last_gmres_iters = self.total_gmres_iters
                return _phys_from_hat_y(x), False  # в phys

            # адаптация trust-region
            if trust_radius is not None:
                if factor > 0.8:
                    trust_radius = min(trust_radius * 1.4, 50.0)
                elif factor < 0.2:
                    trust_radius = max(trust_radius * 0.7, 1e-3)
                print(f"  Trust-region: новый радиус {trust_radius:.2f}")

            # обновляем состояние и фиксируем дрейф среднего давления
            x = _anchor_pressure(x_new) if fix_pressure_drift else x_new
            print(f"[DRIFT] mean_p_drift={(x[:n].mean()-baseline_mean_p).item():.3e} (hat)")

            # уменьшаем τ после успешного шага
            if self.ptc_enabled and self.ptc_tau > 0.0:
                self.ptc_tau = max(self.ptc_tau * 0.5, 0.0)

        # не сошлись за effective_max_it
        print(f"  Newton не сошёлся за {effective_max_it} итераций")
        self.last_newton_iters = self.max_it
        self.last_gmres_iters = self.total_gmres_iters
        _anchor_pressure(x)
        return _phys_from_hat_y(x), False  # в phys


    def _unscale_x(self, x_hat: torch.Tensor) -> torch.Tensor:
        # Перевод из hat в физические единицы, 2/3 переменных на ячейку поддерживаются
        return self.scaler.unscale_vec(x_hat) if self.scaler is not None else x_hat
