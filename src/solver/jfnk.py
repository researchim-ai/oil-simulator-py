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
        # Предпочитаем явный параметр JFNK, если он задан, иначе используем общий GMRES-лимит
        gmres_max_iter  = sim_params.get("jfnk_max_iter", sim_params.get("gmres_max_iter", 60))

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
        TARGET_DS   = 1e-3     # унифицированный целевой физический шаг по насыщенности
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
        TARGET_DS = 1e-3
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

                        # стабилизация масштаба: не более ×2 за итерацию
                        s_scale_sw_limited = min(s_scale_sw, max(0.5 * old_sw_scale, min(2.0 * old_sw_scale, s_scale_sw)))
                        self.scaler.s_scales[0] = s_scale_sw_limited
                        self.scaler.inv_s_scales[0] = 1.0 / s_scale_sw_limited
                        # пересоберём комбинированные массивы
                        self.scaler.scale = [self.scaler.inv_p_scale] + list(self.scaler.inv_s_scales)
                        self.scaler.inv_scale = [self.scaler.p_scale] + list(self.scaler.s_scales)
                        print(f"[scaler] updated s_scale_sw={s_scale_sw_limited:.3e} from median(ds/dy)={med_dsdy:.3e}")
        except Exception:
            pass

        # Дрейф среднего давления: по умолчанию не фиксируем, только если явно задано в конфиге
        baseline_mean_p = x[:n].mean().clone()
        fix_pressure_drift = bool(self.sim.sim_params.get("fix_pressure_drift", False))

        def _anchor_pressure(x_hat: torch.Tensor):
            # Отказ от принудительного якорения среднего давления
            return x_hat

        def _project_zero_mean_p(v_hat: torch.Tensor):
            # Не проецируем mean(δp) → позволяем среднему давлению меняться
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
        x_ref = x0.clone()  # hat

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

        # локальные помощники -------------------------------------------------
        def _F_hat(x_hat: torch.Tensor) -> torch.Tensor:
            # Никакого дополнительного якорения: невязку считаем на реальной точке.
            # ВАЖНО: строим физическое состояние из представления y для насыщенностей
            try:
                x_phys = _phys_from_hat_y(x_hat)
            except Exception:
                x_phys = self._unscale_x(x_hat) if self.scaler is not None else x_hat
            F_phys = self.sim._fi_residual_vec(x_phys, dt)
            # Физические метрики масс-баланса (давленческий блок) для критериев приёма
            try:
                n_loc_mb = self.scaler.n_cells if self.scaler is not None else (x_phys.numel() // 2)
                from simulator.props import compute_cell_props
                props_mb = compute_cell_props(self.sim, x_phys, dt)
                phi_mb = props_mb['phi']; V_mb = props_mb['V']; dt_mb = props_mb['dt']
                pvdt_mb = (phi_mb * V_mb) / (dt_mb + 1e-30)
                mb_cell = (F_phys[:n_loc_mb].abs()) / (pvdt_mb.abs() + 1e-30)
                # Главная метрика как в логах: L∞ по ячейкам
                self._last_mb_max = float(mb_cell.max().item())
                # Дополнительно средняя (для информации)
                self._last_mb_l1 = float(mb_cell.mean().item())
            except Exception:
                self._last_mb_max = None
                self._last_mb_l1 = None

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
            # КРИТИЧЕСКАЯ ДИАГНОСТИКА: проверим масштабы один раз
            if not hasattr(self, "_dbg_scales_logged"):
                try:
                    from simulator.props import compute_cell_props
                    props_dbg = compute_cell_props(self.sim, x_phys, dt)
                    phi_dbg = props_dbg['phi']; V_dbg = props_dbg['V']; dt_dbg = props_dbg['dt']
                    rho_dbg = props_dbg.get('rho_w', torch.ones_like(phi_dbg))
                    pvdt = (phi_dbg * V_dbg) / (dt_dbg + 1e-30)
                    sat_sc = pvdt * rho_dbg
                    
                    print(f"\n{'='*70}")
                    print(f"[МАСШТАБЫ] Анализ согласованности масштабирования")
                    print(f"{'='*70}")
                    print(f"[Переменные x] p_scale = {self.scaler.p_scale:.3e} Па, s_scale = {self.scaler.s_scales}")
                    print(f"[Невязки F] PV/dt = {pvdt.median().item():.3e}, sat_scale = {sat_sc.median().item():.3e}")
                    print(f"[Якобиан A_sp] масштаб = (p_scale) / (sat_scale_F) = {self.scaler.p_scale / sat_sc.median().item():.3e}")
                    print(f"  ⚠️  ЕСЛИ это число >> 1, то coupling блок A_sp раздут!")
                    print(f"\n[ФИЗИЧЕСКИЙ СМЫСЛ]")
                    print(f"  F_p: баланс объёма [м³/с], масштаб PV/dt")
                    print(f"  F_s: баланс массы [кг/с], масштаб PV/dt·ρ")
                    print(f"  p: давление [Па], масштаб p_scale")
                    print(f"  S: насыщенность [безразм], масштаб s_scale=1")
                    print(f"\n[ЯКОБИАН]")
                    print(f"  A_pp ~ ∂F_p/∂p ~ (∂F_p [м³/с]) / (∂p [Па]) / масштабы")
                    print(f"  A_sp ~ ∂F_s/∂p ~ (∂F_s [кг/с]) / (∂p [Па]) / масштабы")
                    print(f"  A_sp_hat = A_sp_phys * (p_scale / sat_scale)")
                    print(f"           = A_sp_phys * {self.scaler.p_scale / sat_sc.median().item():.3e}")
                    print(f"{'='*70}\n")
                except Exception as e:
                    print(f"[scales] ошибка диагностики: {e}")
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
            """
            Правый предобуславливатель для FGMRES в HAT-пространстве.
            ИСПРАВЛЕНО: работает строго в hat→hat, без повторного скейлинга.
            """
            # Обновляем кеш свойств ячеек для CPR
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

            # Применяем CPR: hat → hat (без повторного скейлинга!)
            n = int(self.scaler.n_cells)
            z = torch.zeros_like(r_hat)
            
            # Stage-1: pressure preconditioner (AMG) — hat→hat
            if getattr(self.prec, "backend", "") == "geo2" and hasattr(self.prec, "apply_hat"):
                z_full = self.prec.apply_hat(r_hat)
                z = z_full  # уже содержит и давление, и насыщенности
            else:
                # Fallback для других бэкендов
                z = self.prec.apply(r_hat)
            
            # Диагностика эффективности CPR (один раз на итерацию Ньютона)
            if getattr(self, "_dbg_cpr_iter", -1) != it:
                rp = r_hat[:n]
                zp = z[:n]
                # КРИТИЧЕСКАЯ ДИАГНОСТИКА saturations
                if r_hat.numel() >= 2*n:
                    rsw = r_hat[n:2*n]
                    zsw = z[n:2*n]
                    rsw_norm = rsw.norm().item()
                    zsw_norm = zsw.norm().item()
                    zsw_max = zsw.abs().max().item()
                    prec_ratio_sw = zsw_norm / (rsw_norm + 1e-30)
                    print(f"[M_hat saturations] ||r_sw||={rsw_norm:.3e}, ||z_sw||={zsw_norm:.3e}, max={zsw_max:.3e}")
                    print(f"  [эффективность] ||z_sw||/||r_sw|| = {prec_ratio_sw:.3e} (должно быть > 0.1)")
                    if prec_ratio_sw < 0.01:
                        print(f"  ⚠️  КРИТИЧНО: прекондиционер saturations почти не работает!")
                
                # ПРАВИЛЬНАЯ ПРОВЕРКА: используем ПОЛНЫЙ вектор z (с saturations!)
                Az_full = self._matvec(x, z)
                Az_p = Az_full[:n]  # блок давления
                
                # КРИТИЧЕСКАЯ ДИАГНОСТИКА: декомпозиция coupling
                # Проверим, КАКОЙ блок Якобиана создает огромную норму
                zsw = z[n:2*n] if z.numel() >= 2*n else torch.zeros(n, device=z.device, dtype=z.dtype)
                
                # 1) Только A_pp·z_p (из GeoSolver — изолированный pressure блок)
                try:
                    if hasattr(self.prec, 'solver') and hasattr(self.prec.solver, '_apply_A'):
                        Azp_geo = self.prec.solver._apply_A(0, zp.to(torch.float64)).to(zp.dtype)
                        print(f"  [COUPLING-1] A_pp·z_p (isolate): ||·||={Azp_geo.norm().item():.3e}, cosθ={torch.dot(Azp_geo, rp).item()/(Azp_geo.norm()*rp.norm() + 1e-30):.3f}")
                except:
                    pass
                
                # 2) Полный A·[z_p, z_sw] — все блоки
                Az_full = self._matvec(x, z)
                Az_full_p = Az_full[:n]  # pressure результат
                Az_full_s = Az_full[n:2*n] if Az_full.numel() >= 2*n else torch.zeros(n, device=Az_full.device, dtype=Az_full.dtype)
                
                # 3) A·[z_p, 0] — только давление (покажет A_sp·z_p)
                z_only_p = torch.zeros_like(z)
                z_only_p[:n] = zp
                Az_only_p = self._matvec(x, z_only_p)
                Az_only_p_s = Az_only_p[n:2*n] if Az_only_p.numel() >= 2*n else torch.zeros(n, device=Az_only_p.device, dtype=Az_only_p.dtype)
                
                print(f"  [COUPLING-2] A·[z_p,0]_saturation: ||A_sp·z_p||={Az_only_p_s.norm().item():.3e} ← coupling влияние!")
                print(f"  [COUPLING-3] A·[z_p,z_sw]_pressure: ||·||={Az_full_p.norm().item():.3e}, cosθ={torch.dot(Az_full_p, rp).item()/(Az_full_p.norm()*rp.norm() + 1e-30):.3f}")
                print(f"  [COUPLING-4] A·[z_p,z_sw]_saturation: ||·||={Az_full_s.norm().item():.3e}")
                print(f"  [COUPLING-5] Нормы: ||z_p||={zp.norm().item():.3e}, ||z_sw||={zsw.norm().item():.3e}, ||r_p||={rp.norm().item():.3e}, ||r_sw||={r_hat[n:2*n].norm().item():.3e}")
                
                # Итоговая эффективность
                r_after = r_hat - Az_full
                rho_cpr = r_after.norm().item() / (r_hat.norm().item() + 1e-30)
                print(f"  [COUPLING ИТОГО] ρ_CPR={rho_cpr:.3e} (хорошо если < 0.5)")
                self._dbg_cpr_iter = it
            
            return z

        # основной цикл Ньютона -----------------------------------------------
        for it in range(effective_max_it):
            # цветовой префикс
            GREEN = "\x1b[32m" if bool(self.sim.sim_params.get('color_logs', True)) else ""
            RED   = "\x1b[31m" if bool(self.sim.sim_params.get('color_logs', True)) else ""
            YEL   = "\x1b[33m" if bool(self.sim.sim_params.get('color_logs', True)) else ""
            RESET = "\x1b[0m"  if bool(self.sim.sim_params.get('color_logs', True)) else ""
            # PTC: задаём τ_k ОДИН РАЗ на итерацию Ньютона и далее не переопределяем
            if self.ptc_enabled:
                if it == 0:
                    self.ptc_tau = 20.0 * dt
                elif it == 1:
                    self.ptc_tau = 6.0 * dt
                elif it == 2:
                    self.ptc_tau = 2.0 * dt
                else:
                    self.ptc_tau = 0.0
                # масштабируем через конфиг при необходимости
                try:
                    scale_tau = float(self.sim.sim_params.get("ptc_tau_scale", 1.0))
                except Exception:
                    scale_tau = 1.0
                self.ptc_tau *= scale_tau
            # передаём номер итерации в CPR для динамики связи p→s
            try:
                self.sim._newton_it = it
            except Exception:
                pass
            F = _F_hat(x)
            F_norm = F.norm()
            self.last_res_norm = float(F_norm)
            F_scaled = F_norm / math.sqrt(len(F))
            # сохраняем последнюю масштабированную норму для унифицированного репорта
            self.last_res_scaled = float(F_scaled)

            # ранний приём (с учётом физического MB)
            early_tol = float(self.sim.sim_params.get("early_accept_tol", 1e-4))
            mb_tol_accept = float(self.sim.sim_params.get("mb_tol", 1e-4))
            mb_max = getattr(self, "_last_mb_max", None)
            if (F_scaled < early_tol) and (mb_max is not None and mb_max < mb_tol_accept):
                print(f"  Newton: ||F||_scaled={F_scaled:.3e} < early_tol={early_tol:.1e}, MBmax={mb_max:.3e} → приём")
                self.last_newton_iters = max(1, it)
                self.last_gmres_iters = self.total_gmres_iters
                _anchor_pressure(x)
                return _phys_from_hat_y(x), True  # ВОЗВРАТ В ФИЗИЧЕСКИХ ЕДИНИЦАХ

            if init_F_scaled is None:
                init_F_scaled = F_scaled
                # сохраняем начальную масштабированную норму для отчёта rel_tol
                self.init_res_scaled = float(init_F_scaled)
            mb_tol = float(self.sim.sim_params.get("mb_tol", 1e-4))
            mb_max = getattr(self, "_last_mb_max", float('nan'))
            mb_l1  = getattr(self, "_last_mb_l1",  float('nan'))
            print(f"  Newton #{it}: ||F||={F_norm:.3e}, ||F||_scaled={F_scaled:.3e}, MB[max]={mb_max:.3e}, MB[mean]={mb_l1:.3e} (tol={mb_tol:.1e})")
            try:
                self.sim.log_json({
                    "event": "newton_iter",
                    "iter": int(it),
                    "F_norm": float(F_norm),
                    "F_scaled": float(F_scaled),
                })
            except Exception:
                pass

            # критерий сходимости (абс/относит) + физический MB
            mb_tol = float(self.sim.sim_params.get("mb_tol", 1e-4))
            mb_max = getattr(self, "_last_mb_max", None)
            if ((F_scaled < self.tol) or (F_scaled < self.rtol * init_F_scaled)) and (mb_max is not None and mb_max < mb_tol):
                print(f"  Newton сошёлся за {it} итераций. MBmax={mb_max:.3e} < {mb_tol:.1e}")
                self.last_newton_iters = max(1, it)
                self.last_gmres_iters = self.total_gmres_iters
                _anchor_pressure(x)
                return _phys_from_hat_y(x), True  # ВОЗВРАТ В ФИЗИЧЕСКИХ ЕДИНИЦАХ

            # адаптивный forcing-term η_k
            # ИСПРАВЛЕНО: улучшенный Eisenstat-Walker II форсинг
            if prev_F_norm is None:
                eta_k = float(self.sim.sim_params.get("newton_eta0", 0.25))
            else:
                ratio = (F_norm / (prev_F_norm + 1e-30)).item()
                # Eisenstat–Walker II: η_k = c * (||F_k||/||F_{k-1}||)^α
                alpha = float(self.sim.sim_params.get("ew_alpha", 1.5))
                c = float(self.sim.sim_params.get("ew_c", 0.9))
                eta_k = c * (ratio ** alpha)
            # жёсткие границы (более мягкие для FGMRES)
            eta_min = float(self.sim.sim_params.get("newton_eta_min", 1e-5))
            eta_max = float(self.sim.sim_params.get("newton_eta_max", 0.25))
            eta_k = min(max(eta_k, eta_min), eta_max)

            # требуемая точность GMRES
            gmres_tol_min = max(5e-5, gmres_tol_base)
            gmres_tol = max(gmres_tol_min, eta_k)
            if it <= 2:
                gmres_tol = min(gmres_tol, 5e-4)
            print(f"  [JFNK] GMRES tol={gmres_tol:.3e}, eta_k={eta_k:.3e}, min={gmres_tol_min:.3e}")

            # ИСПРАВЛЕНО: увеличен restart для FGMRES (лучше работает с переменным предобуславливателем)
            budget_max = int(self.sim.sim_params.get("jfnk_max_iter", self.sim.sim_params.get("gmres_max_iter", 400)))
            if (it <= 2):
                gmres_restart = min(120, budget_max)
                gmres_maxiter = min(200, budget_max)
            else:
                if gmres_tol <= 3e-4:
                    gmres_restart = min(150, budget_max)
                    gmres_maxiter = min(300, budget_max)
                elif gmres_tol <= 1e-3:
                    gmres_restart = min(120, budget_max)
                    gmres_maxiter = min(250, budget_max)
                else:
                    gmres_restart = min(100, budget_max)
                    gmres_maxiter = min(200, budget_max)

            # дефляция на крупных задачах
            basis_tensor = None
            if allow_defl and self.defl_basis:
                basis_tensor = torch.stack(self.defl_basis, dim=1)

            # Не переопределяем self.ptc_tau здесь — он уже выбран выше на текущую итерацию
            # и должен совпадать между RHS (F) и матвектором (J·v)

            # решаем линейную подсистему A δ = -F (в hat)
            delta, info, gm_iters = fgmres(
                A, -F, M=M_hat, tol=gmres_tol,
                restart=gmres_restart, max_iter=gmres_maxiter,
                deflation_basis=basis_tensor, min_iters=3
            )
            self.total_gmres_iters += gm_iters
            print((YEL if info != 0 else GREEN) + f"[GMRES] info={info}, iters={gm_iters}, ||δ_hat||={delta.norm():.3e}, "
                f"||δp_hat||={delta[:n].norm():.3e}, ||δs_hat||={(delta[n:].norm() if delta.numel()>n else 0.0):.3e}" + RESET)
            try:
                self.sim.log_json({
                    "event": "gmres_done",
                    "iter": int(it),
                    "gmres_info": int(info),
                    "gmres_iters": int(gm_iters),
                    "delta_norm": float(delta.norm()),
                })
            except Exception:
                pass

            # защита от NaN/Inf + одна повторная попытка GMRES с ослабленными параметрами
            if (not torch.isfinite(delta).all()) or info not in (0,):
                print(YEL + "[GMRES] повторная попытка с ослабленным tol, без превышения бюджета" + RESET)
                retry_restart = min(int(max(80, gmres_restart*1.25)), budget_max)
                retry_maxiter = budget_max - int(self.total_gmres_iters)
                retry_maxiter = max(0, min(retry_maxiter, budget_max))
                delta, info2, gm_iters2 = fgmres(
                    A, -F, M=M_hat, tol=max(gmres_tol*3.0, 1e-8),
                    restart=retry_restart, max_iter=retry_maxiter,
                    deflation_basis=basis_tensor, min_iters=3
                )
                self.total_gmres_iters += gm_iters2
                print((YEL if info2 != 0 else GREEN) + f"[GMRES-retry] info={info2}, iters={gm_iters2}, ||δ_hat||={delta.norm():.3e}" + RESET)
                if (not torch.isfinite(delta).all()) or info2 not in (0,):
                    # Inexact Newton: если получили конечный δ, идём в line-search.
                    if torch.isfinite(delta).all():
                        print(YEL + "  [GMRES] используем неточный шаг (inexact), продолжим line-search" + RESET)
                    else:
                        print(RED + "  GMRES не сошёлся и δ невалиден — прекращаем" + RESET)
                        self.last_newton_iters = self.max_it
                        self.last_gmres_iters = self.total_gmres_iters
                        return _phys_from_hat_y(x), False

            # отладка: диапазоны приращений до проекций
            try:
                if delta.numel() > n:
                    dsw = delta[n:]
                    print(f"[Δraw] δsw[min,med,max]=({dsw.min().item():.3e},{dsw.median().item():.3e},{dsw.max().item():.3e})")
            except Exception:
                pass

            # ИСПРАВЛЕНО: Trust-region по HAT для защиты от перешага
            # --- глобальные ограничители на δ --------------------------------
            
            # 1) Давление: ограничиваем блок давлений в hat
            dp_inf_hat = float(delta[:n].abs().max().item())
            DP_CAP_HAT = float(self.sim.sim_params.get("dp_cap_hat", 50.0))  # безопасное: 50 в hat ~ 5e6 Па при p_scale=1e5
            if dp_inf_hat > DP_CAP_HAT:
                scale_factor = DP_CAP_HAT / (dp_inf_hat + 1e-30)
                delta[:n] *= scale_factor
                if os.environ.get("OIL_DEBUG", "0") == "1":
                    print(f"[TRUST] Pressure cap: ||δp||∞={dp_inf_hat:.3e} > {DP_CAP_HAT:.1e}, масштаб={scale_factor:.3e}")
            
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
                    P_STEP_MAX = 3.0e6  # минимум 3 МПа далее, чтобы не загонять шаги в микроскопические
            alpha_p = min(1.0, P_STEP_MAX / dp_abs_max)
            # Не масштабируем delta[:n] здесь, чтобы не ломать линейный поиск.
            # Применим ограничение давления к candidate шагу ПОСЛЕ умножения на alpha (factor)
            try:
                print(f"[p-cap] dp_abs_max={dp_abs_max:.3e} Pa, P_STEP_MAX={P_STEP_MAX:.3e}, alpha_p={alpha_p:.3e}")
                self.sim.log_json({
                    "event": "pressure_cap",
                    "iter": int(it),
                    "dp_abs_max": float(dp_abs_max),
                    "P_STEP_MAX": float(P_STEP_MAX),
                    "alpha_p": float(alpha_p),
                })
            except Exception:
                pass

            # лимитер по насыщенностям: вычисляем допустимый α по физ. границам Sw
            alpha_sat = 1.0
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
                    sw_curr = swc + denom * sigma
                    pos_mask = (delta_sw_phys > 0)
                    if pos_mask.any():
                        alpha_sw_pos = ((1.0 - sor) - sw_curr[pos_mask]) / (delta_sw_phys[pos_mask] + 1e-30)
                        alpha_sat = min(alpha_sat, float(alpha_sw_pos.min()))
                    neg_mask = (delta_sw_phys < 0)
                    if neg_mask.any():
                        alpha_sw_neg = (sw_curr[neg_mask] - swc) / (-delta_sw_phys[neg_mask] + 1e-30)
                        alpha_sat = min(alpha_sat, float(alpha_sw_neg.min()))
                    alpha_sat = max(1e-6, min(1.0, alpha_sat))
                    print(f"[limiter] alpha_p={alpha_p:.3e}, alpha_sat={alpha_sat:.3e}")
                    self.sim.log_json({
                        "event": "sat_limiter",
                        "iter": int(it),
                        "alpha_p": float(alpha_p),
                        "alpha_sat": float(alpha_sat),
                    })
            except Exception:
                pass

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

            # --- Line search (немонотонный Armijo с окном m) -----------------
            c1 = 1e-3 if it == 0 else 3e-4
            ls_max = 16
            m_hist = int(self.sim.sim_params.get("ls_nonmonotone_m", 5))
            hist = getattr(self, "_ls_hist", [])

            # не позволяем конфигом задрать min_alpha слишком высоко
            cfg_alpha = float(self.sim.sim_params.get("line_search_min_alpha", 1e-8))
            # Позволяем уходить до 1e-6 по умолчанию
            min_factor = min(max(1e-10, cfg_alpha), 1e-6)

            success = False
            base_F = F
            base_norm = float(F_norm)
            Jv_hat_ls = None
            # подготовим базовый шаг; кандидата строим внутри цикла с учётом factor/лимитеров
            delta_ls_base = delta.clone()

            for ls_it in range(ls_max):
                if factor < min_factor:
                    print(RED + f"  LS: достигли минимального α={min_factor:.3e} — стоп" + RESET)
                    break

                # базовый кандидат с учётом trust-region: масштабируем ВЕСЬ шаг
                delta_ls = delta_ls_base.clone()
                if factor < 1.0:
                    delta_ls *= factor

                # мягко соблюдаем предел по давлению: однородно сжимаем δp до p_step_ls_max
                try:
                    p_scale_loc = float(getattr(self.scaler, "p_scale", 1.0)) if self.scaler is not None else 1.0
                    p_step_ls_max = float(self.sim.sim_params.get("p_step_ls_max", P_STEP_MAX))
                    dp_clip_hat = p_step_ls_max / max(p_scale_loc, 1.0)
                    dp_max_hat = float(delta_ls[:n].abs().max().item()) + 1e-30
                    if dp_max_hat > dp_clip_hat:
                        s_dp = dp_clip_hat / dp_max_hat
                        delta_ls[:n] *= s_dp
                        factor *= s_dp
                    print(f"[p-clip] dp_hat_max={dp_max_hat*p_scale_loc:.3e} Pa, limit={p_step_ls_max:.3e} Pa")
                except Exception:
                    pass

                # соблюдаем физические границы по насыщенности (равномерный масштаб δy)
                if delta_ls.numel() >= 2 * n:
                    if 'alpha_sat' in locals() and alpha_sat < 1.0:
                        delta_ls[n:2*n] *= alpha_sat

                if Jv_hat_ls is None:
                    Jv_hat_ls = A(delta_ls)

                x_candidate = x + delta_ls
                # Жёсткий колпак по давлению в HAT на этапе line-search (после применения factor)
                try:
                    n_loc = n
                    p_scale_loc = float(getattr(self.scaler, "p_scale", 1.0)) if self.scaler is not None else 1.0
                    # Синхронизируем с P_STEP_MAX и вводим нижнюю границу на шаг по давлению
                    p_step_ls_max = float(self.sim.sim_params.get("p_step_ls_max", P_STEP_MAX))
                    dp_step_min = float(self.sim.sim_params.get("p_step_ls_min", 1.0e4))
                    p_step_ls_max = max(p_step_ls_max, dp_step_min)
                    dp_clip_hat = p_step_ls_max / max(p_scale_loc, 1.0)
                    dp_hat = x_candidate[:n_loc] - x[:n_loc]
                    dp_hat = dp_hat.clamp(-dp_clip_hat, dp_clip_hat)
                    x_candidate[:n_loc] = x[:n_loc] + dp_hat
                    try:
                        dp_phys_inf = float((dp_hat.abs().max() * p_scale_loc))
                        print(f"[p-clip] dp_hat_max={dp_phys_inf:.3e} Pa, limit={p_step_ls_max:.3e} Pa")
                        self.sim.log_json({
                            "event": "pressure_clip",
                            "iter": int(it),
                            "dp_hat_max": float(dp_phys_inf),
                            "p_step_ls_max": float(p_step_ls_max),
                        })
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
                # немонотонная ссылка: допускаем рост относительно лучшего из последних m
                F_ref = base_norm
                if m_hist > 0 and len(hist) > 0:
                    F_ref = max(hist[-m_hist:])
                sufficient = (f_curr <= (1 - max(c1 * factor, min_rel_drop)) * F_ref)

                if ls_it == 0:
                    # диагностический линейный прогноз по ФАКТИЧЕСКОМУ шагу d_eff = x_candidate - x
                    d_eff = x_candidate - x
                    Jd_eff = A(d_eff)
                    Jn = Jd_eff.norm()
                    if float(Jn) > 1e-20:
                        lin_err = (F_cand - (base_F + Jd_eff)).norm() / (Jn + 1e-30)
                    else:
                        lin_err = float('nan')
                    # диапазон Sw в phys
                    x_cand_phys = _phys_from_hat_y(x_candidate)
                    sw_rng = (
                        x_cand_phys[n:2*n].min().item() if x_cand_phys.numel() >= 2*n else float('nan'),
                        x_cand_phys[n:2*n].max().item() if x_cand_phys.numel() >= 2*n else float('nan'),
                    )
                    ratio_val = f_curr/(base_norm+1e-30)
                    if math.isfinite(float(lin_err)):
                        print(f"    LS try α={factor:.3e}: ||F||={f_curr:.3e} "
                              f"(ratio={ratio_val:.3e}), lin_err={float(lin_err):.3e}, "
                              f"Sw_range=({sw_rng[0]:.3e},{sw_rng[1]:.3e})")
                    else:
                        print(f"    LS try α={factor:.3e}: ||F||={f_curr:.3e} "
                              f"(ratio={ratio_val:.3e}), Sw_range=({sw_rng[0]:.3e},{sw_rng[1]:.3e})")
                    try:
                        payload = {
                            "event": "ls_try",
                            "iter": int(it),
                            "alpha": float(factor),
                            "F_norm": float(f_curr),
                            "ratio": float(ratio_val),
                        }
                        if math.isfinite(float(lin_err)):
                            payload["lin_err"] = float(lin_err)
                        self.sim.log_json(payload)
                    except Exception:
                        pass

                # Дополнительное «мягкое» условие: при очень малом шаге допускаем
                # просто убывание (без строгого Armijo), чтобы не застревать.
                soft_accept = (factor <= 1e-3) and (f_curr < base_norm * (1 - 1e-4))
                if sufficient or soft_accept:
                    # обновим историю
                    try:
                        hist.append(float(f_curr))
                        if len(hist) > max(1, m_hist):
                            hist.pop(0)
                        self._ls_hist = hist
                    except Exception:
                        pass
                    print(GREEN + f"  Line search принял шаг α={factor:.3e}, ||F||={f_curr:.3e}" + RESET)
                    try:
                        self.sim.log_json({
                            "event": "ls_accept",
                            "iter": int(it),
                            "alpha": float(factor),
                            "F_norm": float(f_curr),
                        })
                    except Exception:
                        pass
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
                    # стандартный бэктрекинг
                    factor *= 0.5
                    Jv_hat_ls = None

            # Если line-search не сработал — снимем диагностическую решётку φ(α)
            # и попробуем принять лучший α из решётки, если есть хоть какая-то убыв.
            if not success and bool(self.sim.sim_params.get("ls_probe", True)):
                alphas = [1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
                vals = []
                states = []
                for a in alphas:
                    try:
                        x_try = x + a * delta
                        Fc = _F_hat(x_try)
                        vals.append(float(Fc.norm()))
                        states.append((a, x_try, float(vals[-1])))
                    except Exception:
                        vals.append(float("nan"))
                        states.append((a, None, float("nan")))
                print("[LS-PROBE] " + " ".join(f"α={a:.0e}:{v:.3e}" for a, v in zip(alphas, vals)))
                # Выбираем лучший допустимый α
                best = None
                for (a, x_try, v) in states:
                    if math.isfinite(v):
                        if (best is None) or (v < best[2]):
                            best = (a, x_try, v)
                if best is not None and best[1] is not None and best[2] < base_norm * (1 - 5e-4):
                    print(GREEN + f"  LS-PROBE принял шаг α={best[0]:.3e}, ||F||={best[2]:.3e}" + RESET)
                    x_new = best[1]
                    success = True
                    prev_F_norm = torch.tensor(base_norm, dtype=F.dtype, device=F.device)

            # Без fallback: если LS не нашёл шаг — честно выходим неуспехом
            if not success:
                print(RED + "  Line search не нашёл шаг — завершаем без fallback" + RESET)
                self.last_newton_iters = self.max_it
                self.last_gmres_iters = self.total_gmres_iters
                return _phys_from_hat_y(x), False

            # адаптация trust-region
            if trust_radius is not None:
                if factor > 0.8:
                    trust_radius = min(trust_radius * 1.4, 50.0)
                elif factor < 0.2:
                    trust_radius = max(trust_radius * 0.7, 1e-3)
                print(f"  Trust-region: новый радиус {trust_radius:.2f}")

            # обновляем состояние и фиксируем дрейф среднего давления
            x = x_new
            print(f"[DRIFT] mean_p_drift={(x[:n].mean()-baseline_mean_p).item():.3e} (hat)")

            # уменьшаем τ после успешного шага
            if self.ptc_enabled and self.ptc_tau > 0.0:
                self.ptc_tau = max(self.ptc_tau * 0.5, 0.0)

        # не сошлись за effective_max_it
        print(f"  Newton не сошёлся за {effective_max_it} итераций")
        self.last_newton_iters = self.max_it
        self.last_gmres_iters = self.total_gmres_iters
        # не фиксируем среднее давление на выходе
        return _phys_from_hat_y(x), False  # в phys


    def _unscale_x(self, x_hat: torch.Tensor) -> torch.Tensor:
        # Перевод из hat в физические единицы, 2/3 переменных на ячейку поддерживаются
        return self.scaler.unscale_vec(x_hat) if self.scaler is not None else x_hat
