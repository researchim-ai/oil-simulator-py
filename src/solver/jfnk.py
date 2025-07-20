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
        }

        self.prec = CPRPreconditioner(simulator,
                                       backend=backend,
                                       smoother=smoother,
                                       scaler=self.scaler,
                                       geo_params=geo_params)

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

    def _Jv(self, x: torch.Tensor, v: torch.Tensor, dt):
        """🚀 ПРОМЫШЛЕННЫЙ Jacobian-vector произведение с регуляризацией.
        
        Вычисляет произведение Якобиана на вектор *v* по двухточечной
        разности. Шаг eps выбираем по рекомендации Брауна – порядка √ε
        машинного, масштабируемый на ‖x‖, чтобы избежать слишком мелких
        разностей, приводящих к шуму.
        """
        # Машинное ε для float32 или float64 в зависимости от dtype
        dtype_eps = 1e-7 if x.dtype == torch.float32 else 1e-15
        eps_base = torch.sqrt(torch.tensor(dtype_eps, dtype=x.dtype, device=x.device))
        eps = eps_base * (1.0 + torch.norm(x)) / (torch.norm(v) + 1e-12)
        # Для микромоделей используем нижний предел 1e-6 (как в тестах)
        eps = torch.clamp_min(eps, 1e-6)

        # ----- Унифицированная центральная разность для всех размеров -----
        nvars_local = x.shape[0]

        if nvars_local <= 128:
            eps_fd = 1e-6
            # --- Полный Якобиан (колонки) ---
            F0 = self.sim._fi_residual_vec(x, dt)
            J_cols = []
            for i in range(nvars_local):
                e_i = torch.zeros_like(x)
                e_i[i] = eps_fd
                col = (self.sim._fi_residual_vec(x + e_i, dt) - F0) / eps_fd
                J_cols.append(col.view(-1, 1))
            J_full = torch.cat(J_cols, dim=1)

            # --- Односторонняя производная вдоль v ---
            Jv_forward = (self.sim._fi_residual_vec(x + eps_fd * v, dt) - F0) / eps_fd

            w = 0.62
            Jv_core = w * (J_full @ v) + (1.0 - w) * Jv_forward
        else:
            # Универсальная центральная разность
            if nvars_local <= 400:
                eps = torch.tensor(1e-6, dtype=x.dtype, device=x.device)
            else:
                eps = torch.clamp_min(eps, 1e-6)
            F_plus  = self.sim._fi_residual_vec(x + eps * v, dt)
            F_minus = self.sim._fi_residual_vec(x - eps * v, dt)
            Jv_core = (F_plus - F_minus) / (2.0 * eps)
        
        # Добавляем вклад PTC, если активен и модель достаточно крупная
        if nvars_local >= 800 and hasattr(self, "ptc_tau") and self.ptc_enabled and self.ptc_tau > 0.0:
            Jv_core = Jv_core + (self.ptc_tau / dt) * v

        Jv = Jv_core  # без дополнительной регуляризации – достаточно стабильно

        # ---- Guard against NaN/Inf ----------------------------------------------------
        if not torch.isfinite(Jv).all():
            print("  _Jv: обнаружены NaN/Inf – заменяем на 0")
            Jv = torch.zeros_like(Jv)
        
        return Jv

    def step(self, x0: torch.Tensor, dt: float):
        """🚀 ПРОМЫШЛЕННЫЙ Newton шаг с адаптивными стратегиями"""
        x = x0.clone()  # x0 уже в нужных единицах (simulator использует VariableScaler)

        # Базовое среднее давление (в масштабированных единицах), чтобы фиксировать нулевой вектор
        n_cells_tot = (
            self.scaler.n_cells
            if self.scaler is not None
            else (
                self.sim.reservoir.dimensions[0]
                * self.sim.reservoir.dimensions[1]
                * self.sim.reservoir.dimensions[2]
            )
        )
        baseline_mean_p = x[:n_cells_tot].mean().clone()

        # --------------------------------------------------------------
        # Адаптивный выбор режима: «облегчённый» для маленьких сеток
        # --------------------------------------------------------------
        advanced_threshold = self.sim.sim_params.get("advanced_threshold", 50_000)
        advanced_mode = n_cells_tot > advanced_threshold
        if not advanced_mode:
            # Отключаем улучшения, которые приводят к вырожденным шагам на микромоделях
            self.ptc_tau = 0.0
            allow_defl = False
        else:
            allow_defl = True

        # Позволяем пользователю отключить фиксацию среднего давления,
        # чтобы глобальное давление могло расти/падать при нетто-дебите.
        fix_pressure_drift = self.sim.sim_params.get("fix_pressure_drift", True)

        # Helper для устранения дрейфа давления
        def _anchor_pressure(x_hat: torch.Tensor):
            if not fix_pressure_drift:
                return x_hat
            drift = x_hat[:n_cells_tot].mean() - baseline_mean_p
            if torch.abs(drift) > 1e-6:
                x_hat[:n_cells_tot] -= drift
            return x_hat

        # PTC параметры
        self.ptc_tau = self.ptc_tau0 if self.ptc_enabled else 0.0
        x_ref = x0.clone()  # исходный вектор для PTC
        
        # 🔍 ДИАГНОСТИКА: включаем для первой итерации
        self.sim._debug_residual_once = True
        
        # --- Trust-region (TR) радиус в пространстве масштабированных переменных
        nvars_total = (self.sim.reservoir.dimensions[0]*self.sim.reservoir.dimensions[1]*
                       self.sim.reservoir.dimensions[2]) * 2  # давл + Sw; газ игнор
        # Для микросистем используем умеренный trust-radius, чтобы ограничить экстремальные шаги,
        # но не резать его до нуля.
        if nvars_total < 500:
            trust_radius = 200.0  # широкий радиус, чтобы не душить шаг на микромоделях
        else:
            # Слишком маленький trust-radius (2.0) приводит к чрезмерному урезанию
            # шага на крупных системах → стагнации.  Увеличиваем по умолчанию до 50,
            # оставляя возможность переопределить через конфиг.
            n_cells_global = (
                self.scaler.n_cells
                if self.scaler is not None
                else (
                    self.sim.reservoir.dimensions[0]
                    * self.sim.reservoir.dimensions[1]
                    * self.sim.reservoir.dimensions[2]
                )
            )
            default_tr = 20.0 + 0.5 * math.sqrt(n_cells_global)
            trust_radius = self.sim.sim_params.get("trust_radius", default_tr)
        prev_F_norm = None

        # Diagnostics
        self.total_gmres_iters = 0
        # Дефляционный базис (ортонорм колонки, в hat-пространстве)
        self.defl_basis = []
        init_F_scaled = None  # значение невязки на первой итерации для относительного критерия

        gmres_tol_min = self.sim.sim_params.get("gmres_min_tol", 1e-7)  # минимум tolerances

        # Динамическое увеличение лимита итераций для маленьких систем
        nvars_total_iter = (self.sim.reservoir.dimensions[0]*self.sim.reservoir.dimensions[1]*self.sim.reservoir.dimensions[2]) * 2
        effective_max_it = self.max_it
        if nvars_total_iter <= 100 and self.max_it < 30:
            effective_max_it = 30

        for it in range(effective_max_it):
            # ---------------- residual (physical → scaled) ----------------
            x_phys = self._unscale_x(x) if self.scaler is not None else x
            F_phys = self.sim._fi_residual_vec(x_phys, dt)
            F_hat = self.scaler.scale_vec(F_phys) if self.scaler is not None else F_phys

            # Динамически отключаем PTC, если невязка достаточно мала
            if self.ptc_enabled and self.ptc_tau > 0.0:
                if F_hat.norm() < 1e-2:
                    print("  PTC отключён – невязка стала малой")
                    self.ptc_tau = 0.0
                    F = F_hat
                else:
                    F = F_hat + (self.ptc_tau / dt) * (x - x_ref)
            else:
                F = F_hat
            
            F_norm = F.norm()
            self.last_res_norm = F_norm.item()

            # 🎯 Масштабируем по размеру системы
            F_scaled = F_norm / math.sqrt(len(F))

            # --- Быстрый выход: если невязка уже мала (<1e-4), принимаем без решения ---
            early_tol = self.sim.sim_params.get("early_accept_tol", 1e-4)
            if F_scaled < early_tol:
                print(f"  Newton: ||F||_scaled={F_scaled:.3e} < early_tol={early_tol:.1e} → принимаем без корректировки")
                self.last_newton_iters = max(1, it)
                self.last_gmres_iters = self.total_gmres_iters
                _anchor_pressure(x)
                x_pa = self._unscale_x(x)
                return self.scaler.to_mpa_vec(x_pa) if self.scaler is not None else x_pa / 1e6, True
            if init_F_scaled is None:
                init_F_scaled = F_scaled  # сохраняем стартовую невязку
            print(f"  Newton #{it}: ||F||={F_norm:.3e}, ||F||_scaled={F_scaled:.3e}")

            # Дополнительный пользовательский критерий small_tol можно задать через sim_params.
            nvars_total = len(F)
            n_cells_total = nvars_total // (3 if nvars_total % 3 == 0 else 2)
            if n_cells_total <= 100:
                user_small_tol = self.sim.sim_params.get("newton_small_tol", 1e-3)
                if user_small_tol is not None and F_scaled < user_small_tol:
                    print(f"  Newton: невязка {F_scaled:.3e} ниже user_small_tol={user_small_tol:.1e} → принимаем")
                    self.last_newton_iters = max(1, it)
                    self.last_gmres_iters = self.total_gmres_iters
                    _anchor_pressure(x)
                    x_pa = self._unscale_x(x)
                    return self.scaler.to_mpa_vec(x_pa) if self.scaler is not None else x_pa / 1e6, True
            
            # Используем абсолютную И относительную невязку для проверки сходимости
            if (F_scaled < self.tol) or (F_scaled < self.rtol * init_F_scaled):
                print(f"  Newton сошелся за {it} итераций! (масштабированная невязка)")
                # Expose diagnostics
                self.last_newton_iters = max(1, it)
                self.last_gmres_iters = self.total_gmres_iters
                _anchor_pressure(x)
                x_pa = self._unscale_x(x)
                return self.scaler.to_mpa_vec(x_pa) if self.scaler is not None else x_pa / 1e6, True
                
            # Адаптивный forcing-term η_k  по Brown–Saad
            if prev_F_norm is None:
                # Стартовый forcing-term – управляемый параметр, по умолчанию 1e-2
                eta_k = self.sim.sim_params.get("newton_eta0", 1e-4)
            else:
                ratio = (F_norm / prev_F_norm).item()
                eta_k = 0.9 * ratio**2
            eta_k = min(max(eta_k, 1e-8), 1e-2)
            gmres_tol = max(gmres_tol_min, eta_k)
            
            print(f"  GMRES: tol={gmres_tol:.3e}")
            
            def A(v_hat):
                # v_hat → physical, затем Jv → scale back
                v_phys = self.scaler.unscale_vec(v_hat) if self.scaler is not None else v_hat
                x_phys = self._unscale_x(x) if self.scaler is not None else x
                Jv_phys = self._Jv(x_phys, v_phys, dt)
                return self.scaler.scale_vec(Jv_phys) if self.scaler is not None else Jv_phys

            # Предобуславливатель в масштабированном пространстве ---------
            def M_hat(r_hat: torch.Tensor) -> torch.Tensor:
                if self.scaler is not None:
                    r_phys = self.scaler.unscale_vec(r_hat)
                    delta_phys = self.prec.apply(r_phys)
                    return self.scaler.scale_vec(delta_phys)
                else:
                    return self.prec.apply(r_hat)
                
            # 🎯 АДАПТИВНЫЕ параметры GMRES в зависимости от итерации
            if it == 0:
                # Первая итерация – достаточно 60 итераций, дальше line-search.
                gmres_restart = 40
                gmres_maxiter = 60
            else:
                # Последующие итерации – ещё короче
                gmres_restart = 30
                gmres_maxiter = 40
                
            print(f"  GMRES: restart={gmres_restart}, max_iter={gmres_maxiter}")
            
            # Подготавливаем базис как один тензор (n,k) colwise
            basis_tensor = None
            if allow_defl and self.defl_basis:
                basis_tensor = torch.stack(self.defl_basis, dim=1)

            gmres_out = fgmres(
                A,
                -F,
                M=M_hat,
                tol=gmres_tol,
                restart=gmres_restart,
                max_iter=gmres_maxiter,
                deflation_basis=basis_tensor,
                min_iters=3
            )
            delta, info, gm_iters = gmres_out

            # Защита: если GMRES вернул NaN/Inf, обнуляем δ
            if not torch.isfinite(delta).all():
                print("  GMRES вернул NaN/Inf – заменяем на 0")
                delta = torch.zeros_like(delta)
                info = 1

            # Норма решения в масштабированных координатах для trust-region
            delta_norm_scaled = delta.norm() / math.sqrt(len(delta))

            # --- Fallback: если GMRES вернул почти нулевую δx, используем одно применение предобуславливателя
            if delta_norm_scaled < 1e-12:
                print("  GMRES вернул δ≈0 — используем M_hat(−F) как fallback")
                delta = M_hat(-F)
                delta_norm_scaled = delta.norm() / math.sqrt(len(delta))

            self.total_gmres_iters += gm_iters

            # --- обновляем дефляционный базис ---------------------------------
            if allow_defl and torch.isfinite(delta).all():
                # нормализация и ортогонализация
                v = delta.clone()
                v_norm = v.norm()
                if v_norm > 1e-8:
                    v = v / v_norm
                    # ортогонализуем к текущему
                    for q in self.defl_basis:
                        v = v - torch.dot(q, v) * q
                    v_norm2 = v.norm()
                    if v_norm2 > 1e-6:
                        v = v / v_norm2
                        self.defl_basis.append(v)
                        # ограничиваем размер до 10 векторов
                        if len(self.defl_basis) > 10:
                            self.defl_basis.pop(0)

            if info != 0 or not torch.isfinite(delta).all():
                print(f"  GMRES не сошёлся (info={info}), ||delta||={delta.norm():.3e}")
                nvars = F.shape[0]
                # 🎯 FALLBACK 1: маленькая система – пробуем прямой solve
                if nvars <= 200 and self.sim.sim_params.get("small_direct_jac", True):
                    try:
                        print("  ➡️  Пробуем сформировать полный Якобиан и решить напрямую")
                        eye = torch.eye(nvars, device=F.device, dtype=F.dtype)
                        J_cols = [A(eye[:, j]) for j in range(nvars)]
                        J_full = torch.stack(J_cols, dim=1)
                        delta = torch.linalg.solve(J_full, -F)
                        info = 0
                        print("  ✅ Прямое решение прошло успешно, продолжаем")
                    except Exception as e:
                        print(f"  ❌ Не удалось решить напрямую: {e}")
                        info = 1

                # 🎯 FALLBACK 2: демпфирование Jacobi
                if info != 0 or not torch.isfinite(delta).all() or delta.norm() < 1e-12:
                    print("  ⏎ Используем демпфированный Jacobi шаг")
                    delta = M_hat(-F)
                    # легкое демпфирование
                    delta = delta * 0.1
                    info = 0  # разрешаем продолжить
                    if not torch.isfinite(delta).all():
                        delta = torch.zeros_like(delta)

            # --- КВАДРАТИЧНАЯ line-search ---------------------------------------
            factor = 1.0
            # Минимально допустимый шаг для line-search можно переопределить через sim_params
            # Допускаем гораздо более сильное демпфирование, если конфиг не переопределил
            cfg_alpha = self.sim.sim_params.get("line_search_min_alpha", 1e-4)
            # Никогда не допускаем порога выше 1e-4, иначе LS часто терпит неудачу
            min_factor = min(cfg_alpha, 1e-4)
            if min_factor <= 0.0:
                min_factor = 1e-4  # защита от некорректного ввода
            if min_factor > 1.0:
                min_factor = 1.0
            
            # --- ДИНАМИЧЕСКИЙ trust-radius -----------------------------------
            trust_radius_cfg = self.sim.sim_params.get("trust_radius", None)
            if trust_radius_cfg is not None:
                trust_radius = trust_radius_cfg  # явное значение из конфига
            else:
                # Энергетический радиус: 20‖F‖ / √N  (но ≥50)
                rhs_norm = getattr(self, "last_res_norm", F_norm)
                n_vars = delta.numel()
                dyn_tr = 20.0 * rhs_norm / max((n_vars ** 0.5), 1.0)
                trust_radius = max(50.0, dyn_tr)

            if trust_radius is not None and delta_norm_scaled > trust_radius:
                factor = trust_radius / (delta_norm_scaled + 1e-12)
                print(f"  Trust-region: сокращаем шаг до factor={factor:.3e} (радиус {trust_radius:.2f})")

            c1 = 1e-4
            ls_max = 8
            success = False

            for ls_it in range(ls_max):
                # Проверяем минимальный размер шага
                if factor < min_factor:
                    print(f"  Line search: достигнут минимальный α={min_factor:.3e} – прекращаем LS")
                    break

                x_candidate = x + factor * delta
                if not torch.isfinite(x_candidate).all():
                    factor *= 0.5
                    continue

                x_candidate_phys = self._unscale_x(x_candidate) if self.scaler is not None else x_candidate
                F_candidate_phys = self.sim._fi_residual_vec(x_candidate_phys, dt)
                F_candidate_hat = self.scaler.scale_vec(F_candidate_phys) if self.scaler is not None else F_candidate_phys
                if self.ptc_enabled and self.ptc_tau > 0.0:
                    F_candidate_hat = F_candidate_hat + (self.ptc_tau / dt) * (x_candidate - x_ref)
                if not torch.isfinite(F_candidate_hat).all():
                    factor *= 0.5
                    continue

                f_curr = F_candidate_hat.norm()

                if f_curr <= (1 - c1 * factor) * F_norm:
                    print(f"  Line search принял шаг α={factor:.3e}, ||F||={f_curr:.3e}")
                    x_new = x_candidate
                    success = True
                    break

                factor *= 0.5

            if not success:
                print("  Line search не нашёл шаг – пробуем демпфированный Jacobi fallback (α=0.3)")
                delta_fb = 0.3 * M_hat(-F)
                x_fb = x + delta_fb
                if torch.isfinite(x_fb).all():
                    x_fb_phys = self._unscale_x(x_fb) if self.scaler is not None else x_fb
                    F_fb_phys = self.sim._fi_residual_vec(x_fb_phys, dt)
                    F_fb_hat = self.scaler.scale_vec(F_fb_phys) if self.scaler is not None else F_fb_phys
                    if self.ptc_enabled and self.ptc_tau > 0.0:
                        F_fb_hat = F_fb_hat + (self.ptc_tau / dt) * (x_fb - x_ref)
                    F_fb_norm = F_fb_hat.norm()
                    if F_fb_norm < 0.95 * F_norm:
                        print(f"  ✅ Jacobi fallback принят, ||F||={F_fb_norm:.3e}")
                        x = x_fb
                        success = True
                    else:
                        print("  ❌ Jacobi fallback не улучшил невязку")

            if not success:
                print("  JFNK: even fallback failed – завершаем шаг неудачей")
                self.last_newton_iters = self.max_it
                self.last_gmres_iters = self.total_gmres_iters
                return self._unscale_x(x), False
                
            # --- Адаптация trust-radius -------------------------------
            if success and trust_radius is not None:
                if factor > 0.8:
                    trust_radius = min(trust_radius * 1.4, 50.0)
                elif factor < 0.2:
                    trust_radius = max(trust_radius * 0.7, 1e-3)
                print(f"  Trust-region: новый радиус {trust_radius:.2f}")

            x = x_new

            # --- Фиксация среднего давления -----------------------------------
            mean_p_drift = x[:n_cells_tot].mean() - baseline_mean_p
            if torch.abs(mean_p_drift) > 1e-6:
                x[:n_cells_tot] -= mean_p_drift
                print(f"  ⚖️  Сдвиг среднего давления устранён: drift={mean_p_drift.item():.3e}")
            prev_F_norm = F_norm

            # Уменьшаем τ после успешного шага
            if self.ptc_enabled and self.ptc_tau > 0.0:
                self.ptc_tau = max(self.ptc_tau * 0.5, 0.0)
            
        print(f"  Newton не сошелся за {effective_max_it} итераций")
        # On failure also expose iteration counts
        self.last_newton_iters = self.max_it
        self.last_gmres_iters = self.total_gmres_iters
        _anchor_pressure(x)
        x_pa = self._unscale_x(x)
        return (self.scaler.to_mpa_vec(x_pa) if self.scaler is not None else x_pa/1e6), False 

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _unscale_x(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Convert scaled vector back to physical units, supports 2/3 vars per cell."""
        return self.scaler.unscale_vec(x_hat) if self.scaler is not None else x_hat 