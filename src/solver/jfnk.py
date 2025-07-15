import torch
import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from linear_gpu.gmres import gmres
from .cpr import CPRPreconditioner

class FullyImplicitSolver:
    def __init__(self, simulator, backend="amgx"):
        self.sim = simulator

        # --- Variable scaling (pressure → dimensionless) ---------------
        self.scaler = simulator.scaler  # используем уже созданный в Simulator

        # CPR preconditioner (pressure block) ------------------------------
        smoother = simulator.sim_params.get("smoother", "jacobi")
        self.prec = CPRPreconditioner(simulator.reservoir,
                                       simulator.fluid,
                                       backend=backend,
                                       smoother=smoother)

        # Newton params ----------------------------------------------------
        self.tol = simulator.sim_params.get("newton_tolerance", 1e-7)  # абсолютная
        self.rtol = simulator.sim_params.get("newton_rtol", 1e-4)       # относительная
        self.max_it = simulator.sim_params.get("newton_max_iter", 15)

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
        eps = torch.sqrt(torch.tensor(dtype_eps, dtype=x.dtype, device=x.device))
        eps = eps * (1.0 + torch.norm(x)) / (torch.norm(v) + 1e-12)
        
        # 🎯 ПРОМЫШЛЕННАЯ регуляризация для стабильности
        regularization = 1e-6
        Jv_core = (self.sim._fi_residual_vec(x + eps * v, dt) -
                   self.sim._fi_residual_vec(x, dt)) / eps

        # Добавляем вклад PTC, если активен
        if hasattr(self, "ptc_tau") and self.ptc_enabled and self.ptc_tau > 0.0:
            Jv_core = Jv_core + (self.ptc_tau / dt) * v

        # Диагональная регуляризация для стабильности (умеренная величина)
        regularization = 1e-8
        Jv = Jv_core + regularization * v
        
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
            trust_radius = self.sim.sim_params.get("trust_radius", 50.0)
        prev_F_norm = None

        # Diagnostics
        self.total_gmres_iters = 0
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
            
            # 🎯 Масштабируем по размеру системы
            F_scaled = F_norm / math.sqrt(len(F))
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
                    self.last_newton_iters = it
                    self.last_gmres_iters = self.total_gmres_iters
                    _anchor_pressure(x)
                    x_pa = self._unscale_x(x)
                    return self.scaler.to_mpa_vec(x_pa) if self.scaler is not None else x_pa / 1e6, True
            
            # Используем абсолютную И относительную невязку для проверки сходимости
            if (F_scaled < self.tol) or (F_scaled < self.rtol * init_F_scaled):
                print(f"  Newton сошелся за {it} итераций! (масштабированная невязка)")
                # Expose diagnostics
                self.last_newton_iters = it
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
                # Первая итерация - строгие параметры
                gmres_restart = 50
                gmres_maxiter = 200
            else:
                # Последующие итерации - более мягкие параметры
                gmres_restart = 30
                gmres_maxiter = 100
                
            print(f"  GMRES: restart={gmres_restart}, max_iter={gmres_maxiter}")
            
            gmres_out = gmres(
                A,
                -F,
                M=M_hat,
                tol=gmres_tol,
                restart=gmres_restart,
                max_iter=gmres_maxiter,
            )

            if len(gmres_out) == 3:
                delta, info, gm_iters = gmres_out
            else:
                delta, info = gmres_out
                gm_iters = gmres_maxiter  # pessimistic estimate

            self.total_gmres_iters += gm_iters

            if info != 0 or not torch.isfinite(delta).all():
                print(f"  GMRES не сошёлся (info={info}), ||delta||={delta.norm():.3e}")
                nvars = F.shape[0]
                # �� FALLBACK стратегия для маленьких систем: прямое решение J δ = -F
                if nvars <= 200 and self.sim.sim_params.get("small_direct_jac", True):
                    try:
                        print("  ➡️  Пробуем сформировать полный Якобиан и решить напрямую")
                        eye = torch.eye(nvars, device=F.device, dtype=F.dtype)
                        J_cols = []
                        for j in range(nvars):
                            col = A(eye[:, j])  # J * e_j
                            J_cols.append(col)
                        J_full = torch.stack(J_cols, dim=1)
                        delta = torch.linalg.solve(J_full, -F)
                        info = 0
                        print("  ✅ Прямое решение прошло успешно, продолжаем")
                    except Exception as e:
                        print(f"  ❌ Не удалось решить напрямую: {e}")
                        info = 1
                # 🎯 FALLBACK стратегия: демпфирование (без полного Якобиана)
                if info != 0 or not torch.isfinite(delta).all():
                    if torch.isfinite(delta).all() and delta.norm() > 0:
                        n_small = len(delta)
                        if n_small <= 100:
                            print("  Маленькая система – используем полное решение GMRES без демпфирования")
                            # без дополнительного масштаба
                        else:
                            print("  Используем демпфированное решение GMRES")
                            delta = delta * 0.1
                    else:
                        print("  GMRES failed полностью. Прерывание JFNK.")
                        self.last_newton_iters = self.max_it
                        self.last_gmres_iters = self.total_gmres_iters
                        return self._unscale_x(x), False

            # 🚀 ПРОМЫШЛЕННЫЙ line-search с логированием
            # Определяем число ячеек до первого использования, чтобы избежать UnboundLocalError
            n_cells = (
                self.scaler.n_cells
                if self.scaler is not None
                else (
                    self.sim.reservoir.dimensions[0]
                    * self.sim.reservoir.dimensions[1]
                    * self.sim.reservoir.dimensions[2]
                )
            )

            # Удаляем компоненту постоянного смещения давления (null-space)
            if delta.shape[0] >= n_cells:
                n_cells_local = n_cells  # то же самое значение
                if n_cells_local <= 100:
                    mean_dp = delta[:n_cells_local].mean()
                    delta[:n_cells_local] -= mean_dp
                    print(f"  ⬇️  Убрано среднее δp={mean_dp.item():.3e} (компонента null-space)")
            vars_per_cell = delta.shape[0] // n_cells
            if self.scaler is not None:
                pressure_scaled = delta[:n_cells] * self.scaler.inv_p_scale
            else:
                pressure_scaled = delta[:n_cells] / 1e6
            if vars_per_cell == 3:
                delta_scaled = torch.cat([pressure_scaled, delta[n_cells:]])
            else:
                delta_scaled = torch.cat([pressure_scaled, delta[n_cells:]])
            delta_norm_scaled = delta_scaled.norm()
            print(f"  Line search: ||delta||_scaled={delta_norm_scaled:.3e}")

            # --- Small-step termination -----------------------------------
            small_delta_tol = self.sim.sim_params.get("delta_small_tol", 1e-4)
            small_F_tol = self.sim.sim_params.get("F_small_tol", 1e-2)
            if delta_norm_scaled < small_delta_tol and F_scaled < small_F_tol:
                print("  Newton: очень малый шаг и малая невязка – считаем, что сошлось")
                self.last_newton_iters = it
                self.last_gmres_iters = self.total_gmres_iters
                _anchor_pressure(x)
                x_pa = self._unscale_x(x)
                return (
                    self.scaler.to_mpa_vec(x_pa) if self.scaler is not None else x_pa / 1e6,
                    True,
                )

            # --- BACKTRACKING Armijo line-search ---------------------------------
            factor = 1.0
            # --- Trust-region (если включён) ---------------------------
            if trust_radius is not None:
                if delta_norm_scaled > trust_radius:
                    factor = trust_radius / (delta_norm_scaled + 1e-12)
                    print(f"  Trust-region: сокращаем шаг до factor={factor:.3e} (радиус {trust_radius:.2f})")

            c1 = 1e-4  # Armijo constant
            ls_max_iter = 12
            success = False

            for ls_it in range(ls_max_iter):
                x_candidate = x + factor * delta

                # Проверяем числовую корректность
                if not torch.isfinite(x_candidate).all():
                    factor *= 0.5
                    continue

                F_candidate = self.sim._fi_residual_vec(x_candidate, dt)
                if self.ptc_enabled and self.ptc_tau > 0.0:
                    F_candidate = F_candidate + (self.ptc_tau / dt) * (x_candidate - x_ref)
                if not torch.isfinite(F_candidate).all():
                    factor *= 0.5
                    continue

                F_candidate_norm = F_candidate.norm()
                # Условие Армихо: ||F(x+αΔ)|| <= (1 - c1*α) * ||F(x)||
                if F_candidate_norm <= (1.0 - c1 * factor) * F_norm:
                    print(f"  Line search успешно (Armijo): factor={factor:.3e}, ||F_new||={F_candidate_norm:.3e}")
                    x_new = x_candidate
                    success = True
                    break
                else:
                    print(f"  Line search уменьшает шаг: factor={factor:.3e} -> {(factor*0.5):.3e}, ||F_new||={F_candidate_norm:.3e}")
                    factor *= 0.5

            if not success:
                print("  Line search не смогло найти приемлемый шаг. Прерывание JFNK.")
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