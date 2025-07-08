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

        # --- scaling layer ------------------------------------------------
        try:
            from .scaling import VariableScaler
        except ImportError:
            VariableScaler = None

        self.scaler = VariableScaler(simulator.reservoir, simulator.fluid) if VariableScaler else None

        # CPR preconditioner (pressure block) ------------------------------
        self.prec = CPRPreconditioner(simulator.reservoir,
                                       simulator.fluid,
                                       backend=backend)

        # Newton params ----------------------------------------------------
        self.tol = simulator.sim_params.get("newton_tolerance", 1e-6)  # абсолютная
        self.rtol = simulator.sim_params.get("newton_rtol", 1e-3)       # относительная
        self.max_it = simulator.sim_params.get("newton_max_iter", 15)

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
        Jv = (self.sim._fi_residual_vec(x + eps * v, dt) -
              self.sim._fi_residual_vec(x, dt)) / eps
        
        # Добавляем диагональную регуляризацию для предотвращения сингулярности
        Jv = Jv + regularization * v
        
        return Jv

    def step(self, x0: torch.Tensor, dt: float):
        """🚀 ПРОМЫШЛЕННЫЙ Newton шаг с адаптивными стратегиями"""
        x = x0.clone()
        
        # 🔍 ДИАГНОСТИКА: включаем для первой итерации
        self.sim._debug_residual_once = True
        
        trust_radius = 1e12  # практически без ограничения (оставляем для защиты от NaN)
        prev_F_norm = None

        # Diagnostics
        self.total_gmres_iters = 0
        init_F_scaled = None  # значение невязки на первой итерации для относительного критерия

        gmres_tol_min = self.sim.sim_params.get("gmres_min_tol", 1e-5)  # раньше 1e-8, но по умолчанию 1e-5 достаточно

        for it in range(self.max_it):
            F_phys = self.sim._fi_residual_vec(x if self.scaler is None else self._unscale_x(x), dt)

            # Since _fi_residual_vec already outputs scaled pressure when scaler is present,
            # we can use it directly as nonlinear residual.
            F = F_phys
            
            F_norm = F.norm()
            
            # 🎯 Масштабируем по размеру системы
            F_scaled = F_norm / math.sqrt(len(F))
            if init_F_scaled is None:
                init_F_scaled = F_scaled  # сохраняем стартовую невязку
            print(f"  Newton #{it}: ||F||={F_norm:.3e}, ||F||_scaled={F_scaled:.3e}")
            
            # Используем абсолютную И относительную невязку для проверки сходимости
            if (F_scaled < self.tol) or (F_scaled < self.rtol * init_F_scaled):
                print(f"  Newton сошелся за {it} итераций! (масштабированная невязка)")
                # Expose diagnostics
                self.last_newton_iters = it
                self.last_gmres_iters = self.total_gmres_iters
                return x, True
                
            # Адаптивный forcing-term η_k  по Brown–Saad
            if prev_F_norm is None:
                # Стартовый forcing-term – управляемый параметр, по умолчанию 1e-2
                eta_k = self.sim.sim_params.get("newton_eta0", 1e-2)
            else:
                ratio = (F_norm / prev_F_norm).item()
                eta_k = 0.9 * ratio**2
            eta_k = min(max(eta_k, 1e-8), 1e-1)
            gmres_tol = max(gmres_tol_min, eta_k)
            
            print(f"  GMRES: tol={gmres_tol:.3e}")
            
            def A(v):
                # Convert v to physical for Jv evaluation if scaling active
                Ncells = self.sim.reservoir.dimensions[0]*self.sim.reservoir.dimensions[1]*self.sim.reservoir.dimensions[2]
                if self.scaler is not None:
                    v_phys = v.clone()
                    v_phys[:Ncells] = v[:Ncells] * self.scaler.p_scale
                else:
                    v_phys = v

                Jv_phys = self._Jv(self._unscale_x(x) if self.scaler else x, v_phys, dt)
                return Jv_phys
                
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
                M=self.prec.apply,
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
                
                # 🎯 FALLBACK стратегия: простое демпфирование
                if torch.isfinite(delta).all() and delta.norm() > 0:
                    print(f"  Используем демпфированное решение GMRES")
                    delta = delta * 0.1  # Сильное демпфирование
                else:
                    print(f"  GMRES failed полностью. Прерывание JFNK.")
                    # On failure also expose iteration counts
                    self.last_newton_iters = self.max_it
                    self.last_gmres_iters = self.total_gmres_iters
                    return x, False

            # 🚀 ПРОМЫШЛЕННЫЙ line-search с логированием
            Ncells = delta.shape[0] // (3 if delta.shape[0] % 2 == 0 and delta.shape[0] // (delta.shape[0] // 2) == 3 else 2)
            pressure_scaled = delta[:Ncells] / 1e6
            delta_scaled = torch.cat([pressure_scaled, delta[Ncells:]])
            delta_norm_scaled = delta_scaled.norm()
            print(f"  Line search: ||delta||_scaled={delta_norm_scaled:.3e}")

            factor = 1.0
            if delta_norm_scaled > trust_radius:
                factor = trust_radius / (delta_norm_scaled + 1e-12)
            
            x_new = None
            while factor > 1e-4:
                x_candidate = x + factor * delta
                if torch.isfinite(self.sim._fi_residual_vec(x_candidate, dt)).all():
                    x_new = x_candidate
                    print(f"  Line search успешно: factor={factor:.3e}")
                    break
                factor *= 0.5
                
            if x_new is None:
                print(f"  Line search failed. Прерывание JFNK.")
                # On failure also expose iteration counts
                self.last_newton_iters = self.max_it
                self.last_gmres_iters = self.total_gmres_iters
                return x, False
                
            # Адаптируем trust-radius в зависимости от фактического уменьшения ||F||
            if x_new is not None:
                reduct = prev_F_norm - F_norm if prev_F_norm is not None else None
                if reduct is not None and reduct > 0:
                    trust_radius = min(trust_radius * 1.5, 50.0)
                elif reduct is not None:
                    trust_radius = max(trust_radius * 0.5, 1e-2)

            x = x_new
            prev_F_norm = F_norm
            
        print(f"  Newton не сошелся за {self.max_it} итераций")
        # On failure also expose iteration counts
        self.last_newton_iters = self.max_it
        self.last_gmres_iters = self.total_gmres_iters
        return x, False 

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _unscale_x(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Convert scaled vector back to physical units, supports 2/3 vars per cell."""
        if self.scaler is None:
            return x_hat
        Ncells = self.sim.reservoir.dimensions[0]*self.sim.reservoir.dimensions[1]*self.sim.reservoir.dimensions[2]
        x_phys = x_hat.clone()
        x_phys[:Ncells] = x_hat[:Ncells] * self.scaler.p_scale  # back to Pa
        return x_phys 