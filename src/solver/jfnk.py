import torch
from .krylov import gmres
from .cpr import CPRPreconditioner

class FullyImplicitSolver:
    def __init__(self, simulator, backend="amgx"):
        self.sim = simulator
        self.prec = CPRPreconditioner(simulator.reservoir, simulator.fluid,
                                       backend=backend)
        self.tol = simulator.sim_params.get("newton_tolerance", 1e-3)
        self.max_it = simulator.sim_params.get("newton_max_iter", 12)

    def _Jv(self, x: torch.Tensor, v: torch.Tensor, dt):
        """Вычисляет произведение Якобиана на вектор *v* по двухточечной
        разности. Шаг eps выбираем по рекомендации Брауна – порядка √ε
        машинного, масштабируемый на ‖x‖, чтобы избежать слишком мелких
        разностей, приводящих к шуму.
        """
        # Машинное ε для float32 или float64 в зависимости от dtype
        dtype_eps = 1e-7 if x.dtype == torch.float32 else 1e-15
        eps = torch.sqrt(torch.tensor(dtype_eps, dtype=x.dtype, device=x.device))
        eps = eps * (1.0 + torch.norm(x)) / (torch.norm(v) + 1e-12)
        return (self.sim._fi_residual_vec(x + eps * v, dt) -
                self.sim._fi_residual_vec(x, dt)) / eps

    def step(self, x0: torch.Tensor, dt: float):
        x = x0.clone()
        for it in range(self.max_it):
            F = self.sim._fi_residual_vec(x, dt)
            if F.norm() < self.tol:
                return x, True
            def A(v):
                return self._Jv(x, v, dt)
            delta, info = gmres(A, -F, self.prec.apply, tol=1e-6, maxiter=200)

            if info != 0 or not torch.isfinite(delta).all():
                # Линейный солвер не сошёлся – прерываемся, чтобы симулятор
                # мог попробовать fallback, вместо того чтобы портить x.
                print(f"  GMRES не сошёлся (info={info}). Прерывание JFNK.")
                return x, False

            # простая line-search
            factor = 1.0
            while factor > 1e-4:
                x_new = x + factor*delta
                if torch.isfinite(self.sim._fi_residual_vec(x_new, dt)).all():
                    x = x_new
                    break
                factor *= 0.5
        return x, False 