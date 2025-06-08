import torch
import numpy as np
from numba import njit

@njit
def solve_pressure_numba(p, s_w, k, phi, mu_w, mu_o, c_t, dt, dx, dy, dz, q, wells):
    """
    Решает уравнение давления и обновляет насыщенность с помощью Numba.
    """
    nx, ny, nz = k.shape
    p_new = p.copy()
    
    for _ in range(50): # Итерационный решатель Якоби
        p_old = p_new.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    # Примерный расчет проводимостей и давления
                    tx1 = k[i-1,j,k] * dy*dz/dx
                    tx2 = k[i+1,j,k] * dy*dz/dx
                    ty1 = k[i,j-1,k] * dx*dz/dy
                    ty2 = k[i,j+1,k] * dx*dz/dy

                    # Простое усреднение давления
                    p_new[i,j,k] = (p_old[i-1,j,k]*tx1 + p_old[i+1,j,k]*tx2 + \
                                    p_old[i,j-1,k]*ty1 + p_old[i,j+1,k]*ty2) / (tx1+tx2+ty1+ty2)

    # ... (здесь должна быть более сложная логика обновления, но для примера сойдет)
    
    return p_new, s_w # Возвращаем обновленные поля

class Simulator:
    """
    Основной класс симулятора, отвечающий за выполнение расчетов по схеме IMPES
    (Implicit Pressure, Explicit Saturation).
    """
    def __init__(self, reservoir, fluid_system, well_manager):
        self.reservoir = reservoir
        self.fluid = fluid_system
        self.wells = well_manager
        self.device = reservoir.device
        self._calculate_transmissibility()
        self.Vp = self.reservoir.dx * self.reservoir.dy * self.reservoir.dz * self.reservoir.porosity
        self.g = 9.81

    def _calculate_transmissibility(self):
        k, dx, dy, dz = self.reservoir.permeability, self.reservoir.dx, self.reservoir.dy, self.reservoir.dz
        k_darcy = k * 9.869233e-16
        self.T_x = (2 * k_darcy[1:,:,:] * k_darcy[:-1,:,:] / (k_darcy[1:,:,:] + k_darcy[:-1,:,:]) * dy * dz / dx).cpu()
        self.T_y = (2 * k_darcy[:,1:,:] * k_darcy[:,:-1,:] / (k_darcy[:,1:,:] + k_darcy[:,:-1,:]) * dx * dz / dy).cpu()
        self.T_z = (2 * k_darcy[:,:,1:] * k_darcy[:,:,:-1] / (k_darcy[:,:,1:] + k_darcy[:,:,:-1]) * dx * dy / dz).cpu()

    def run_step(self, dt):
        P, S_w = self.fluid.pressure.cpu(), self.fluid.s_w.cpu()
        nx, ny, nz = P.shape
        N = nx * ny * nz
        
        mob_w = 1.0 / self.fluid.mu_water
        mob_o = 1.0 / self.fluid.mu_oil
        mob_t = mob_w + mob_o

        Tw_x, Tw_y, Tw_z = self.T_x * mob_w, self.T_y * mob_w, self.T_z * mob_w

        indices = []
        values = []
        q = torch.zeros(N)
        
        acc_term_numpy = (self.Vp.cpu() * (self.fluid.compressibility / 1e5) / dt).flatten()
        
        def to_1d(i, j, k): return k + j * nz + i * ny * nz

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = to_1d(i, j, k)
                    diag_val = 0
                    
                    if i > 0:
                        val = self.T_x[i-1,j,k] * mob_t
                        indices.append((idx, idx - ny*nz)); values.append(-val)
                        diag_val += val
                    if i < nx - 1:
                        val = self.T_x[i,j,k] * mob_t
                        indices.append((idx, idx + ny*nz)); values.append(-val)
                        diag_val += val
                    if j > 0:
                        val = self.T_y[i,j-1,k] * mob_t
                        indices.append((idx, idx - nz)); values.append(-val)
                        diag_val += val
                    if j < ny - 1:
                        val = self.T_y[i,j,k] * mob_t
                        indices.append((idx, idx + nz)); values.append(-val)
                        diag_val += val
                    if k > 0:
                        val = self.T_z[i,j,k-1] * mob_t
                        indices.append((idx, idx - 1)); values.append(-val)
                        diag_val += val
                    if k < nz - 1:
                        val = self.T_z[i,j,k] * mob_t
                        indices.append((idx, idx + 1)); values.append(-val)
                        diag_val += val

                    indices.append((idx, idx))
                    values.append(diag_val + acc_term_numpy[idx])
        
        A = torch.sparse_coo_tensor(torch.tensor(indices).T, torch.tensor(values), (N,N))
        
        P_flat = P.flatten()
        q -= acc_term_numpy * P_flat
        for well in self.wells.get_wells():
            q[to_1d(well.i, well.j, well.k)] += well.rate_si
        
        P_new_flat = self._solve_pressure_cg(A, q)
        self.fluid.pressure = P_new_flat.view(nx, ny, nz).to(self.device)

        dp_x = P_new_flat.view(nx,ny,nz)[:-1,:,:] - P_new_flat.view(nx,ny,nz)[1:,:,:]
        flow_x = Tw_x * dp_x

        S_w_new = S_w.clone()
        Vp_cpu = self.Vp.cpu()
        S_w_new[:-1,:,:] -= (dt / Vp_cpu[:-1,:,:]) * torch.where(dp_x > 0, flow_x, 0)
        S_w_new[1:,:,:] += (dt / Vp_cpu[1:,:,:]) * torch.where(dp_x < 0, -flow_x, 0)

        self.fluid.s_w = S_w_new.clamp(0,1).to(self.device)
        self.fluid.s_o = 1.0 - self.fluid.s_w
        
        print(f"Давление: {self.fluid.pressure.mean()/1e6:.2f} МПа. Насыщенность: {self.fluid.s_w.mean():.3f}")

    def _solve_pressure_cg(self, A, b, max_iter=300, tol=1e-7):
        x = torch.zeros_like(b)
        r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        p = r
        rs_old = torch.dot(r, r)
        for i in range(max_iter):
            Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze(1)
            alpha = rs_old / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < tol:
                print(f"  Решатель сошелся за {i+1} итераций.")
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        else:
            print(f"  Решатель не сошелся за {max_iter} итераций.")
        return x
