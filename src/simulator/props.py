import torch
from typing import Dict

def compute_cell_props(sim, x_hat: torch.Tensor, dt_sec: float) -> Dict[str, torch.Tensor]:
    """Return basic per-cell properties required by CPR-tail.

    Currently returns only porosity phi (with rock compressibility) and cell volume.
    Expand later with phase mobilities & compressibilities.
    """
    res = sim.reservoir
    # porosity with rock compressibility
    phi0 = res.porosity_ref
    c_r = res.rock_compressibility if hasattr(res, 'rock_compressibility') else 0.0
    p_ref = getattr(res, 'pressure_ref', 1e5)
    # current pressure (physical) is first n_cells elements of x_hat (already physical)
    n_cells = phi0.numel()
    p_vec = x_hat[:n_cells].view_as(phi0)
    phi = phi0 * (1.0 + c_r * (p_vec - p_ref))
    cell_vol = res.cell_volume
    props = {
        'phi': phi.reshape(-1),               # 1-D
        'V': torch.full((n_cells,), cell_vol, device=phi.device, dtype=phi.dtype),
        'dt': torch.tensor(dt_sec, device=phi.device, dtype=phi.dtype)
    }
    return props 