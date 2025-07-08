import json, os, sys, argparse, datetime
from typing import Tuple
import numpy as np
import torch

# --- путь к пакету src -----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator
from plotting.plotter import Plotter


def run_sim(config_path: str, solver_type: str, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Запускает расчёт и возвращает final поля (pressure[Pa], Sw, Sg/None)."""
    cfg = json.load(open(config_path))
    cfg_sim = cfg.get('simulation', {}).copy()
    cfg_sim['solver_type'] = solver_type
    if solver_type == 'fully_implicit':
        # Гасим остаток давления до уровня IMPES
        cfg_sim.setdefault('gmres_min_tol', 1e-9)
        cfg_sim.setdefault('newton_eta0', 1e-4)

    # Отрубаем все промежуточные PNG/VTK – будем строить только финальный кадр
    cfg_sim['steps_per_output'] = 10 ** 9

    device = torch.device('cpu')  # для репродукции
    reservoir = Reservoir(cfg['reservoir'], device)
    wells = WellManager(cfg['wells'], reservoir)
    fluid = Fluid(cfg['fluid'], reservoir, device)

    sim = Simulator(reservoir, fluid, wells, cfg_sim, device)
    dt = sim.dt
    for _ in range(steps):
        ok = sim.run_step(dt)
        if not ok:
            raise RuntimeError(f"{solver_type} не сошёлся")

    pressure = fluid.pressure.cpu().numpy()
    Sw = fluid.s_w.cpu().numpy()
    Sg = fluid.s_g.cpu().numpy() if hasattr(fluid, 's_g') else None
    return pressure, Sw, Sg


def make_plot(pressure_impes, pressure_fi, Sw, out_dir):
    """Сохраняет side-by-side карты давления IMPES и FI + карту их разности."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    # Берём центральный Z-срез
    z_idx = pressure_impes.shape[2] // 2
    p_imp = gaussian_filter(pressure_impes[:, :, z_idx] / 1e6, sigma=0.7)
    p_fi = gaussian_filter(pressure_fi[:, :, z_idx] / 1e6, sigma=0.7)
    p_diff = p_fi - p_imp

    vmin = min(p_imp.min(), p_fi.min())
    vmax = max(p_imp.max(), p_fi.max())
    vdiff = max(abs(p_diff.min()), abs(p_diff.max()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ims = []
    for ax, data, title, vmn, vmx, cmap in (
        (axes[0], p_imp, 'IMPES P (MPa)', vmin, vmax, 'turbo'),
        (axes[1], p_fi,  'FI P (MPa)',     vmin, vmax, 'turbo'),
        (axes[2], p_diff,'ΔP FI-IMPES (MPa)', -vdiff, vdiff, 'seismic')
    ):
        im = ax.imshow(data, origin='lower', cmap=cmap, interpolation='bilinear', vmin=vmn, vmax=vmx)
        ax.set_title(title)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, 'compare_pressure.png')
    plt.savefig(png)
    plt.close(fig)
    print(f"Сравнительный график сохранён: {png}")


def main():
    parser = argparse.ArgumentParser(description="Сравнить IMPES и FI на одном конфиге")
    parser.add_argument('--config', required=True)
    parser.add_argument('--steps', type=int, default=1)
    args = parser.parse_args()

    out_dir = os.path.join('results', 'compare_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    print("=== Запуск IMPES …")
    p_imp, Sw, _ = run_sim(args.config, 'impes', args.steps)
    print("=== Запуск Fully-Implicit …")
    p_fi, _, _ = run_sim(args.config, 'fully_implicit', args.steps)

    make_plot(p_imp, p_fi, Sw, out_dir)


if __name__ == '__main__':
    main() 