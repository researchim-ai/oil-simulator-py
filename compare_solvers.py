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


def run_sim(config_path: str, solver_type: str, steps: int,
            gmres_tol: float | None = None,
            newton_eta0: float | None = None,
            newton_tol: float | None = None,
            newton_rtol: float | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Запускает расчёт и возвращает final поля (pressure[Pa], Sw, Sg/None)."""
    cfg = json.load(open(config_path))
    cfg_sim = cfg.get('simulation', {}).copy()
    cfg_sim['solver_type'] = solver_type
    if solver_type == 'fully_implicit':
        # Гасим остаток давления до уровня IMPES (значения могут быть переопределены аргументами)
        if gmres_tol is None:
            gmres_tol = 1e-9
        if newton_eta0 is None:
            newton_eta0 = 1e-4
        cfg_sim['gmres_min_tol'] = gmres_tol
        cfg_sim['newton_eta0'] = newton_eta0

        # Дополнительные критерии Ньютона можно задавать
        if newton_tol is not None:
            cfg_sim['newton_tolerance'] = newton_tol
        if newton_rtol is not None:
            cfg_sim['newton_rtol'] = newton_rtol

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


def make_plot(pressure_impes, pressure_fi, Sw, out_dir, sigma: float = 1.0,
              pressure_limits: tuple | None = None):
    """Сохраняет side-by-side карты давления IMPES и FI + карту их разности."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    # Берём центральный Z-срез
    z_idx = pressure_impes.shape[2] // 2
    p_imp = gaussian_filter(pressure_impes[:, :, z_idx] / 1e6, sigma=sigma)
    p_fi = gaussian_filter(pressure_fi[:, :, z_idx] / 1e6, sigma=sigma)
    p_diff = p_fi - p_imp

    if pressure_limits is not None:
        vmin, vmax = pressure_limits
    else:
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
    parser.add_argument('--fi_gmres_tol', type=float, default=1e-9,
                        help='Минимальный tol для GMRES в FI')
    parser.add_argument('--fi_newton_eta0', type=float, default=1e-4,
                        help='Стартовый forcing term η₀ Ньютона')
    parser.add_argument('--fi_newton_tol', type=float, default=1e-7,
                        help='Абсолютная невязка Ньютона')
    parser.add_argument('--fi_newton_rtol', type=float, default=1e-4,
                        help='Относительная невязка Ньютона')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                        help='Sigma для Gaussian фильтра при визуализации давлений')
    parser.add_argument('--fix_pressure_limits', nargs=2, type=float,
                        metavar=('PMIN', 'PMAX'),
                        help='Фиксированный диапазон цветов давления (МПа)')
    args = parser.parse_args()

    out_dir = os.path.join('results', 'compare_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    print("=== Запуск IMPES …")
    p_imp, Sw, _ = run_sim(args.config, 'impes', args.steps)
    print("=== Запуск Fully-Implicit …")
    p_fi, _, _ = run_sim(
        args.config, 'fully_implicit', args.steps,
        gmres_tol=args.fi_gmres_tol,
        newton_eta0=args.fi_newton_eta0,
        newton_tol=args.fi_newton_tol,
        newton_rtol=args.fi_newton_rtol
    )

    limits = tuple(args.fix_pressure_limits) if args.fix_pressure_limits else None
    make_plot(p_imp, p_fi, Sw, out_dir,
              sigma=args.smooth_sigma,
              pressure_limits=limits)


if __name__ == '__main__':
    main() 