import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter  # для мягкого сглаживания давления

class Plotter:
    """
    Класс, отвечающий за визуализацию и сохранение результатов симуляции.
    """
    def __init__(self, reservoir):
        """
        Инициализация плоттера.
        :param reservoir: Экземпляр пласта для получения информации о сетке.
        """
        self.reservoir = reservoir

    def save_plots(self, pressure, saturation_w, filename, time_info=None, saturation_g=None):
        """
        Сохраняет 2D-карты давления и насыщенности в файл.
        Визуализируется центральный срез по оси Z.
        :param time_info: (Опционально) Строка с информацией о времени (например, "День 50").
        """
        # Выбираем Z-срез с максимальной дисперсией насыщенности,
        # чтобы на графике было видно изменения.
        if self.reservoir.nz > 1:
            stds = [float(np.std(saturation_w[:, :, k])) for k in range(self.reservoir.nz)]
            z_slice_idx = int(np.argmax(stds))
        else:
            z_slice_idx = 0
            
        p_slice = pressure[:, :, z_slice_idx]
        s_slice = saturation_w[:, :, z_slice_idx]

        # ------------------------------------------------------------------
        # Сохраняем базовую насыщенность (первый вызов) для отображения ∆Sw
        # ------------------------------------------------------------------
        if not hasattr(self, "_baseline_sw"):
            self._baseline_sw = s_slice.copy()

        fig, axes = plt.subplots(1, 3 if hasattr(self.reservoir, 'nz') else 2, figsize=(18, 5))
        ax1 = axes[0]
        
        title_suffix = f" (Срез Z={z_slice_idx})"
        if time_info:
            title_suffix = f" ({time_info}, Срез Z={z_slice_idx})"

        # ------------------------------------------------------------------
        # Давление – квадратные ячейки без «ступенек» DPI
        # ------------------------------------------------------------------
        nx, ny, _ = self.reservoir.dimensions
        # Лёгкое Гауссово сглаживание (σ=1) делает градиент давления
        p_img = gaussian_filter(p_slice / 1e6, sigma=1)

        im1 = ax1.imshow(p_img,
                         cmap='turbo',  # плавная непрерывная палитра без резких границ
                         origin='lower',
                         extent=(0, nx, 0, ny),
                         interpolation='lanczos',
                         aspect='equal')
        ax1.set_aspect('equal', adjustable='box')
        # Убираем тики – остаётся чистая картинка
        ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax1.set_title(f'Давление (МПа){title_suffix}')
        ax1.set_xlabel('Ячейка X')
        ax1.set_ylabel('Ячейка Y')
        fig.colorbar(im1, ax=ax1)

        # --------------------------------------------------------------
        # Газонасыщенность (если газовая фаза есть)
        # --------------------------------------------------------------
        if len(axes) == 3:
            ax_g = axes[1]
        else:
            ax_g = None

        if ax_g is not None:
            if saturation_g is not None:
                s_g_slice = saturation_g[:, :, z_slice_idx]
            else:
                s_g_slice = np.zeros_like(s_slice)
            im_g = ax_g.imshow(s_g_slice,
                               cmap='magma', origin='lower', extent=(0, nx, 0, ny),
                               interpolation='lanczos', vmin=0, vmax=1, aspect='equal')
            ax_g.set_title(f'Gas Saturation Sg{title_suffix}')
            ax_g.set_xlabel('Ячейка X')
            ax_g.set_ylabel('Ячейка Y')
            fig.colorbar(im_g, ax=ax_g)

        # Переставляем ax2 индекс в зависимости от наличия газового
        ax2 = axes[-1]

        # ------------------------------------------------------------------
        # ∆Sw: сглаживаем и выводим приятной палитрой ----------------
        # ------------------------------------------------------------------
        delta_sw = s_slice - self._baseline_sw

        # Гауссово сглаживание убирает «пиксельность»
        delta_sw_smooth = gaussian_filter(delta_sw, sigma=1)

        # Отбрасываем экстремальные значения для авто-контраста
        p2, p98 = np.percentile(delta_sw_smooth, [2, 98])
        span = max(abs(p2), abs(p98), 1e-4)

        # Если ∆Sw только положительная – сдвигаем минимум к 0
        vmin = 0.0 if delta_sw_smooth.min() >= 0 else -span
        vmax = span

        im2 = ax2.imshow(delta_sw_smooth,
                         cmap='viridis',
                         origin='lower',
                         extent=(0, nx, 0, ny),
                         interpolation='lanczos',
                         vmin=vmin, vmax=vmax,
                         aspect='equal')
        # Контур нуля (граница фронта)
        try:
            ax2.contour(delta_sw_smooth, levels=[0.0], colors='k', linewidths=0.5, origin='lower', extent=(0, nx, 0, ny))
        except Exception:
            pass  # может не быть уровня 0, игнорируем

        ax2.set_aspect('equal', adjustable='box')
        ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax2.set_xlabel('Ячейка X')
        ax2.set_ylabel('Ячейка Y')
        ax2.set_title(f'Приращение водонасыщенности ∆Sw{title_suffix}')
        fig.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig) # Закрываем фигуру, чтобы не потреблять память
        
        print(f"Графики сохранены в файл {filename}") 