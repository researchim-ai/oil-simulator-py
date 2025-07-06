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

    def save_plots(self, pressure, saturation, filename, time_info=None):
        """
        Сохраняет 2D-карты давления и насыщенности в файл.
        Визуализируется центральный срез по оси Z.
        :param time_info: (Опционально) Строка с информацией о времени (например, "День 50").
        """
        # Выбираем Z-срез с максимальной дисперсией насыщенности,
        # чтобы на графике было видно изменения.
        if self.reservoir.nz > 1:
            stds = [float(np.std(saturation[:, :, k])) for k in range(self.reservoir.nz)]
            z_slice_idx = int(np.argmax(stds))
        else:
            z_slice_idx = 0
            
        p_slice = pressure[:, :, z_slice_idx]
        s_slice = saturation[:, :, z_slice_idx]

        # ------------------------------------------------------------------
        # Сохраняем базовую насыщенность (первый вызов) для отображения ∆Sw
        # ------------------------------------------------------------------
        if not hasattr(self, "_baseline_sw"):
            self._baseline_sw = s_slice.copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
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

        # ------------------------------------------------------------------
        # ∆Sw = текущая – базовая: видно даже прирост 0.01
        # ------------------------------------------------------------------
        delta_sw = s_slice - self._baseline_sw

        # Убираем экстремальные пиксели: берём 2-й и 98-й перцентили
        p2, p98 = np.percentile(delta_sw, [2, 98])
        span = max(abs(p2), abs(p98), 1e-4)  # чтобы не было нуля

        im2 = ax2.imshow(delta_sw,
                         cmap='RdBu_r',
                         origin='lower',
                         extent=(0, nx, 0, ny),
                         interpolation='bilinear',
                         vmin=-span, vmax=span,
                         aspect='equal')
        ax2.contour(delta_sw, levels=[0.0], colors='k', linewidths=0.5, origin='lower', extent=(0, nx, 0, ny))
        ax2.set_aspect('equal', adjustable='box')
        ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax2.set_title(f'∆Sw (текущая – начальная){title_suffix}')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig) # Закрываем фигуру, чтобы не потреблять память
        
        print(f"Графики сохранены в файл {filename}") 