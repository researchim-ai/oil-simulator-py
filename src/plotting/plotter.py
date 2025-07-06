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
        # Насыщенность – адаптивная шкала: если изменения <5%, зумим
        # ------------------------------------------------------------------
        vmin = float(s_slice.min())
        vmax = float(s_slice.max())
        diff = vmax - vmin

        # Очень маленькие изменения (<2 %) – показываем окошко 0.02, чтобы увидеть фронт
        if diff < 0.02:
            vmin = vmin
            vmax = min(1.0, vmin + 0.02)
        elif diff < 0.05:  # изменения 2–5 % – окно 0.05
            vmax = min(1.0, vmin + 0.05)
        elif diff < 1e-4:  # совсем нет изменений
            vmin, vmax = 0.0, 1.0

        # Подстраховка на случай vmin == vmax (одинаковая насыщенность)
        if abs(vmax - vmin) < 1e-6:
            vmin = max(0.0, vmin - 0.01)
            vmax = min(1.0, vmax + 0.01)

        # Используем более контрастную палитру и bilinear для плавности
        im2 = ax2.imshow(s_slice,
                         cmap='plasma',
                         origin='lower',
                         extent=(0, nx, 0, ny),
                         interpolation='bilinear',
                         vmin=vmin, vmax=vmax,
                         aspect='equal')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title(f'Насыщенность водой{title_suffix}')
        ax2.set_xlabel('Ячейка X')
        ax2.set_ylabel('Ячейка Y')
        fig.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig) # Закрываем фигуру, чтобы не потреблять память
        
        print(f"Графики сохранены в файл {filename}") 