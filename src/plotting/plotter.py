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
        # Храним предыдущий срез насыщенности, чтобы показывать
        # приращение ∆Sw между соседними шагами, а не относительно
        # самого первого состояния. Так изменения будут заметны
        # даже при медленной эволюции.
        self._prev_sw = None

    def save_plots(self, pressure, saturation_w, filename, time_info=None, saturation_g=None,
                   pressure_limits: tuple | None = None, show_delta_p: bool = False):
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

        # --------------------------------------------------------------
        # Подготовка figure и заголовка
        # --------------------------------------------------------------
        extra_p_delta = show_delta_p and (hasattr(self, '_prev_pressure') and self._prev_pressure is not None)
        n_panels = 3 if self.reservoir.nz > 1 else 2
        if extra_p_delta:
            n_panels += 1
        fig, axes = plt.subplots(1, n_panels, figsize=(18, 5))
        ax1 = axes[0]

        title_suffix = f" (Срез Z={z_slice_idx})"
        if time_info:
            title_suffix = f" ({time_info}, Срез Z={z_slice_idx})"

        # ------------------------------------------------------------------
        # Давление – квадратные ячейки без «ступенек» DPI
        # ------------------------------------------------------------------
        nx, ny, _ = self.reservoir.dimensions
        # Мягкое сглаживание убирает численный «шум» 5×5 без размывания крупного градиента
        p_mpa = p_slice / 1e6
        p_img = gaussian_filter(p_mpa, sigma=0.7)

        # Фиксированный диапазон, если указан pressure_limits (в МПа)
        if pressure_limits is not None:
            vmin, vmax = pressure_limits
        else:
            vmin, vmax = p_img.min(), p_img.max()

        im1 = ax1.imshow(p_img,
                         cmap='turbo',
                         origin='lower',
                         extent=(0, nx, 0, ny),
                         interpolation='bilinear',
                         vmin=vmin, vmax=vmax,
                         aspect='equal')
        ax1.set_aspect('equal', adjustable='box')
        # Убираем тики – остаётся чистая картинка
        ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax1.set_title(f'Давление (МПа){title_suffix}')
        ax1.set_xlabel('Ячейка X')
        ax1.set_ylabel('Ячейка Y')
        fig.colorbar(im1, ax=ax1)

        # Переставляем индексы панелей
        # Теперь: ax1 – Pressure, ax2 – Water Saturation, ax3 – Gas Saturation (если есть)
        # Indices now variable due to optional delta panel
        idx = 1
        ax2 = axes[idx]
        idx += 1
        if self.reservoir.nz > 1:
            ax3 = axes[idx]
            idx += 1
        else:
            ax3 = None
        ax_delta = axes[idx] if extra_p_delta else None

        # --------------------------------------------------------------
        # Water Saturation (absolute)
        # --------------------------------------------------------------
        s_img = gaussian_filter(s_slice, sigma=0.7)
        # Динамический диапазон — делаем минимальное видимое отличие даже при узком спрэде S_w
        sw_min, sw_max = float(np.min(s_img)), float(np.max(s_img))
        # Если фронт ещё не двинулся, оставляем глобальные пределы, чтобы палитра не схлопнулась
        if abs(sw_max - sw_min) < 1e-4:
            sw_min, sw_max = 0.0, 1.0
        im_sw = ax2.imshow(s_img, cmap='viridis', origin='lower', extent=(0, nx, 0, ny),
                           interpolation='bilinear', vmin=sw_min, vmax=sw_max, aspect='equal')
        ax2.set_title(f'Water Saturation S_w{title_suffix}')
        ax2.set_xlabel('Ячейка X')
        ax2.set_ylabel('Ячейка Y')
        fig.colorbar(im_sw, ax=ax2)

        # --------------------------------------------------------------
        # Gas Saturation (if provided)
        # --------------------------------------------------------------
        if ax3 is not None:
            if saturation_g is not None:
                s_g_slice = saturation_g[:, :, z_slice_idx]
                s_g_img = gaussian_filter(s_g_slice, sigma=0.7)
            else:
                s_g_img = np.zeros_like(s_slice)

            sg_min, sg_max = float(np.min(s_g_img)), float(np.max(s_g_img))
            if sg_max < 1e-6:
                # газа нет – оставляем чёрную картинку на полной шкале
                sg_min, sg_max = 0.0, 1.0
            im_g = ax3.imshow(s_g_img, cmap='magma', origin='lower', extent=(0, nx, 0, ny),
                               interpolation='bilinear', vmin=sg_min, vmax=sg_max, aspect='equal')
            ax3.set_title(f'Gas Saturation S_g{title_suffix}')
            ax3.set_xlabel('Ячейка X')
            ax3.set_ylabel('Ячейка Y')
            fig.colorbar(im_g, ax=ax3)

        # --------------------------------------------------------------
        # ΔP panel (optional)
        # --------------------------------------------------------------
        if extra_p_delta and ax_delta is not None:
            dp = (p_mpa - self._prev_pressure[:, :, z_slice_idx] / 1e6)
            dp_img = gaussian_filter(dp, sigma=0.7)
            dmax = np.max(np.abs(dp_img))
            im_dp = ax_delta.imshow(dp_img, cmap='seismic', origin='lower', extent=(0, nx, 0, ny),
                                    interpolation='bilinear', vmin=-dmax, vmax=dmax, aspect='equal')
            ax_delta.set_title(f'ΔP (МПа){title_suffix}')
            ax_delta.set_xlabel('Ячейка X')
            ax_delta.set_ylabel('Ячейка Y')
            fig.colorbar(im_dp, ax=ax_delta)

        # Обновляем предыдущий Sw, чтобы в будущем можно было снова строить ∆Sw при необходимости
        self._prev_sw = s_slice.copy()
        # Сохраняем текущее давление для следующего шага (для ΔP)
        self._prev_pressure = pressure.copy()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig) # Закрываем фигуру, чтобы не потреблять память
        
        print(f"Графики сохранены в файл {filename}") 