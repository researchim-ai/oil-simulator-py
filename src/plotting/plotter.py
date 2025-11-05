import matplotlib.pyplot as plt
import numpy as np

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

    def save_plots(self, pressure, saturation, filename, time_info=None, gas_saturation=None):
        """
        Сохраняет 2D-карты давления и насыщенностей в файл.
        Визуализируется центральный срез по оси Z.
        :param pressure: Массив давления
        :param saturation: Массив водонасыщенности
        :param filename: Имя файла для сохранения
        :param time_info: (Опционально) Строка с информацией о времени (например, "День 50")
        :param gas_saturation: (Опционально) Массив газонасыщенности для трехфазной модели
        """
        # Выбираем Z-срез с максимальной дисперсией насыщенности,
        # чтобы на графике было видно изменения.
        if self.reservoir.nz > 1:
            stds = [float(np.std(saturation[:, :, k])) for k in range(self.reservoir.nz)]
            z_slice_idx = int(np.argmax(stds))
        else:
            z_slice_idx = 0
            
        p_slice = pressure[:, :, z_slice_idx]
        sw_slice = saturation[:, :, z_slice_idx]
        
        # Определяем количество графиков в зависимости от наличия газа
        has_gas = gas_saturation is not None and np.any(gas_saturation > 1e-6)
        
        if has_gas:
            sg_slice = gas_saturation[:, :, z_slice_idx]
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        title_suffix = f" (Срез Z={z_slice_idx})"
        if time_info:
            title_suffix = f" ({time_info}, Срез Z={z_slice_idx})"

        # График давления
        im1 = ax1.imshow(p_slice / 1e6, cmap='jet', origin='lower', aspect='auto')
        ax1.set_title(f'Давление (МПа){title_suffix}')
        ax1.set_xlabel('Ячейка X')
        ax1.set_ylabel('Ячейка Y')
        fig.colorbar(im1, ax=ax1)

        # График водонасыщенности
        vmin_w = float(sw_slice.min())
        vmax_w = float(sw_slice.max())
        if abs(vmax_w - vmin_w) < 1e-4:
            vmin_w = max(0.0, vmin_w - 0.05)
            vmax_w = min(1.0, vmax_w + 0.05)
        im2 = ax2.imshow(sw_slice, cmap='viridis', origin='lower', vmin=vmin_w, vmax=vmax_w, aspect='auto')
        ax2.set_title(f'Насыщенность водой{title_suffix}')
        ax2.set_xlabel('Ячейка X')
        ax2.set_ylabel('Ячейка Y')
        fig.colorbar(im2, ax=ax2)
        
        # График газонасыщенности (если есть газ)
        if has_gas:
            vmin_g = float(sg_slice.min())
            vmax_g = float(sg_slice.max())
            if abs(vmax_g - vmin_g) < 1e-4:
                vmin_g = max(0.0, vmin_g - 0.05)
                vmax_g = min(1.0, vmax_g + 0.05)
            im3 = ax3.imshow(sg_slice, cmap='plasma', origin='lower', vmin=vmin_g, vmax=vmax_g, aspect='auto')
            ax3.set_title(f'Насыщенность газом{title_suffix}')
            ax3.set_xlabel('Ячейка X')
            ax3.set_ylabel('Ячейка Y')
            fig.colorbar(im3, ax=ax3)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig) # Закрываем фигуру, чтобы не потреблять память
        
        print(f"Графики сохранены в файл {filename}") 