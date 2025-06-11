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

        im1 = ax1.imshow(p_slice / 1e6, cmap='jet', origin='lower', aspect='auto')
        ax1.set_title(f'Давление (МПа){title_suffix}')
        ax1.set_xlabel('Ячейка X')
        ax1.set_ylabel('Ячейка Y')
        fig.colorbar(im1, ax=ax1)

        vmin = float(s_slice.min())
        vmax = float(s_slice.max())
        if abs(vmax - vmin) < 1e-4:
            # если изменений почти нет, задаём небольшой диапазон вокруг значения
            vmin = max(0.0, vmin - 0.05)
            vmax = min(1.0, vmax + 0.05)
        im2 = ax2.imshow(s_slice, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
        ax2.set_title(f'Насыщенность водой{title_suffix}')
        ax2.set_xlabel('Ячейка X')
        ax2.set_ylabel('Ячейка Y')
        fig.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig) # Закрываем фигуру, чтобы не потреблять память
        
        print(f"Графики сохранены в файл {filename}") 