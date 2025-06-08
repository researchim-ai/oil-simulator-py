import torch

class Well:
    """
    Класс для представления одной скважины.
    """
    def __init__(self, name, well_type, coordinates, reservoir_dimensions, rate):
        """
        Инициализация скважины.

        :param name: Имя скважины (str).
        :param well_type: Тип скважины ('producer' или 'injector') (str).
        :param coordinates: Кортеж (i, j, k) с индексами ячейки скважины.
        :param reservoir_dimensions: Кортеж (nx, ny, nz) с размерами пласта.
        :param rate: Дебит в м^3/сутки (float). Положительный для нагнетания, конвенция сделает его отрицательным для добычи.
        """
        self.name = name
        self.well_type = well_type
        
        assert len(coordinates) == 3, "Координаты должны быть в формате (i, j, k)"
        self.i, self.j, self.k = coordinates
        
        nx, ny, nz = reservoir_dimensions
        
        assert 0 <= self.i < nx and 0 <= self.j < ny and 0 <= self.k < nz, f"Координаты скважины {name} выходят за пределы пласта."

        # Конвертируем дебит в м^3/с и устанавливаем знак
        # Добыча < 0, Нагнетание > 0
        if self.well_type == 'producer':
            self.rate_si = -rate / 86400.0
        else: # injector
            self.rate_si = rate / 86400.0

        # Рассчитываем одномерный индекс ячейки
        self.cell_idx = self.i * ny * nz + self.j * nz + self.k

        print(f"  Создана скважина '{self.name}':")
        print(f"    Тип: {self.well_type}, Расположение: ({self.i}, {self.j}, {self.k}), Дебит: {rate} м^3/сутки")


class WellManager:
    """
    Класс для управления всеми скважинами в симуляции.
    """
    def __init__(self):
        """
        Инициализация менеджера скважин.
        """
        self.wells = []
        print("Создан менеджер скважин.")

    def add_well(self, well):
        """
        Добавляет скважину в список.
        """
        self.wells.append(well)
        print(f"  > Скважина '{well.name}' добавлена в менеджер.")

    def get_wells(self):
        return self.wells
