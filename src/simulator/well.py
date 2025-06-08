class Well:
    """
    Класс для представления одной скважины.
    """
    def __init__(self, name, location, well_type, rate):
        """
        Инициализация скважины.

        :param name: Имя скважины (str).
        :param location: Кортеж (i, j, k) - координаты ячейки в сетке.
        :param well_type: Тип скважины ('producer' или 'injector').
        :param rate: Дебит (для producer - положительное число, для injector - отрицательное), м^3/сутки.
        """
        self.name = name
        self.i, self.j, self.k = location
        self.type = well_type
        self.rate = rate # м^3/сутки

        # Конвертируем расход в м^3/секунду для расчетов
        self.rate_si = rate / (24 * 3600)

        print(f"  Создана скважина '{self.name}':")
        print(f"    Тип: {self.type}, Расположение: ({self.i}, {self.j}, {self.k}), Дебит: {self.rate} м^3/сутки")


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
