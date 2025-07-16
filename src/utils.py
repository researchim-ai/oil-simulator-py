import os
import re
import imageio


def create_animation(image_folder, output_path, fps=5):
    """
    Создает GIF-анимацию из серии .png изображений в указанной папке.

    Изображения сортируются на основе числа в их имени файла (например, ..._step_10.png).

    :param image_folder: Папка с исходными изображениями.
    :param output_path: Путь для сохранения итогового GIF-файла.
    :param fps: Количество кадров в секунду для анимации.
    """
    images = []

    # Поддерживаем два формата имён: frame_0001.png и foo_step_12.png
    regex = re.compile(r'(?:^|_)(?:step|frame)_(\d+)\.png$')

    # Собираем файлы и их номера шагов
    file_tuples = []
    print(f"Поиск изображений в {image_folder}...")
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            match = regex.search(filename)
            if match:
                step_number = int(match.group(1))
                file_tuples.append((step_number, os.path.join(image_folder, filename)))

    # Сортируем файлы по номеру шага, чтобы анимация была последовательной
    file_tuples.sort()

    if not file_tuples:
        print(f"В папке {image_folder} не найдено подходящих изображений для анимации.")
        return

    print(f"Найдено {len(file_tuples)} кадров. Создание GIF...")

    # Читаем отсортированные изображения
    for _, filepath in file_tuples:
        images.append(imageio.imread(filepath))

    # Сохраняем GIF
    # Используем fps, переданный из конфига
    imageio.mimsave(output_path, images, fps=fps)
    print(f"Анимация успешно сохранена в {output_path}")


class PIDController:
    """Расширенный экспоненциальный PID-контроллер для адаптации шага времени.

    * anti-windup интеграла;
    * low-pass фильтр производной;
    * ограничение относительного изменения `dt` за шаг;
    * масштабирование через `exp(u)` с границами `scale_min/scale_max`.

    Метод `update(error)` возвращает множитель `scale`, такой что
    `dt_new = clamp(dt * scale, dt_min, dt_max)`.
    """

    def __init__(self,
                 kp: float = 0.6,
                 ki: float = 0.3,
                 kd: float = 0.0,
                 dt_min: float = 60.0,
                 dt_max: float = 86400.0 * 10,
                 scale_min: float = 0.1,
                 scale_max: float = 10.0,
                 integral_limit: float | None = None,
                 derivative_alpha: float = 1.0,
                 max_scale_change: float | None = None):
        # PID коэффициенты
        self.kp, self.ki, self.kd = kp, ki, kd

        # Ограничения по самому `dt`
        self.dt_min, self.dt_max = dt_min, dt_max

        # Ограничения по коэффициенту масштабирования
        self.scale_min, self.scale_max = scale_min, scale_max

        # Anti-windup лимит для интеграла (abs)
        self.integral_limit = integral_limit

        # Фильтр производной: alpha=1 → без фильтра
        self.derivative_alpha = max(0.0, min(1.0, derivative_alpha))

        # Макс. относительное изменение dt за шаг (например, 2.0 = ×2)
        self.max_scale_change = max_scale_change if (max_scale_change is None or max_scale_change > 1.0) else None

        # Внутренние состояния
        self.integral = 0.0
        self.prev_error: float | None = None
        self._last_derivative = 0.0  # для фильтра
        self._prev_scale = 1.0

    # ------------------------------------------------------------------
    # Свойства (используются в тестах/отладке)
    # ------------------------------------------------------------------
    @property
    def last_derivative(self) -> float:
        return self._last_derivative

    # ------------------------------------------------------------------
    # Основной шаг контроллера
    # ------------------------------------------------------------------
    def update(self, error: float) -> float:
        """Возвращает множитель `scale` (>0) для обновления `dt`.

        Вызывать **раз** за расчётный шаг.
        """
        # --- Интегральная часть с anti-windup ------------------------
        self.integral += error
        if self.integral_limit is not None:
            lim = abs(self.integral_limit)
            self.integral = max(-lim, min(lim, self.integral))

        # --- Производная с low-pass фильтром -------------------------
        raw_derivative = 0.0 if self.prev_error is None else (error - self.prev_error)
        alpha = self.derivative_alpha
        self._last_derivative = alpha * raw_derivative + (1.0 - alpha) * self._last_derivative
        self.prev_error = error

        # --- PID регулятор ------------------------------------------
        u = self.kp * error + self.ki * self.integral + self.kd * self._last_derivative

        # Экспоненциальное управление + жёсткая обрезка
        from math import exp
        scale = exp(u)
        scale = max(self.scale_min, min(self.scale_max, scale))

        # --- Ограничиваем темп изменения dt --------------------------
        if self.max_scale_change is not None:
            max_up = self.max_scale_change
            max_down = 1.0 / self.max_scale_change
            scale = max(max_down, min(max_up, scale))

        self._prev_scale = scale
        return scale

    # ------------------------------------------------------------------
    # Утилита: обрезка самого dt по физическим границам
    # ------------------------------------------------------------------
    def clamp(self, dt: float) -> float:
        return max(self.dt_min, min(self.dt_max, dt)) 