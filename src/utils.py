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