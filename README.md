# Симулятор нефтяного месторождения

Высокопроизводительный симулятор для моделирования многофазной фильтрации в пористой среде с использованием Python и PyTorch.

## Цель

Создать open-source альтернативу коммерческим продуктам для гидродинамического моделирования.

## Математическая модель

В основе симулятора лежит решение системы дифференциальных уравнений в частных производных, описывающих течение двух несжимаемых флюидов (нефти и воды) в пористой среде. Для решения этой системы мы используем стандартный для индустрии метод **IMPES (Implicit Pressure, Explicit Saturation)**, который включает неявное решение для давления и явное для насыщенности.

## Визуализация результатов

По завершении симуляции создается файл `final_results.png`, содержащий две карты для центрального среза пласта:

1.  **Карта давлений (слева):**
    -   Изображает итоговое распределение давления в мегапаскалях (МПа).
    -   Цветовая схема `jet` помогает визуализировать зоны: синие области соответствуют низкому давлению (вокруг добывающих скважин), а красные — высокому (вокруг нагнетательных).

2.  **Карта насыщенности водой (справа):**
    -   Показывает долю порового объема, занятую водой (от 0 до 1).
    -   Цветовая схема `viridis` наглядно демонстрирует фронт вытеснения: желто-зеленые цвета показывают распространение нагнетаемой воды, а темно-синие — зоны, все еще насыщенные нефтью.

## План развития
-   [x] Учет относительных фазовых проницаемостей (ОФП) - *Реализовано с использованием модели Кори.*
-   [ ] Векторизация и перенос вычислений на GPU
-   [ ] Усложнение моделей скважин и граничных условий


## Требования

Для работы симулятора необходимы следующие библиотеки:
- `torch`
- `numpy`
- `matplotlib`
- `tqdm`
- `scipy`
- `pytest` (для запуска тестов)

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone <URL-вашего-репозитория>
   cd oil-simulator-py
   ```

2. Установите все необходимые зависимости с помощью `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   *Примечание: `torch` будет установлен с поддержкой CUDA, если у вас есть совместимый GPU и настроено окружение. В противном случае будет использоваться CPU-версия.*

## Запуск симуляции

Симулятор запускается из командной строки с указанием файла конфигурации. В директории `configs/` находятся примеры конфигураций.

Для запуска симуляции с 2D-моделью среднего размера выполните:
```bash
python src/main.py --config configs/medium_2d.json
```
Результаты (числовые данные в `.txt` и графики в `.png`) будут сохранены в директорию `results/`.

## Запуск тестов

Для проверки корректности работы симулятора и отслеживания регрессий используются тесты, написанные с помощью `pytest`.

Для запуска тестов выполните из корневой директории проекта:
```bash
pytest
```
Тесты запустят симуляцию с конфигурацией `configs/test_config.json` и сравнят итоговые карты давления и насыщенности с эталонными файлами, хранящимися в `tests/test_data/`.