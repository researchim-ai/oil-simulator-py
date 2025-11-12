# Быстрый старт

Этот раздел поможет настроить окружение и выполнить первый расчёт с AMG-решателем давления.

## 1. Предварительные требования

- **Python** ≥ 3.10
- **CUDA** 11.8+ и драйвер NVIDIA, совместимый с выбранной версией PyTorch
- **GPU** с ≥ 16 ГБ памяти (для миллионного кейса требуется 24 ГБ)
- Установленные зависимости проекта (см. [Установка](INSTALLATION.md))

## 2. Создание окружения

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Для генерации документации потребуется дополнительный набор пакетов:

```bash
pip install -r requirements-docs.txt
```

## 3. Проверка GPU

```bash
python - <<'PY'
import torch
print("CUDA доступна:", torch.cuda.is_available())
print("Устройство:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

## 4. Запуск тестового расчёта (2D)

```bash
PYTHONPATH=src python src/main.py --config configs/medium_2d.json
```

Результаты сохранятся в директории `results/` вместе с графиками и npz-файлом состояний.

## 5. Запуск 3D-симуляции с AMG

```bash
PYTHONPATH=src python src/main.py --config configs/impes_large_multiwell_3d.json
```

Параметры AMG находятся в блоке `simulation.amg` выбранного конфигурационного файла. При первом шаге строится AMG-иерархия; последующие шаги используют кэш и выполняют только обновление коэффициентов матрицы.

## 6. Миллионный «mega»-кейс (stress-test)

```bash
PYTHONPATH=src python src/main.py --config configs/mega_3phase_million.json
```

Рекомендации:

- Убедитесь, что на GPU свободно не менее 22 ГБ VRAM.
- Для экспресс-проверки уменьшите `total_time_days` и `time_step_days` до 0.05–0.1.
- После первого построения AMG следующий запуск с теми же параметрами проходит значительно быстрее.

## 7. Постобработка

- Графики по давлению и дебитам: `results/<run_id>/*final.png`
- Поля для последующей визуализации: `results/<run_id>/*.npz`
- Лог построения AMG: стандартный вывод (`stdout`) и при необходимости файл в `results`.

## 8. Минимальный сценарий для разработки

```bash
PYTHONPATH=src python tests/test_amg_pressure_solver.py -k classical_amg_quality
```

Тест проверяет, что AMG снижает градиентную компоненту остатка ниже 5·10⁻² за 3 V-цикла.

## Следующие шаги

- Изучите [Архитектуру](ARCHITECTURE.md), чтобы понимать взаимодействие модулей.
- Ознакомьтесь с [Математическими моделями](MATHEMATICS.md) и [Физикой](PHYSICS_MODELS.md) для настройки собственных сценариев.

