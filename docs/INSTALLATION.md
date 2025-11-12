# Установка и настройка окружения

## 1. Клонирование репозитория

```bash
git clone https://github.com/<your-org>/oil-simulator-py-impes.git
cd oil-simulator-py-impes
```

## 2. Виртуальное окружение

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
```

## 3. Основные зависимости

```bash
pip install -r requirements.txt
```

Файл `requirements.txt` включает:

- `torch` (CUDA-сборка)
- `numpy`, `scipy`, `numba` (поддержка тестов и вспомогательных утилит)
- `tqdm`, `matplotlib`

## 4. Дополнительные зависимости для документации

```bash
pip install -r requirements-docs.txt
```

Состав:

- `mkdocs`, `mkdocs-material`, `mkdocs-print-site-plugin`, `mkdocs-mermaid2-plugin`
- `pymdown-extensions`, `markdown`, `Pillow`
- `playwright` (для рендеринга PDF через Chromium)

После установки Playwright выполните:

```bash
playwright install chromium
```

## 5. Проверка CUDA

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA не обнаружена!"
print("GPU:", torch.cuda.get_device_name(0))
print("Версия CUDA:", torch.version.cuda)
print("Версия PyTorch:", torch.__version__)
PY
```

## 6. Разработка и тесты

Установите проект в editable-режиме:

```bash
pip install -e .
```

Запуск основных тестов:

```bash
pytest tests/test_amg_pressure_solver.py
pytest tests/test_impes_3phase.py
```

## 7. Генерация документации

```bash
mkdocs build
python scripts/export_pdf.py
```

Подробнее см. [Генерация PDF](HOW_TO_GENERATE_PDF.md).

## 8. Обновление зависимостей

- Для PyTorch регулярно проверяйте совместимость с вашей версией CUDA.
- Документация требует Playwright ≥ 1.40 (Chromium). При обновлении Playwright повторите `playwright install`.

## 9. Типичные проблемы

- **`ModuleNotFoundError: solver`** — добавьте `PYTHONPATH=src` при запуске.
- **`torch.OutOfMemoryError`** — снижайте размер задачи, увеличивайте агрессивность коэрснинга, очищайте кэш GPU (`torch.cuda.empty_cache()`), проверяйте отсутствие других процессов на GPU.
- **`playwright` не находит Chromium** — выполните `playwright install chromium --with-deps`.

## Полезные ссылки

- [Документация PyTorch CUDA](https://pytorch.org/get-started/locally/)
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [Playwright Python](https://playwright.dev/python/)

