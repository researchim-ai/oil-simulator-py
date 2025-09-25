# Руководство пользователя

### Установка
- Требования: Python 3.10+, PyTorch (CUDA, если GPU), NumPy, SciPy.
- Установка:
  - `pip install -r requirements.txt`
  - Проверьте CUDA: `python -c "import torch; print(torch.cuda.is_available())"`.

### Быстрый старт
- Одношаговый прогон FI 100×100×100 (GPU):
  - `python bench/bench.py --nx 100 --ny 100 --nz 100 --mode fi --steps 1`
- Много шагов и расширенные лимиты:
  - `python bench/bench.py --nx 100 --ny 100 --nz 100 --mode fi --steps 10 --newton 8 --jfnk 240 --geo_cycles 3 --geo_pre 3 --geo_post 3 --geo-tol 1e-6 --geo-max-iter 15`

### Конфигурации и параметры
- Параметры CLI перекрывают значения из JSON‑конфига и `sim_params`.
- Примеры конфигов: `configs/examples/*`.
- Важные флаги см. в `docs/CONFIG_REFERENCE.md`.

### Логирование
- Терминал (ANSI‑цвета): ключевые события Ньютона/GMRES/CPR/AMG.
- JSONL: укажите `--log-json-dir logs/json` (или `sim_params["log_json_dir"]`).
  - События: `newton_iter`, `gmres_done`, `pressure_cap`, `ls_try`, `ls_accept`, `step_summary`.
  - Каждая строка — отдельный JSON‑объект (удобно для парсинга).

### Интерпретация метрик
- `||F||`, `||F||_scaled`: невязка и её масштабированный вариант.
- `ρ_v` (VC): снижение нормы за V‑цикл AMG (типично ~0.18–0.20).
- `[CPR P/S]`: нормы по давлению/насыщенности в hat‑пространстве.
- `mass err`, `vol err`: относительные ошибки баланса за шаг (см. NUMERICS).

### Производительность
- GPU по умолчанию (если доступен). Для CPU‑режима: установите `CUDA_VISIBLE_DEVICES=""` или используйте сборку PyTorch без CUDA.
- Детальные VC/RBGS‑логи — только для отладки; для быстрых прогонов держите `OIL_DEBUG=0` (по умолчанию).

### Воспроизводимость
- Зафиксируйте конфиг, размеры сетки, версию PyTorch/CUDA.
- Сохраняйте JSONL‑логи; для сравнения прогонов используйте одинаковые `sim_params`.



