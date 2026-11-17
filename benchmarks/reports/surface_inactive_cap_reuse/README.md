# Повторное использование неактивных ограничений дебита

Приёмка оптимизации: **True**. Связанный изотермический black-oil
пласт, 1024 ячейки / 16 добывающих скважин / 30 суток; квазистационарная
сборная сеть и площадка. Не полная нестационарная PR-сеть и не Norne.

Реальных решений FIM: 13321 → 11892.
Снижение числа решений: 10.727%.
Наблюдавшееся время CPU: 2381.499 → 1989.933 с.
Хост не изолирован: отношение времени не является чистым аппаратным замером.

Конфигурации различаются только `control.reuse_inactive_caps`; допуски FIM,
временного контроля и поиска скобки сохранены. Бюджеты регрессии заданы
в `scripts/check_facility_cap_reuse.py` до получения результата. Проверены
конечные поля, накопления и скважины на 60 общих отчётных датах.

Число точек поиска и реальных решений FIM разделено. Источник сертификата —
только сошедшаяся недопустимая проба того же поиска; допустимые состояния
рассчитываются полностью. Проверены переключения команд, перезапуск,
откат, аудит сертификатов, CPU и RTX 3090 Ti (память ограничена 8%).

Контрольная траектория: `../surface_bhp_time_control/accepted_candidate/`.
Сравнение с архивной более строгой траекторией также проходит только при
`accepted=true` соответствующих разделов `comparison.json`. Исходники этих
старых и новых прогонов отличаются: это не новая квалификация уточнения
на одной версии исходников и не экспериментальная валидация физики.

Повтор проверки:

```sh
PYTHONPATH=src:scripts python scripts/check_facility_cap_reuse.py \
  benchmarks/reports/surface_bhp_time_control/accepted_candidate \
  benchmarks/reports/surface_inactive_cap_reuse/optimized \
  --days 30 --output /tmp/inactive_cap_regression.json
```

Конфиг выполнения сохранён в `optimized/trajectory.json.gz`, исходники —
в `executed_sources.tar.gz`. Контрольные суммы — `artifact_sha256.json`.

Готовый конфиг: `configs/fim_blackoil_facility_depletion_fast.json`; после разрешения PVT-пути точно совпадает с конфигом принятого прогона. Исходный adaptive-конфиг сохранён.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src:scripts python scripts/run_fim_facility_forecast.py \
  --config configs/fim_blackoil_facility_depletion_fast.json \
  --days 30 --report-days .5 --device cpu --host-memory-limit-mib 2048 \
  --output /tmp/depletion_fast_month
```
