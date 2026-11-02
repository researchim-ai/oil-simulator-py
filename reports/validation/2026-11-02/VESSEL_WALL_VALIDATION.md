# Тепловая инерция стенок: проверка 2026-11-02

## Изменение

Добавлена явная сосредоточенная теплоёмкость стенки, монолитно связанная
с компонентными балансами, объёмом и энергией PR-смеси. Энергия стенки
входит в контрольные точки и общий аудит; поток стенка–флюид внутренний,
поток окружение–стенка внешний. Нагреватель может отключиться при сохранении
теплового выбега от накопленной энергии стенки.

## Проверки

`tests/test_surface_vessel_wall.py`: 10 passed, 18.67 s.
Проверяются закрытая система, независимое решение двух теплоёмкостей
обратным Эйлером, сходимость к аналитическому матричному решению при
уменьшении шага, отключённый UA, повреждённое состояние и модель,
смешанная сеть со стенками и без, двухфазная смесь, тепловой выбег после
защиты, локализация температуры стенки, команды окружения и точный restart.

CLI-пример `configs/surface_vessel_network_wall.json`: два связанных аппарата,
три периода, 10 принятых подшагов. Полный JSON совпал с результатом
остановки после первого периода и продолжения из файла.
Максимальная абсолютная невязка энергии шага: 4.110e-11 J;
максимальная невязка энергии стенки: 2.826e-11 J.
Эти величины проверяют дискретный баланс, а не погрешность физической модели.

Команды воспроизведения:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_surface_vessel_wall.py
python scripts/run_vessel_network.py configs/surface_vessel_network_wall.json --output results/vessel_wall_20261102/full.json
python scripts/run_vessel_network.py configs/surface_vessel_network_wall.json --stop-after 1 --output results/vessel_wall_20261102/partial.json
python scripts/run_vessel_network.py configs/surface_vessel_network_wall.json --checkpoint results/vessel_wall_20261102/partial.json --output results/vessel_wall_20261102/resumed.json
```

Артефакты: `results/vessel_wall_20261102/` — JSON запусков, контрольные суммы
исходников/конфига в `validation.json`, журналы тестов и сборки документации.
Численные артефакты остаются в игнорируемом каталоге результатов.

## Область результата

CPU reference, постоянные явные Cw/UA, равномерная стенка, заданная температура
окружения. PR-смесь с постоянными идеальными Cp и kij, максимум две фазы.
Нет геометрического расчёта UA, распределённой теплопроводности, воды,
инерции привода или GPU-реализации. Пластовый FIM и Norne не изменялись
и этим протоколом повторно не валидировались.

## Общая регрессия

`420 passed, 23 skipped, 2 warnings in 283.99s (0:04:43)` в наборе `tests/test_surface*.py`,
`tests/test_fim*surface*.py`, `tests/test_fim_facility_coupling.py`,
`tests/test_fim_blackoil_active.py`, `tests/test_fim_blackoil_valve.py`
с `-m "not slow"`, CUDA скрыта. GPU-проверки пропущены.
Десять проверок стенки входят в это число и отдельно не суммируются.
Ruff check/format, `git diff --check` и строгая сборка MkDocs прошли.
