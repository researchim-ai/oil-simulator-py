# MODEL1 (MSW)

Источник: [OPM/opm-tests](https://github.com/OPM/opm-tests), каталог `model1`, коммит `562d54e`.
Лицензия данных: Open Database License, содержимое — Database Contents License (см. заголовки файлов deck).

- `BASE_MODEL_1.DATA`, `MSW_MODEL_1.DATA`, `include/` — deck без изменений; MSW-вариант задаёт скважины через WELSEGS/COMPSEGS.
- `opm/{BASE,MSW}_MODEL_1.{SMSPEC,UNSMRY}` — расчёт OPM Flow (`opm-simulation-reference/flow`).

Дополнительный эталон `opm_tuning/BASE_MODEL_1` рассчитан независимо 29 октября 2026 в
OPM Flow 2026.04 с `--enable-tuning=true`, чтобы применялся тот же предел шага TUNING=5 суток,
который соблюдает наш решатель. Сохранены полный PRT, команда, версия и SHA-256 входов/выходов
в manifest.json. Исходные эталоны не заменены; тест BASE_MODEL_1 выбирает `opm_tuning/`.

При неизменённых допусках средняя нормированная ошибка FGOR — 1.243% вместо 3.141%,
WWPT PROD-3 — 1.405% вместо 4.115%, WGOR PROD-3 — 3.390% вместо 7.099%.
Разница эталонов существенна: оценивать вклад алгоритма отдельно от шага по времени обязательно.
