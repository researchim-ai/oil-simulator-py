# SPE9 (corner-point)

Источник: [OPM/opm-tests](https://github.com/OPM/opm-tests), каталог `spe9`, коммит `6a3257e`.
Лицензия данных: Open Database License (см. заголовки файлов deck).

- `SPE9_CP.DATA`, `SPE9.GRDECL` (COORD/ZCORN/ACTNUM), `PERMVALUES.DATA` — deck 24×25×15, 26 скважин, 900 суток.
- `eclipse/SPE9_CP.{SMSPEC,UNSMRY}` — расчёт Eclipse (`eclipse-simulation`).
- `opm/SPE9_CP.{SMSPEC,UNSMRY}` — расчёт OPM Flow (`opm-simulation-reference/flow_legacy`).
  В этой сводке нет единиц измерения, а FGOR записан в ст.м³/ст.м³; `scripts/compare_eclipse.py` это учитывает.

Известное отличие: на 370 сутки (первый отчёт после подъёма дебитов в 15 раз на 360 сутки) вода у части скважин
ниже эталона, у PRODU17 — 2100 против 2590 ст.баррелей/сут. Eclipse проходит 360→370 одним шагом в 10 суток
(`SPE9_CP.PRT`), OPM совпадает с ним до 1 барреля/сут; у нас этот шаг дробится до 0.36 суток. При меньшем шаге
отличие растёт (шаг ≤ 0.5 суток — 1950), то есть оно от сглаживания переходного процесса большим шагом в эталоне.
К 400 суткам отличие уходит. Состояние пласта на 360 сутки совпадает с рестартом Eclipse и OPM до тысячных.
