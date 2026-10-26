# VAPPARS-01 (влажный газ, VAPOIL без DRSDT)

Источник: [OPM/opm-tests](https://github.com/OPM/opm-tests), каталог `vappars`, коммит `562d54e`.
Лицензия данных: Open Database License (см. заголовок deck).

- `VAPPARS-01.DATA`, `include/` — deck 12×12×7 (corner-point, 288 выклиненных ячеек отсекаются MINPV),
  DISGAS + VAPOIL, VAPPARS 2.5, нагнетание газа в газовую шапку, добывающая скважина на ORAT, 212 суток.
- `opm/VAPPARS-01.{SMSPEC,UNSMRY}` — расчёт OPM Flow (`opm-simulation-reference/flow`).
