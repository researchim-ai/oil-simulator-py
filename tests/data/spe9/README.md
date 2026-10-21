# SPE9 (corner-point)

Источник: [OPM/opm-tests](https://github.com/OPM/opm-tests), каталог `spe9`, коммит `6a3257e`.
Лицензия данных: Open Database License (см. заголовки файлов deck).

- `SPE9_CP.DATA`, `SPE9.GRDECL` (COORD/ZCORN/ACTNUM), `PERMVALUES.DATA` — deck 24×25×15, 26 скважин, 900 суток.
- `eclipse/SPE9_CP.{SMSPEC,UNSMRY}` — расчёт Eclipse (`eclipse-simulation`).
- `opm/SPE9_CP.{SMSPEC,UNSMRY}` — расчёт OPM Flow (`opm-simulation-reference/flow_legacy`).
  В этой сводке нет единиц измерения, а FGOR записан в ст.м³/ст.м³; `scripts/compare_eclipse.py` это учитывает.
