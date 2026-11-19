# Полный Norne с резервным поиском поправки

Свежий расчёт 0–3312 суток на RTX 3090, 44 431 активная ячейка, 36 скважин.
Изотермический пласт и скважины, без дополнительной подробной наземной сети.
AMG step, CNV=1e-3, MB=1e-7, relaxed_max_pv_fraction=0,
NORNE_NEWTON_FALLBACK=1, NORNE_CPR_REUSE_GRAPH=0.

## Результат

Полное время 499,453 с (8 мин 19 с). 569 принятых шагов, 3 отклонённых
попытки, 3073 поправки Ньютона на принятых шагах. Выполнено три резервных
повтора; все три тоже отказали, после чего обычный контроллер уменьшил dt.
Число поправок на принятых шагах не включает работу отказавших попыток.
Итоговый check_norne_run.py: passed_recorded_audit. Проверены часы,
сходимость, накопленные балансы, хеши исходников и входов.
Максимальная накопленная относительная ошибка баланса 2,91754e-6.

Все сохранённые поля p/Sw/Sg/So/Rs/PV, их часы, все SUMMARY-векторы,
сетка принятых/отклонённых попыток и числа поправок на принятых шагах
совпали ТОЧНО с `.scratch/norne_full_newton/run` (8 мин 31 с).
Это подтверждает восстановление траектории после трёх неудачных резервных
попыток на данном тесте. Равенство полей относится к сохранённым снимкам,
не к каждому промежуточному полю каждой итерации.

Накопленная добыча против сохранённого Eclipse: нефть −0,07846%,
вода +0,18216%, газ −0,09091%. Это интегральные показатели, не верхняя
граница ошибки отдельных скважин или пространственных полей.

Разница времени 499,453 против 511,281 с — отдельные запуски, не повторный
контролируемый A/B. Ускорение не заявляется: резервный алгоритм добавил
работу и не спас ни одного обычного полного шага. Его польза показана
на строгих/измельчённых окнах соседнего отчёта norne_newton_safeguard.
Режим остаётся выключенным по умолчанию.

## Сопряжённая наземная часть

CPU-регрессия существующих тестов test_fim_blackoil_surface,
test_fim_blackoil_active, test_fim_facility_coupling,
test_fim_accepted_well_energy: 35 passed, 3 skipped (CUDA).
Она подтверждает работу прежнего сопряжения с новым режимом выключенным,
а НЕ квалификацию backtracking для наземной сети.

В SurfaceCoupledBlackOilWells.assemble активная строка ограничения
rate/surface/bhp_limit выбирается при сборке; могут возникать ошибки
гидравлической области допустимости и обновляться surface_report/кэши
производных. Текущий поиск рассчитан на обычные BlackOilWells без сети.
Простое снятие проверки типа/наличия сети не является реализацией поддержки.
Для неё необходимы отдельные проверки области допустимости проб, merit
согласованной активной системы и транзакционного отката данных сети.
Запрет остаётся явным.

## Артефакты и воспроизведение

Полные поля, checkpoint и frozen source snapshot находятся в
`.scratch/norne_full_safeguard/run`. Здесь сохранены компактные результаты,
manifest, итоговый аудит и сравнение. Запуск:

```bash
CUDA_VISIBLE_DEVICES=GPU-9ade3f1f-7ead-7504-f762-07be60234d31 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/norne_full_safeguard NORNE_DIR=.scratch/ot2/norne NORNE_OUT=.scratch/norne_full_safeguard/run NORNE_DAYS=3312 NORNE_DEVICE=cuda:0 NORNE_THREADS=1 NORNE_CPR_REUSE_GRAPH=0 NORNE_AMG_REBUILD=step NORNE_NEWTON_FALLBACK=1 /home/user/miniconda3/bin/python scripts/validate_norne.py
PYTHONPATH=src:scripts /home/user/miniconda3/bin/python scripts/check_norne_run.py .scratch/norne_full_safeguard/run --deck-dir .scratch/ot2/norne --previous-run .scratch/norne_full_newton/run
```

Для повторения нужен новый пустой NORNE_OUT. Коммит и push не выполнялись.
