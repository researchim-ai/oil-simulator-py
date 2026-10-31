# oil-simulator-py

<p align="center"><img src="src/imgs/oil-sim.png" alt="Oil Simulator" width="70%"></p>

Открытый гидродинамический симулятор пласта на Python и PyTorch. Полностью неявный трёхфазный black-oil на GPU
читает DATA-файл Eclipse и считает модель целиком: corner-point сетку с разломами, жирный газ, регионы, скважины
по историческому расписанию.

Полный строгий **Norne** (44 431 активная ячейка, 36 скважин, 3312 суток)
проверен на **RTX 3090 Ti за 10 мин 02 с**: CNV `1e-3`, MB `1e-7`, без локального ослабления.
Накопленная нефть относительно Eclipse 2014.2 отличается на **−0,073%**, вода **+0,175%**, газ **−0,093%**.
Остаточные расхождения отдельных скважин и зависимость от временной сетки сохраняются.

Объединённое сглаживание Якоби в CPR дало около **1,5%** сокращения времени
в двух парных продолжениях на одной RTX 3090 Ti, сохранив результат побитово.
Разницу полного времени с прежними 10 мин 28 с на обычной RTX 3090 нельзя
приписывать только коду. Проверки: 302 CPU-теста прошли, 85 пропущены;
дополнительный набор с CUDA — 101 проверка.
Методика — [FGMRES/CPR и Norne](docs/NORNE_CPR_20261031.md),
предыдущие измерения — [GPU–CPU обмены](docs/NORNE_GPU_PACKETS_20261031.md),
ограничения и диагностика — [Norne](docs/NORNE_WATER_DIAGNOSTICS.md).
Наземная часть включает существующую VFP-сеть и отдельный физический решатель
однофазных жидкостных трубопроводов: кольца, гидростатика, трение и местные сопротивления.
Пример: `python scripts/run_surface_network.py configs/surface_liquid_ring.json`.
Добавлены насосы с квадратичной характеристикой, изменяемой скоростью, КПД,
мощностью на валу и проверкой NPSH; пример — `configs/surface_liquid_booster.json`.
Есть обратные клапаны с автоматическим открытием/закрытием и регулятор
скорости по давлению с явным выходом на пределы скорости.
Добавлены табличные карты напора/КПД/NPSH, жидкостные дроссели и идеальная
ступенчатая black-oil сепарация с балансом компонентов (Rs, без Rv).
Сквозной пример: `python scripts/run_surface_facility.py configs/surface_facility.json` —
сепарация, передача отделённой воды в трубную сеть, насос, дроссель и отчёт ограничений.
Контрольный набор наземных моделей и FIM: более 200 проверок прошли;
сохранённый пример перезапускается без изменения физических результатов.
Это стационарная изотермическая модель с идеальным регулированием отвода воды;
добавлен газовый тракт сухого идеального газа: многоступенчатая компрессия,
охлаждение, смешение, изотермическая экспортная труба и общие ограничения мощности.
Пример: `python scripts/run_surface_facility.py configs/surface_gas_facility.json`.
Добавлены ограничение подачи под мощность площадки, расписание установившихся
режимов, остановка/запуск, накопленные объёмы/энергия и проверяемый перезапуск:
`python scripts/run_surface_schedule.py configs/surface_capacity_limited_facility.json configs/surface_operation_schedule.json`.
Диспетчеризация подачи остаётся внешней. Отдельно добавлена неявная связь
водяных добывающих FIM-скважин с пассивной физической сетью, с полным
межскважинным Якобианом и исключением Шура. Пример:
`python scripts/run_fim_surface.py configs/fim_water_surface.json --output /tmp/fim_surface.json`.
Водяная связь поддерживает аналитическую `oil_water`. Добавлено CPU/CUDA-подключение
black-oil FIM к физическому дереву сбора с общим межскважинным Якобианом и Шуром 3×3:
`python scripts/run_fim_blackoil_surface.py configs/fim_blackoil_surface.json --output /tmp/blackoil_surface.json`.
Для GPU добавьте `--device cuda`: пласт, скважинные блоки и Шур считаются на CUDA,
сеть по умолчанию использует CPU-эталон. Опция `--network-backend torch_device`
переносит PVT, квадратуру и autograd сети на устройство пласта; обход дерева
и скалярный поиск давления остаются на CPU. На проверенной двухскважинной сети
гибрид быстрее, поэтому он оставлен по умолчанию. CUDA-граф свойств сохраняется.
Обмены сгруппированы в пакеты; ответ сети загружается асинхронно через pinned memory.
Статистика сходимости и отчёты скважин также читаются пакетно, без изменения допусков.
Пока требуются положительная добыча, перфорации на отметке BHP и WEFAC=1;
распределённый ствол, остановленные ветви и активное оборудование в FIM ещё впереди.
Добавлена отдельная многофазная black-oil сеть сбора: смешение компонентов,
выделение растворённого газа по PVT, неявная инверсия давления и полные
производные по составу притоков. Сквозной пример со ступенями сепарации,
водяной сетью и экспортом газа:
`python scripts/run_surface_gathering.py configs/surface_gathering_facility.json`.
Поддержано направленное дерево горизонтальных/восходящих труб без проскальзывания
фаз, с одной PVT-изотермой. Для положительно работающих скважин есть ограниченная
неявная CPU/CUDA-связь с black-oil FIM, описанная выше.
Режимы и перезапуск:
`python scripts/run_surface_schedule.py configs/surface_gathering_facility.json configs/surface_gathering_schedule.json --gathering`.
Произвольная газовая сеть, real-gas EOS, удержание/проскальзывание фаз и многофазная связь
физической сети с пластом — следующие этапы
([план](docs/ROADMAP.md), [конфигурация](docs/CONFIG_REFERENCE.md#физическая-жидкостная-сеть)).

- Сайт: [`web/`](web/index.html) — описание симулятора и [результаты Norne](web/norne.html)
- Документация: [`docs/`](docs/index.md), HTML и PDF собираются MkDocs (см. ниже)

## Возможности

- **Black-oil FIM:** нефть, вода, газ, Rs и Rv, переключение переменных ячейки, аналитический Якобиан, Ньютон
  с допусками CNV и баланса массы.
- **GPU-решатель:** FGMRES + CPR (классический AMG по давлению, блочный Якоби), float64.
- **Deck Eclipse:** COORD/ZCORN, разломы, NNC, PINCH, MINPV, MULTFLT, регионы PVT/SATNUM/EQLNUM, EQUIL,
  ENDSCALE, гистерезис, THPRES, VAPPARS, VFPPROD, VFPINJ, газлифт LIFTOPT/WLIFTOPT/GLIFTOPT, WEFAC, поверхностная сеть BRANPROP/NODEPROP/GRUPNET.
- **Скважины:** WCONPROD/WCONINJE/WCONHIST/WCONINJH (RESV), COMPDAT, WELOPEN, WTEST, группы, переток в стволе, ограничение THP.
- **Сравнение:** чтение сводок Eclipse/OPM, `scripts/compare_eclipse.py`.
- **Агенты:** `iter_deck`, чекпоинты, среда gymnasium `DeckEnv`.
- **IMPES** на JSON-конфигах — учебные задачи и стресс-тест на миллион ячеек.

## Установка

```bash
git clone https://github.com/researchim-ai/oil-simulator-py.git
cd oil-simulator-py
pip install -r requirements.txt
```

Для GPU нужна CUDA-сборка PyTorch.

## Расчёт deck Eclipse

```python
import sys, torch
sys.path.insert(0, "src")
from simulator.eclipse_run import run_deck

summary, info = run_deck("tests/data/spe1/SPE1CASE1.DATA", device=torch.device("cuda"))
print(summary["FOPT"][-1])
```

Сравнение со сводками Eclipse и OPM:

```bash
python scripts/compare_eclipse.py tests/data/spe1/SPE1CASE1.DATA \
    --ref eclipse=tests/data/spe1/eclipse/SPE1CASE1 --ref opm=tests/data/spe1/opm/SPE1CASE1
```

## Norne

```bash
git clone --filter=blob:none https://github.com/OPM/opm-tests.git
NORNE_DIR=opm-tests/norne NORNE_OUT=results/norne python scripts/run_norne.py
NORNE_DIR=opm-tests/norne NORNE_OUT=results/norne python scripts/norne_figures.py
```

Подробности — [`docs/NORNE.md`](docs/NORNE.md).

## IMPES на JSON-конфигах

```bash
PYTHONPATH=src python src/main.py --config configs/medium_2d.json
PYTHONPATH=src python src/main.py --config configs/mega_3phase_million.json   # 1 млн ячеек, 24 ГБ GPU
```

## Тесты

```bash
pytest
NORNE_DIR=opm-tests/norne pytest tests/test_norne_regression.py   # нужна GPU и данные Norne
```

## Документация

```bash
pip install -r requirements-docs.txt
playwright install chromium
mkdocs build                  # HTML в site/
python scripts/export_pdf.py  # PDF в site/pdf/OilSimulatorDocs.pdf
```

## План

[`docs/ROADMAP.md`](docs/ROADMAP.md): ускорение (float32, пакетный прогон), неявные многосегментные скважины
с клапанами, вывод в форматах Eclipse, градиенты для оптимизации.

Активная водяная сеть теперь связана с FIM: насосы, дроссели, обратные клапаны,
регулирование давления и точные межскважинные производные, включая изменение оборотов.
Пласт и оператор Шура могут работать на CUDA; малая гидравлическая сеть работает на CPU.

```bash
python scripts/run_fim_surface.py configs/fim_water_surface_active.json --device cuda --output /tmp/fim_active.json
python scripts/run_surface_schedule.py configs/surface_station_facility.json configs/surface_station_schedule.json --output /tmp/station.json
```

Параллельная насосная станция выбирает состав с учётом доступности, мощности, NPSH
и минимального расхода; расписание сохраняет накопления и состояние для перезапуска.
Границы модели и проверка: [активное оборудование](docs/SURFACE_ACTIVE_EQUIPMENT.md).

В многофазном black-oil дереве добавлены управляемые распределённые сопротивления
и неявный регулятор давления открытием ветви, включая производные FIM и команды
расписания площадки. Примеры: `configs/fim_blackoil_surface_regulated.json`,
`configs/surface_gathering_active.json`. Это однородное приближение K/a²,
а не модель многофазного насоса или специализированного критического штуцера.
[Физика и воспроизведение](docs/BLACKOIL_ACTIVE_EQUIPMENT.md).

В black-oil сети доступен отдельный **локальный жидкостный клапан**:
учитывает PVT, проверяет выделение газа внутри сжатого сечения и запас
относительно давления пара, имеет производные для FIM и регулятора.
Свободный газ и flashing отклоняются.
[Модель, ограничения и команды запуска](docs/BLACKOIL_LOCAL_VALVE.md).

Газовый экспорт получил конечную пропускную способность через `GasExportValve`:
докритический/критический расход сухого идеального газа, расчёт открытия,
обратные задачи давления, ограничение подачи и команды расписания.
Пример: `python scripts/run_surface_schedule.py configs/surface_gas_valve_facility.json configs/surface_gas_valve_schedule.json --output /tmp/gas_schedule.json`.
[Модель, область применимости и проверки](docs/GAS_EXPORT_VALVE.md).

Добавлен внешний контур **пласт → сбор → подготовка → экспорт → ограничения
дебитов скважин**. Каждая проба пересчитывает FIM из исходного состояния;
ограничения площадки меняют фактическую добычу, а не масштабируют готовый результат.
Включение: `simulation.coupled_facility`; пример —
`configs/fim_blackoil_facility_coupled.json`.
[Алгоритм, проверка и ограничения](docs/FIM_FACILITY_COUPLING.md).
