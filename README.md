# oil-simulator-py

<p align="center"><img src="src/imgs/oil-sim.png" alt="Oil Simulator" width="70%"></p>

Открытый гидродинамический симулятор пласта на Python и PyTorch. Полностью неявный трёхфазный black-oil на GPU
читает DATA-файл Eclipse и считает модель целиком: corner-point сетку с разломами, жирный газ, регионы, скважины
по историческому расписанию.

Полная проверка **Norne** (44 431 активная ячейка, 36 скважин, 3312 суток) на RTX 3090 Ti:
накопленная нефть отличается от Eclipse 2014.2 на **+0.0015%**, вода на **+0.497%**, газ на **−0.057%**.
Профиль по умолчанию `strict`: CNV `1e-3`, MB `1e-7`, без локального ослабления.

Первый строгий прогон занял **25 мин 10 с** с отключением CUDA-графа CPR после нехватки памяти.
После исправления пулов памяти полный `strict` прошёл на отдельной **RTX 3090 за 15 мин 58 с**,
графы работали до конца. Это измерение до последующего исправления порядка соединений ствола;
после включения временных параметров `TUNING` отдельный контрольный прогон занял **14 мин 35 с**.
Последняя правка усреднения RESV проверена отдельно короткими тестами.
Прежние **9 минут** относятся к опубликованному прогону `relaxed`.
В контроле с `TUNING` остаются локальные отличия по накопленной воде B-1H (−13.05%) и E-4AH (+5.41%);
два малых эталонных теста пока вне прежних допусков. Подробнее — [Norne](docs/NORNE.md).

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
