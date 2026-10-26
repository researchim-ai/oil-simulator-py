# oil-simulator-py

<p align="center"><img src="src/imgs/oil-sim.png" alt="Oil Simulator" width="70%"></p>

Открытый гидродинамический симулятор пласта на Python и PyTorch. Полностью неявный трёхфазный black-oil на GPU
читает DATA-файл Eclipse и считает модель целиком: corner-point сетку с разломами, жирный газ, регионы, скважины
по историческому расписанию.

Месторождение **Norne** (44 431 активная ячейка, 36 скважин, 3312 суток) считается целиком за **9 минут** на одной
RTX 3090 Ti. Накопленная нефть отличается от Eclipse 2014.2 на −0.04%, вода на +0.6%, газ на −0.06%.

- Сайт: [`web/`](web/index.html) — описание симулятора и [результаты Norne](web/norne.html)
- Документация: [`docs/`](docs/index.md), HTML и PDF собираются MkDocs (см. ниже)

## Возможности

- **Black-oil FIM:** нефть, вода, газ, Rs и Rv, переключение переменных ячейки, аналитический Якобиан, Ньютон
  с допусками CNV и баланса массы.
- **GPU-решатель:** FGMRES + CPR (классический AMG по давлению, блочный Якоби), float64.
- **Deck Eclipse:** COORD/ZCORN, разломы, NNC, PINCH, MINPV, MULTFLT, регионы PVT/SATNUM/EQLNUM, EQUIL,
  ENDSCALE, гистерезис, THPRES, VAPPARS.
- **Скважины:** WCONPROD/WCONINJE/WCONHIST/WCONINJH (RESV), COMPDAT, WELOPEN, WTEST, группы, переток в стволе.
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

[`docs/ROADMAP.md`](docs/ROADMAP.md): ускорение (float32, пакетный прогон), VFP и устьевые ограничения, WECON,
многосегментные скважины, вывод в форматах Eclipse, градиенты для оптимизации.
