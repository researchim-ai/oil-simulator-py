"""Соединения скважин (COMPDAT): CF/Kh/направление/Ro по соединению, SHUT, WELOPEN по соединениям, вскрытие по расписанию."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.eclipse_deck import EclipseCase  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402
from simulator.fim_wells import connection_wi, peaceman_wi  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")
BASE_COMPDAT = "\t'PROD'\t10\t10\t3\t3\t'OPEN'\t1*\t1*\t0.5 /\n"
FIRST_YEAR = "31 28 31 30 31 30 31 31 30 31 30 31 \n"


def _deck(tmp_path, compdat=None, after_year=None, name="C.DATA"):
    text = open(SPE1).read()
    if compdat is not None:
        assert BASE_COMPDAT in text
        text = text.replace(BASE_COMPDAT, compdat, 1)
    if after_year is not None:
        text = text.replace(FIRST_YEAR, FIRST_YEAR + "/\n" + after_year + "TSTEP\n", 1)
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_connection_wi_formulas():
    kx, ky, kz, dx, dy, dz, rw = 2e-13, 1e-13, 2e-14, 100.0, 80.0, 10.0, 0.1
    z = dict(ijk=(0, 0, 0), dir="Z", rw=rw, skin=1.0)
    assert math.isclose(connection_wi(z, kx, ky, kz, dx, dy, dz), peaceman_wi(kx, ky, dx, dy, dz, rw, 1.0), rel_tol=1e-14)
    x = dict(ijk=(0, 0, 0), dir="X", rw=rw, skin=0.0)
    r0 = 0.28 * math.sqrt(math.sqrt(kz / ky) * dy ** 2 + math.sqrt(ky / kz) * dz ** 2) / ((kz / ky) ** 0.25 + (ky / kz) ** 0.25)
    expect = 2 * math.pi * math.sqrt(ky * kz) * dx / math.log(r0 / rw)
    assert math.isclose(connection_wi(x, kx, ky, kz, dx, dy, dz), expect, rel_tol=1e-14)
    given = dict(ijk=(0, 0, 0), dir="Z", rw=rw, skin=0.5, kh=3e-12, ro=20.0)
    assert math.isclose(connection_wi(given, kx, ky, kz, dx, dy, dz), 2 * math.pi * 3e-12 / (math.log(20.0 / rw) + 0.5))
    assert connection_wi(dict(given, wi=1.5e-12), kx, ky, kz, dx, dy, dz) == 1.5e-12


def test_shut_connection_and_explicit_cf_reproduce_base(tmp_path):
    base, _ = run_deck(SPE1, max_days=400)
    case = EclipseCase(SPE1)
    case.build()
    wi = case.well_connections("PROD", 0)
    assert len(wi) == 1
    # CF = WI в единицах FIELD (rb·cP/сут/psi) — тот же индекс, заданный явно
    _, _, wm, _ = case.build()
    w_si = [w for w in wm.wells if w.name == "PROD"][0]
    from simulator.fim_wells import FIMWells
    res, fluid, wm, _ = case.build()
    fw = FIMWells(wm, res, fluid, case.dims, "cpu")
    wi_si = float(fw.conn_wi[fw.conn_well == [w.name for w in wm.wells].index("PROD")][0])
    u = case.u
    cf = wi_si / ((u["liq"] / 86400.0) * 1e-3 / u["p"])
    compdat = (f"\t'PROD'\t10\t10\t2\t2\t'SHUT'\t1*\t1*\t0.5 /\n"
               f"\t'PROD'\t10\t10\t3\t3\t'OPEN'\t1*\t{cf:.15e}\t0.5 /\n")
    alt, _ = run_deck(_deck(tmp_path, compdat), max_days=400)
    # пути вычислений различаются на уровне округления (CF ↔ WI, лишнее закрытое соединение), а адаптивный шаг
    # может выбрать соседнюю траекторию — сравнение с допуском дискретизации по времени
    for key in ("FOPR", "FGPR", "WBHP:PROD"):
        assert np.allclose(alt[key][1:], base[key][1:], rtol=1e-3), key
    assert w_si.name == "PROD"


def test_late_perforation_and_welopen_connection(tmp_path):
    add = "COMPDAT\n 'PROD' 10 10 2 2 'OPEN' 1* 1* 0.5 /\n/\n"
    case = EclipseCase(_deck(tmp_path, after_year=add))
    flags = lambda s: [c["open"] for c in case.well_connections("PROD", s)]
    assert [c["ijk"][2] for c in case.well_connections("PROD", 0)] == [2, 1]
    assert flags(0) == [True, False] and flags(11) == [True, False] and flags(12) == [True, True]

    shut = add + "TSTEP\n 31 /\nWELOPEN\n 'PROD' 'SHUT' 0 0 2 /\n/\n"
    case2 = EclipseCase(_deck(tmp_path, after_year=shut, name="C2.DATA"))
    fl2 = lambda s: [c["open"] for c in case2.well_connections("PROD", s)]
    assert fl2(12) == [True, True] and fl2(13) == [True, False]

    base, _ = run_deck(SPE1, max_days=500)
    late, _ = run_deck(_deck(tmp_path, after_year=add, name="C3.DATA"), max_days=500)
    n_year = int(np.searchsorted(base["TIME"], 365.0, side="right"))
    assert np.allclose(late["FOPR"][1:n_year], base["FOPR"][1:n_year], rtol=1e-6)
    assert np.allclose(late["WBHP:PROD"][1:n_year], base["WBHP:PROD"][1:n_year], rtol=1e-6)
    assert np.abs(late["WBHP:PROD"][n_year + 1:] - base["WBHP:PROD"][n_year + 1:]).max() > 1.0
