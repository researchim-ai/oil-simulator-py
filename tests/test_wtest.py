"""Закрытие скважины симулятором (не может работать на ограничении BHP) и её проверки WTEST / открытие WELOPEN."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.eclipse_deck import EclipseCase  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")
FIRST_YEAR = "31 28 31 30 31 30 31 31 30 31 30 31 \n"
DEAD = "WCONPROD\n 'PROD' 'OPEN' 'BHP' 5* 9000 /\n/\nTSTEP\n 31 31 /\n"
BACK = "WCONPROD\n 'PROD' 'OPEN' 'ORAT' 20000 4* 1000 /\n/\n"
TAIL = "TSTEP\n 31 31 31 31 /\n"
MAX_DAYS = 365 + 62 + 124


def _deck(tmp_path, extra, name):
    text = open(SPE1).read()
    text = text.replace(FIRST_YEAR, FIRST_YEAR + "/\n" + DEAD + BACK + extra + TAIL + "TSTEP\n", 1)
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _events(info):
    return [(round(t), name, ev) for t, name, ev in info["well_events"]]


def test_wtest_parsing(tmp_path):
    case = EclipseCase(_deck(tmp_path, "WTEST\n 'PROD' 60 'PE' 2 /\n/\n", "P.DATA"))
    assert case.wtest_steps[0] == {}
    assert case.wtest_steps[14] == {"PROD": dict(interval=60.0, reasons="PE", tests=2)}
    off = EclipseCase(_deck(tmp_path, "WTEST\n 'PROD' 60 'P' /\n/\nTSTEP\n 1 /\nWTEST\n 'PROD' 0 'P' /\n/\n", "Q.DATA"))
    assert "PROD" in off.wtest_steps[14] and off.wtest_steps[15] == {}
    with pytest.raises(ValueError):
        EclipseCase(_deck(tmp_path, "WTEST\n 'PROD' 60 'X' /\n/\n", "R.DATA"))


def test_dead_well_is_closed_and_reopened_by_wtest(tmp_path):
    s, info = run_deck(_deck(tmp_path, "WTEST\n 'PROD' 60 'P' /\n/\n", "W.DATA"), max_days=MAX_DAYS)
    t0 = 365 + 31
    assert _events(info) == [(t0, "PROD", "close"), (t0 + 62, "PROD", "test"), (t0 + 93, "PROD", "reopen")]
    n = int(np.searchsorted(s["TIME"], t0)) + 1
    # закрыта до проверки (через 60 сут), на шаге проверки снова добывает на целевом дебите
    assert s["FOPR"][n] == 0.0 and s["FOPR"][n + 1] == 0.0
    assert np.isclose(s["FOPR"][n + 2], 20000.0, rtol=1e-6) and np.isclose(s["FOPR"][n + 3], 20000.0, rtol=1e-6)


def test_closed_well_stays_closed_without_wtest_and_welopen_reopens(tmp_path):
    s, info = run_deck(_deck(tmp_path, "", "N.DATA"), max_days=MAX_DAYS)
    t0 = 365 + 31
    assert _events(info) == [(t0, "PROD", "close")]
    n = int(np.searchsorted(s["TIME"], t0)) + 1
    assert np.all(s["FOPR"][n:] == 0.0)

    s2, info2 = run_deck(_deck(tmp_path, "WELOPEN\n 'PROD' 'OPEN' /\n/\n", "O.DATA"), max_days=MAX_DAYS)
    assert _events(info2) == [(t0, "PROD", "close")]
    assert s2["FOPR"][n] == 0.0 and np.allclose(s2["FOPR"][n + 1:], 20000.0, rtol=1e-6)
