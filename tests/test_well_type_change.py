"""Смена типа скважины (закачка ↔ добыча) и фазы закачки по расписанию; накопленные объёмы раздельно."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.eclipse_deck import EclipseCase  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")
FIRST_YEAR = "31 28 31 30 31 30 31 31 30 31 30 31 \n"
MAX_DAYS = 365 + 92


def _deck(tmp_path, change, name):
    text = open(SPE1).read()
    text = text.replace(FIRST_YEAR, FIRST_YEAR + "/\n" + change + "TSTEP\n", 1)
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _well(info, name):
    return next(w for w in info["solver"].wells.wells if w.name == name)


def test_injector_becomes_producer(tmp_path):
    base, binfo = run_deck(SPE1, max_days=365)
    gi_year = _well(binfo, "INJ").cum_inj["gas"]
    path = _deck(tmp_path, "WCONPROD\n 'INJ' 'OPEN' 'BHP' 5* 3000 /\n/\n", "T.DATA")
    case = EclipseCase(path)
    assert case.well_type["INJ"] == "injector"
    assert case.control_cfg("INJ", case.steps[12][1]["INJ"])["type"] == "producer"
    s, info = run_deck(path, max_days=MAX_DAYS)
    n = int(np.searchsorted(s["TIME"], 365.0)) + 1
    assert np.allclose(s["FOPR"][1:n], base["FOPR"][1:n], rtol=1e-10)
    w = _well(info, "INJ")
    assert w.type == "producer"
    # закачка за первый год сохранена, после смены — только добыча
    assert np.isclose(w.cum_inj["gas"], gi_year, rtol=1e-10)
    assert w.cum_prod["gas"] > 0 and w.last_surface_rates["gas"] > 0
    assert np.all(s["FGIR"][n:] == 0.0) and np.all(s["FGPR"][n:] > s["FOPR"][n:])
    assert np.isclose(w.bhp_mpa * 1e6 / case.u["p"], 3000.0)


def test_injected_phase_change(tmp_path):
    path = _deck(tmp_path, "WCONINJE\n 'INJ' 'WATER' 'OPEN' 'RATE' 5000 1* 9014 /\n/\n", "P.DATA")
    s, info = run_deck(path, max_days=MAX_DAYS)
    n = int(np.searchsorted(s["TIME"], 365.0)) + 1
    w = _well(info, "INJ")
    assert w.type == "injector" and w.injected_phase == "water"
    assert np.all(s["FGIR"][n:] == 0.0) and np.allclose(s["FWIR"][n:], 5000.0, rtol=1e-4)
    assert w.cum_inj["gas"] > 0 and np.isclose(w.cum_inj["water"], 5000.0 * 90 * 0.158987294928, rtol=1e-4)


def test_zero_rate_target_stops_well(tmp_path):
    path = _deck(tmp_path, "WCONINJE\n 'INJ' 'GAS' 'OPEN' 'RATE' 0 1* 9014 /\n/\n", "R0.DATA")
    case = EclipseCase(path)
    assert case.control_cfg("INJ", case.steps[12][1]["INJ"])["control_type"] == "shut"
    s, info = run_deck(path, max_days=MAX_DAYS)
    n = int(np.searchsorted(s["TIME"], 365.0)) + 1
    assert np.all(s["FGIR"][n:] == 0.0) and np.all(s["FGIR"][1:n] > 0)
    assert info["substeps"] < 60


def test_zero_history_producer_then_injector(tmp_path):
    hist = "WCONHIST\n 'INJ' 'OPEN' 'ORAT' 0 0 0 /\n/\n"
    text = open(SPE1).read().replace("WCONINJE", hist + "TSTEP\n 31 /\nWCONINJE", 1)
    path = tmp_path / "Z.DATA"
    path.write_text(text)
    case = EclipseCase(str(path))
    assert case.well_type["INJ"] == "injector"
    s, info = run_deck(str(path), max_days=31 + 59)
    assert s["FGIR"][1] == 0.0 and np.all(s["FGIR"][2:] > 0)
