"""
Исторический режим (DATES, WCONHIST, WCONINJH, WELOPEN) на SPE1: дебиты, снятые с прогноза, подаются
как история и должны воспроизводиться; закрытие по WELOPEN обнуляет отбор.
"""
import datetime
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.eclipse_run import run_deck  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")
DAYS = 365
MON = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


@pytest.fixture(scope="module")
def base():
    s, _ = run_deck(SPE1, max_days=DAYS)
    return s


def _history_deck(tmp_path, s, mode, shut_step=None):
    text = open(SPE1).read()
    out = [text[:text.index("WCONPROD")]]
    t0 = datetime.datetime(2015, 1, 1)
    for n in range(1, len(s["TIME"])):
        qo, qw, qg = s["WOPR:PROD"][n], s["WWPR:PROD"][n], s["WGPR:PROD"][n]
        out.append(f"WCONHIST\n 'PROD' 'OPEN' '{mode}' {qo:.6f} {qw:.6f} {qg:.6f} /\n/\n")
        out.append(f"WCONINJH\n 'INJ' 'GAS' 'OPEN' {s['WGIR:INJ'][n]:.6f} /\n/\n")
        if n == shut_step:
            out.append("WELOPEN\n 'PROD' 'SHUT' /\n/\n")
        d = t0 + datetime.timedelta(days=float(s["TIME"][n]))
        out.append(f"DATES\n {d.day} '{MON[d.month - 1]}' {d.year} /\n/\n")
    out.append("END\n")
    path = tmp_path / f"SPE1_HIST_{mode}.DATA"
    path.write_text("".join(out))
    return str(path)


def test_wconhist_orat_reproduces_forecast(base, tmp_path):
    s, _ = run_deck(_history_deck(tmp_path, base, "ORAT"))
    assert np.allclose(s["TIME"], base["TIME"])
    for key in ("WOPR:PROD", "WGPR:PROD", "WBHP:PROD", "WGIR:INJ"):
        a, b = s[key][1:], base[key][1:]
        assert np.max(np.abs(a - b) / np.abs(b).max()) < 2e-3, key


def test_wconhist_resv_close_to_forecast(base, tmp_path):
    s, _ = run_deck(_history_deck(tmp_path, base, "RESV"))
    a, b = s["WOPR:PROD"][1:], base["WOPR:PROD"][1:]
    # RESV-цель пересчитывается из истории при среднем давлении пласта, а не у забоя
    assert np.max(np.abs(a - b) / b.max()) < 0.05


def test_welopen_shut_stops_production(base, tmp_path):
    s, _ = run_deck(_history_deck(tmp_path, base, "ORAT", shut_step=4))
    assert s["WOPR:PROD"][4] == pytest.approx(0.0, abs=1e-9)
    assert s["WOPR:PROD"][3] == pytest.approx(base["WOPR:PROD"][3], rel=2e-3)
    assert s["WOPR:PROD"][5] > 0
    assert s["WOPT:PROD"][5] < base["WOPT:PROD"][5]
