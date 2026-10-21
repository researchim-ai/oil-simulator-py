"""Сравнение с эталонными расчётами Eclipse и OPM Flow (SPE1, SPE9) по векторам SUMMARY deck."""
import os
import sys

import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from compare_eclipse import compare  # noqa: E402
from simulator.ecl_io import EclSummary  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

DATA = os.path.join(ROOT, "tests", "data")


def _run(case, deck, base):
    refs = {"eclipse": EclSummary(os.path.join(DATA, case, "eclipse", base)),
            "opm": EclSummary(os.path.join(DATA, case, "opm", base))}
    summary, _ = run_deck(os.path.join(DATA, case, deck), device=torch.device("cpu"))
    return compare(summary, refs)


def _check(rows, tol):
    """tol(ключ) -> допуск на среднюю ошибку/масштаб относительно каждого эталона."""
    bad = [(r["vector"], name, r[name]) for r in rows for name in ("eclipse", "opm")
           if name in r and r[name] > tol(r["vector"])]
    assert not bad, bad


@pytest.mark.slow
def test_spe1_matches_eclipse_and_opm():
    rows = _run("spe1", "SPE1CASE1.DATA", "SPE1CASE1")
    assert len(rows) > 30
    _check(rows, lambda key: 0.01)


@pytest.mark.slow
def test_spe9_matches_eclipse_and_opm():
    rows = _run("spe9", "SPE9_CP.DATA", "SPE9_CP")
    assert len(rows) > 200

    def tol(key):
        kw = key.split(":")[0]
        if kw[0] in "FB" or kw == "WBHP":
            return 0.005
        return 0.02

    _check(rows, tol)
