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
    refs = {name: EclSummary(os.path.join(DATA, case, name, base)) for name in ("eclipse", "opm")
            if os.path.exists(os.path.join(DATA, case, name, base + ".SMSPEC"))}
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
        return 0.025

    _check(rows, tol)


@pytest.mark.slow
def test_vappars_wet_gas_matches_opm():
    rows = _run("vappars", "VAPPARS-01.DATA", "VAPPARS-01")
    assert len(rows) >= 4
    _check(rows, lambda key: 0.02)


@pytest.mark.slow
def test_vfpprod_thp_control_matches_opm():
    rows = _run("vfpprod", "SPE1CASE1_METRIC_VFP1.DATA", "SPE1CASE1_METRIC_VFP1")
    assert any(r["vector"] == "WTHP:PROD" for r in rows)
    _check(rows, lambda key: 0.025 if key.startswith("BGSAT") else 0.01)


@pytest.mark.slow
def test_vfpinj_thp_control_matches_opm():
    rows = _run("vfpinj", "VFPINJ_WCONINJE-01.DATA", "VFPINJ_WCONINJE-01")
    assert any(r["vector"] == "WTHP:INJ1" for r in rows)
    # BWSAT в переходной зоне у INJ2 расходится с OPM на 0.0045 уже в начальном равновесии
    _check(rows, lambda key: 0.05 if key.startswith("BWSAT") else 0.01)


@pytest.mark.slow
def test_gaslift_matches_opm():
    """GASLIFT-04 (opm-tests): газлифт LIFTOPT/WLIFTOPT, THP, групповая цель, WEFAC. Сверка до 213 сут: дальше
    OPM закрывает скважины без газлифта и не открывает их по WTEST (в его эталоне нули до конца)."""
    ref = EclSummary(os.path.join(DATA, "gaslift", "opm", "GASLIFT-04"))
    summary, _ = run_deck(os.path.join(DATA, "gaslift", "GASLIFT-04.DATA"), device=torch.device("cpu"),
                          max_days=213)
    rows = compare(summary, {"opm": ref})
    assert any(r["vector"] == "FGLIR" for r in rows)

    def tol(key):
        kw = key.split(":")[0]
        if kw.endswith("GLIR"):
            return 0.2          # шаг газлифта 40 000: OPM доводит газлифт до цели не кратно шагу
        if kw in ("FWPR", "FWIR"):
            return 0.08
        return 0.01

    _check(rows, tol)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["NETWORK-01", "NETWORK-01-REROUTE"])
def test_network_matches_opm(case):
    """Расширенная сеть (BRANPROP/NODEPROP/NETBALAN) из opm-tests: давление узлов — ограничение THP скважин,
    групповая цель, VREP закачка. REROUTE — смена маршрута шлейфов и ветка 9999 без потерь."""
    ref = EclSummary(os.path.join(DATA, "network", "opm", case))
    summary, _ = run_deck(os.path.join(DATA, "network", case + ".DATA"), device=torch.device("cpu"))
    rows = compare(summary, {"opm": ref})
    assert any(r["vector"] == "GPR:PROD" for r in rows)

    def tol(key):
        kw = key.split(":")[0]
        if kw in ("FWIR", "WWIR"):
            return 0.08         # VREP: закачка по отбору прошлого шага, первый интервал после пуска
        if kw.endswith("SAT"):
            return 0.03         # ячейка у нагнетательной — вслед за закачкой
        if "REROUTE" in case and kw[0] in "FWG" and kw[-2:] in ("PR", "PT"):
            return 0.03         # первые сутки после смены маршрута: OPM балансирует сеть по ходу Ньютона
        return 0.01

    _check(rows, tol)
