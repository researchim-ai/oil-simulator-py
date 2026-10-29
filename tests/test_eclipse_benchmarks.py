"""Сравнение с эталонными расчётами Eclipse и OPM Flow (SPE1, SPE9) по векторам SUMMARY deck."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from compare_eclipse import compare  # noqa: E402
from simulator.ecl_io import EclSummary  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

DATA = os.path.join(ROOT, "tests", "data")


def _run(case, deck, base, opm_dir="opm"):
    dirs = {"eclipse": "eclipse", "opm": opm_dir}
    refs = {name: EclSummary(os.path.join(DATA, case, folder, base)) for name, folder in dirs.items()
            if os.path.exists(os.path.join(DATA, case, folder, base + ".SMSPEC"))}
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
    # The legacy reference ignored TUNING (1-day cap); compare to an
    # independent OPM run honoring the same deck settings. Keep both fixtures.
    rows = _run("vfpinj", "VFPINJ_WCONINJE-01.DATA", "VFPINJ_WCONINJE-01", opm_dir="opm_tuning")
    assert any(r["vector"] == "WTHP:INJ1" for r in rows)
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
            return 0.08         # переходный процесс при запуске закачки по VREP
        if kw.endswith("SAT"):
            return 0.03         # ячейка у нагнетательной — вслед за закачкой
        if "REROUTE" in case and kw[0] in "FWG" and kw[-2:] in ("PR", "PT"):
            return 0.03         # короткий переходный процесс после смены маршрута
        return 0.01

    _check(rows, tol)

    if case == "NETWORK-01-REROUTE":
        # Средняя ошибка за всю историю не должна скрывать неправильную долю
        # PROD2 сразу после смены маршрута (прежде ~895 вместо ~1320 м³/сут).
        ours = float(np.interp(62.0, summary["TIME"], summary["WOPR:PROD2"]))
        expected = float(np.interp(62.0, ref.days, ref.get("WOPR", "PROD2")))
        assert abs(ours - expected) < 0.02 * abs(expected)


@pytest.mark.slow
def test_wecon_wtest_network_matches_opm():
    """NETWORK-01-WTEST (opm-tests): WECON — минимальный дебит нефти 2000 м³/сут, WTEST раз в сутки по причине E.
    PROD2 циклически закрывается и открывается на пробу; мгновенные дебиты зависят от фазы цикла в момент отчёта,
    поэтому строго сверяются накопленные."""
    ref = EclSummary(os.path.join(DATA, "network", "opm", "NETWORK-01-WTEST"))
    summary, info = run_deck(os.path.join(DATA, "network", "NETWORK-01-WTEST.DATA"), device=torch.device("cpu"))
    ev = [(name, kind) for _, name, kind in info["well_events"]]
    assert ("PROD2", "close") in ev and ("PROD2", "test") in ev and ("PROD2", "reopen") in ev
    rows = compare(summary, {"opm": ref})

    def tol(key):
        kw = key.split(":")[0]
        if kw.endswith("PT") or kw.endswith("IT"):
            return 0.02
        if kw.endswith("SAT"):
            return 0.03
        return 0.2

    _check(rows, tol)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["MSW_MODEL_1", "BASE_MODEL_1"])
def test_model1_msw_matches_opm(case):
    """
    MODEL1 (opm-tests): STONE1, разные DENSITY по PVTNUM, SWATINIT, WCONHIST RESV. MSW_MODEL_1 — те же скважины
    многосегментными (WELSEGS/COMPSEGS). У PROD-3 барьер между слоями 4 и 5: нефть слоя 5 через ствол уходит
    в газовую шапку (переток смесью ствола; у многосегментной — смесью своего сегмента).
    """
    # BASE_MODEL_1 requires an OPM reference with TUNING enabled, as here.
    # Original references remain available for diagnosing timestep effects.
    ref_dir = "opm_tuning" if case == "BASE_MODEL_1" else "opm"
    rows = _run("model1", case + ".DATA", case, opm_dir=ref_dir)
    assert any(r["vector"] == "WOPR:PROD-3" for r in rows)

    def tol(key):
        if case == "BASE_MODEL_1" and key.endswith("PROD-3") and key[:4] in ("WGOR", "WOPR"):
            return 0.06         # первый месяц после пуска PROD-3: OPM считает его меньшими шагами
        return 0.03

    _check(rows, tol)


@pytest.mark.slow
def test_wcutback_gor_spe1(tmp_path):
    """WCUTBACK на SPE1 (эталона нет: OPM Flow WCUTBACK не поддерживает): после прорыва газа GOR PROD выше
    2 тыс. фут³/барр., и в конце каждого шага её дебит нефти умножается на 0.8, пока предел нарушен."""
    src = open(os.path.join(DATA, "spe1", "SPE1CASE1.DATA")).read().replace("SUMMARY\n", "SUMMARY\nFOPT\n", 1)
    key = "'PROD' 'OPEN' 'ORAT' 20000 4* 1000 /\n/\n"
    assert key in src
    (tmp_path / "BASE.DATA").write_text(src)
    (tmp_path / "CUT.DATA").write_text(src.replace(key, key + "\nWCUTBACK\n 'PROD' 1* 2.0 2* 0.8 /\n/\n"))
    base, _ = run_deck(str(tmp_path / "BASE.DATA"), device=torch.device("cpu"), max_days=1100)
    cut, info = run_deck(str(tmp_path / "CUT.DATA"), device=torch.device("cpu"), max_days=1100)
    times = [t for t, name, kind in info["well_events"] if name == "PROD" and kind == "cutback"]
    assert len(times) >= 5
    for t, q, gor in zip(cut["TIME"][1:], cut["WOPR:PROD"][1:], cut["WGOR:PROD"][1:]):
        k = sum(tc < t - 1e-6 for tc in times)
        assert q == pytest.approx(20000 * 0.8 ** k, rel=1e-4)
        assert (gor > 2.0) == any(abs(tc - t) < 1e-6 for tc in times)
    assert cut["FOPT"][-1] < 0.9 * base["FOPT"][-1]
