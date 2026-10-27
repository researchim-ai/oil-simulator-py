"""
Чтение deck-файлов Eclipse (*.DATA) и сборка модели для black-oil FIM.

Поддерживается подмножество, достаточное для SPE1/SPE9-подобных моделей:
  общее:   INCLUDE
  RUNSPEC: DIMENS, FIELD | METRIC, OIL/WATER/GAS/DISGAS
  GRID:    DX, DY, DZ, TOPS (декартова сетка, горизонтальные слои) или COORD/ZCORN/ACTNUM (corner-point
           без разломов); PORO, PERMX/Y/Z, NTG, BOX/ENDBOX, EQUALS/COPY/ADD/MULTIPLY
  PROPS:   PVTO | PVDO, PVDG, PVTW, ROCK, DENSITY, SWOF, SGOF, STONE2
  SOLUTION: EQUIL (с капиллярным давлением, инициализация по центрам ячеек), RSVD
  SUMMARY: список запрошенных векторов (F*, W*, B*) сохраняется для отчёта
  SCHEDULE: WELSPECS, COMPDAT, WCONPROD, WCONINJE (в т.ч. смена режимов по расписанию, маски имён), TSTEP, DRSDT
Неподдерживаемые ключевые слова, влияющие на результат, вызывают ошибку, а не игнорируются.
"""
import datetime
import fnmatch
import re
from pathlib import Path

import numpy as np
import torch

from .blackoil_props import DeckPVT, ThreePhaseRelPerm, interp1_lin

# ключевые слова без данных
FLAG_KEYWORDS = {
    "RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE", "END",
    "OIL", "WATER", "GAS", "DISGAS", "VAPOIL", "FIELD", "METRIC", "LAB", "UNIFIN", "UNIFOUT",
    "INIT", "NOECHO", "ECHO", "ENDBOX", "STONE2", "STONE1", "STONE", "FMTOUT", "FMTIN", "NOSIM",
    "NEWTRAN", "OLDTRAN", "IMPES", "NOINSPEC", "NORSSPEC", "ALL", "MSUMLINS", "MSUMNEWT", "SEPARATE",
    "EXCEL", "RUNSUM", "FOPR", "FOPT", "FGOR", "FGPR", "FGPT", "FWPR", "FWPT", "FWCT", "FPR",
}
# ключевые слова, которые читаются, но не влияют на расчёт
IGNORED = {"TITLE", "EQLDIMS", "TABDIMS", "WELLDIMS", "RPTSCHED", "RPTRST", "RPTSOL", "RPTPROPS",
           "RPTGRID", "RPTRUNSP", "NSTACK", "MESSAGES", "TUNING", "SPECGRID", "GRIDFILE", "MAPAXES", "GRIDUNIT",
           "NETWORK"}
# влияют на физику, но пока не поддерживаются
UNSUPPORTED = {"SWFN", "SGFN", "SOF3", "SLGOF", "AQUFETP", "AQUCT", "PDVD",
               "PRESSURE", "SWAT", "SGAS", "RS", "NNC", "EDITNNC",
               "DRVDT", "ROCKTAB", "TRANX", "TRANY", "TRANZ",
               "COMPORD", "OPERATE", "MINPVV", "MINPORV",
               "OLDTRAN", "HYSTER", "LGR", "CARFIN"}
ARRAY_KEYWORDS = {"DX", "DY", "DZ", "TOPS", "PORO", "PERMX", "PERMY", "PERMZ", "NTG", "ACTNUM",
                  "MULTX", "MULTY", "MULTZ", "MULTX-", "MULTY-", "MULTZ-", "MULTPV", "FLUXNUM", "MULTNUM",
                  "SATNUM", "PVTNUM", "EQLNUM", "FIPNUM", "IMBNUM", "SWATINIT",
                  "SWL", "SWCR", "SWU", "SOWCR", "SGL", "SGCR", "SGU", "SOGCR", "KRW", "KRO", "KRG", "PCW", "PCG",
                  "ISWL", "ISWCR", "ISWU", "ISOWCR", "ISGL", "ISGCR", "ISGU", "ISOGCR"}
GRID_SECTIONS = ("RUNSPEC", "GRID", "EDIT")
RAW_KEYWORDS = {"COORD", "ZCORN"}
EDIT_OPS = {"EQUALS", "COPY", "ADD", "MULTIPLY"}

UNITS = {
    "FIELD": dict(p=6894.757293168, L=0.3048, liq=0.158987294928, gas=28.316846592,
                  rho=16.01846337396, comp=1.0 / 6894.757293168),
    "METRIC": dict(p=1e5, L=1.0, liq=1.0, gas=1.0, rho=1.0, comp=1e-5),
}
MD = 9.869233e-16
DAY = 86400.0

_KW_RE = re.compile(r"^([A-Z][A-Z0-9_+-]{0,7})\s*$")
_TOK_RE = re.compile(r"'[^']*'|/|[^\s/']+")


def _strip_comment(line):
    if "--" not in line:
        return line
    if "'" not in line:
        return line[:line.index("--")]
    out, q = [], False
    for i, ch in enumerate(line):
        if ch == "'":
            q = not q
        if not q and line.startswith("--", i):
            break
        out.append(ch)
    return "".join(out)


def _expand(tok):
    if tok.startswith("'"):
        return [tok[1:-1].strip()]
    m = re.fullmatch(r"(\d+)\*(.*)", tok)
    if m:
        n, val = int(m.group(1)), m.group(2)
        return [None] * n if val == "" else [_num(val)] * n
    return [_num(tok)]


def _num(s):
    try:
        return float(s.replace("D", "E").replace("d", "e"))
    except ValueError:
        return s


def parse_deck(path, paths=None):
    """
    Возвращает список (KEYWORD, [record, ...]); record — список значений (float | str | None).
    paths — псевдонимы PATHS ($NAME в INCLUDE), пути — от каталога основного deck.
    """
    if paths is None:
        paths = {"": Path(path).parent}
    if Path(path).suffix == ".gz":
        import gzip
        text = gzip.decompress(Path(path).read_bytes()).decode("utf-8", errors="replace")
    else:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [_strip_comment(l).rstrip() for l in text.splitlines()]
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        m = _KW_RE.match(line) if line and not line[0].isspace() else None
        if not m:
            i += 1
            continue
        kw = m.group(1)
        i += 1
        if kw == "TITLE":
            i += 1
            out.append((kw, []))
            continue
        if kw in FLAG_KEYWORDS:
            out.append((kw, []))
            continue
        body = []
        while i < n:
            l2 = lines[i]
            if l2 and not l2[0].isspace() and _KW_RE.match(l2):
                break
            body.append(l2)
            i += 1
        records, cur = [], []
        for l2 in body:
            if not l2:
                continue
            for tok in _TOK_RE.findall(l2):
                if tok == "/":
                    records.append(cur)
                    cur = []
                    break          # текст после «/» в строке — комментарий
                cur.extend(_expand(tok))
        if cur:
            records.append(cur)
        while records and records[-1] == [] and len(records) > 1:
            records.pop()
        if kw == "PATHS":
            for r in records:
                if len(r) >= 2:
                    paths[str(r[0]).strip()] = paths[""] / str(r[1]).strip()
            continue
        if kw == "INCLUDE":
            name = str(records[0][0]).strip()
            m_alias = re.match(r"\$(\w+)(.*)", name)
            if m_alias and m_alias.group(1) in paths:
                inc = Path(str(paths[m_alias.group(1)]) + m_alias.group(2))
            else:
                inc = Path(path).parent / name
            if not inc.exists() and inc.with_name(inc.name + ".gz").exists():
                inc = inc.with_name(inc.name + ".gz")        # сжатая копия (большие таблицы в tests/data)
            if not inc.exists():
                raise FileNotFoundError(f"INCLUDE: файл {inc} не найден")
            out.extend(parse_deck(inc, paths))
            continue
        out.append((kw, records))
    return out


_MONTHS = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                            "NOV", "DEC"))}
_MONTHS["JLY"] = 7


def _date(rec):
    """Запись START/DATES: день, месяц (JAN…DEC), год, [ЧЧ:ММ:СС]."""
    day, mon, year = int(rec[0]), _MONTHS[str(rec[1]).upper()[:3]], int(rec[2])
    t = datetime.datetime(year, mon, day)
    if len(rec) > 3 and rec[3] is not None:
        hh, mm, *ss = (float(x) for x in str(rec[3]).split(":"))
        t += datetime.timedelta(hours=hh, minutes=mm, seconds=ss[0] if ss else 0.0)
    return t


def _get(rec, idx, default=None):
    return rec[idx] if idx < len(rec) and rec[idx] is not None else default


class EclipseCase:
    """Модель, собранная из deck: сетка, свойства, начальное состояние, скважины, расписание (в СИ)."""

    def __init__(self, path, grid_only=False, compact=True):
        """
        grid_only=True — читаются только RUNSPEC/GRID/EDIT (статическая модель: сетка, свойства, связи).
        compact=True — неактивные ячейки corner-point сетки исключаются из расчёта (поля — через to_full).
        """
        self.path = str(path)
        self.grid_only = grid_only
        self.compact = compact
        self.cells = None
        self.deck = parse_deck(path)
        kws = [k for k, _ in self.deck]
        self.units_name = "METRIC"
        for k in kws:
            if k in UNITS:
                self.units_name = k
        if "LAB" in kws:
            raise NotImplementedError("единицы LAB не поддерживаются")
        self.u = UNITS[self.units_name]
        self.kro_model = "stone2" if ("STONE2" in kws or "STONE" in kws) else "default"
        self.vapoil = "VAPOIL" in kws
        if "STONE1" in kws and not grid_only:
            raise NotImplementedError("STONE1 не поддерживается")
        self._read()

    # ------------------------------------------------------------ разбор
    def _read(self):
        u = self.u
        self.arrays = {}
        self.raw = {}
        box = None
        self.wells = {}
        self.steps = []          # [(сутки, {скважина: (ключевое слово, запись)}), ...]
        controls = {}
        self.summary_requests = []
        self.drsdt = None
        self.tables = {}
        self.equil = None
        self.rsvd = None
        self.faults = {}
        self.multflt = []
        self.multregt = []
        self.pinch = None
        self.minpv = 1e-6   # м³, умолчание OPM при отсутствии MINPV
        self.region_tables = {}
        self.endscale = False
        self.scalecrs = False
        self.hysteresis = False
        self.ehystr = None
        self.start = None
        self.well_group = {}
        self.group_parent = {}
        self.group_steps = []
        self.vappars = None      # (VAP1, VAP2)
        self.eqlopts = set()
        self.thpres = []         # [(r1, r2, порог Па | None — из начального равновесия)], регионы 0-based
        self.conn_steps = []     # на каждый отчётный шаг: {скважина: {ijk: параметры соединения}}
        self.wtest = {}          # текущие WTEST: {скважина: dict(interval сут, reasons, tests)}
        self.wtest_steps = []    # снимок WTEST на каждый отчётный шаг
        self.wecon = {}          # WECON: {скважина: dict(oil, gas, liq — мин. дебиты м³/сут; wct, gor, wgr, glr; work)}
        self.wecon_steps = []
        self.reopen_steps = []   # скважины с явным WELOPEN OPEN перед отчётным шагом
        self._reopen = set()
        self.vfp = {}            # текущие таблицы VFP: {номер: VFPProd, ('INJ', номер): VFPInj}
        self.vfp_steps = []      # снимок таблиц VFP на каждый отчётный шаг
        self.liftopt = None      # LIFTOPT: dict(inc — шаг газлифта в ед. ALQ, grad — мин. ст. м³ нефти / ст. м³ газа,
                                 # interval — мин. интервал оптимизации, сут)
        self.wliftopt = {}       # WLIFTOPT: {скважина: dict(opt, max — None: по таблице VFP, weight, min)}
        self.liftopt_steps, self.wliftopt_steps = [], []
        self.wefac = {}          # WEFAC: {скважина: коэффициент эксплуатации}
        self.gliftopt = {}       # GLIFTOPT: {группа: макс. газлифт, ед. ALQ}
        self.wefac_steps, self.gliftopt_steps = [], []
        self.branches = {}       # BRANPROP: {нижний узел: dict(up — верхний узел, vfp — таблица шлейфа, alq)}
        self.nodes = {}          # NODEPROP: {узел: dict(p — заданное давление, Па, или None; lift — газлифт в шлейф)}
        self.netbalan = dict(interval=0.0, tol=0.1 * u["p"], iters=10)
        self.grupnet = {}        # GRUPNET (стандартная сеть): {группа: dict(p, vfp, alq)}, ветка — к родителю GRUPTREE
        self.network_steps = []
        gcon = {}
        date = None
        opened = {}
        section = None
        unknown = set()
        for kw, recs in self.deck:
            if kw in ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE"):
                section = kw
                continue
            if self.grid_only and (section not in GRID_SECTIONS or (section == "RUNSPEC" and kw != "DIMENS")):
                continue
            if section == "SUMMARY":
                self.summary_requests.append((kw, recs))
                continue
            if kw in UNSUPPORTED:
                raise NotImplementedError(f"{Path(self.path).name}: ключевое слово {kw} пока не поддерживается")
            if (kw in FLAG_KEYWORDS and kw != "ENDBOX") or kw in IGNORED:
                continue
            if kw == "DIMENS":
                self.dims = tuple(int(v) for v in recs[0][:3])
                self.n = int(np.prod(self.dims))
            elif kw == "BOX":
                b = [int(v) - 1 for v in recs[0][:6]]
                box = (b[0], b[1] + 1, b[2], b[3] + 1, b[4], b[5] + 1)
            elif kw == "ENDBOX":
                box = None
            elif kw in ARRAY_KEYWORDS:
                self._set_array(kw, recs[0], box)
            elif kw in RAW_KEYWORDS:
                self.raw[kw] = np.array([np.nan if v is None else float(v) for v in recs[0]])
            elif kw in EDIT_OPS:
                for r in recs:
                    if r:
                        self._edit(kw, r, box)
            elif kw == "PVTO":
                tabs, cur = [], []
                for r in recs + [[]]:
                    if r:
                        cur.append(r)
                    elif cur:
                        tabs.append(self._pvto(cur))
                        cur = []
                self.region_tables["pvto"] = tabs
            elif kw == "PVTG":
                tabs, cur = [], []
                for r in recs + [[]]:
                    if r:
                        cur.append(r)
                    elif cur:
                        tabs.append(self._pvtg(cur))
                        cur = []
                self.region_tables["pvtg"] = tabs
            elif kw == "PVDO":
                self.region_tables["pvdo"] = [[(r[0] * u["p"], r[1], r[2] * 1e-3) for r in self._rows(rec, 3)]
                                              for rec in recs if rec]
            elif kw == "PVDG":
                bg = u["liq"] / u["gas"]
                self.region_tables["pvdg"] = [[(r[0] * u["p"], r[1] * bg, r[2] * 1e-3) for r in self._rows(rec, 3)]
                                              for rec in recs if rec]
            elif kw == "PVTW":
                self.region_tables["pvtw"] = [(r[0] * u["p"], r[1], _get(r, 2, 0.0) * u["comp"],
                                               _get(r, 3, 0.5) * 1e-3, _get(r, 4, 0.0) * u["comp"])
                                              for r in recs if r]
            elif kw == "ROCK":
                self.region_tables["rock"] = [(r[0] * u["p"], _get(r, 1, 0.0) * u["comp"]) for r in recs if r]
            elif kw == "DENSITY":
                self.region_tables["density"] = [tuple(float(v) * u["rho"] for v in r[:3]) for r in recs if r]
            elif kw == "SWOF":
                tabs = []
                for rec in recs:
                    if rec:
                        rows = self._rows(rec, 4)
                        tabs.append({"sw": [r[0] for r in rows], "krw": [r[1] for r in rows],
                                     "kro": [r[2] for r in rows], "pcow": [r[3] * u["p"] for r in rows]})
                self.region_tables["swof"] = tabs
            elif kw == "SGOF":
                tabs = []
                for rec in recs:
                    if rec:
                        rows = self._rows(rec, 4)
                        tabs.append({"sg": [r[0] for r in rows], "krg": [r[1] for r in rows],
                                     "kro": [r[2] for r in rows], "pcog": [r[3] * u["p"] for r in rows]})
                self.region_tables["sgof"] = tabs
            elif kw == "EQUIL":
                self.region_tables["equil"] = [
                    dict(datum=r[0] * u["L"], p=r[1] * u["p"], woc=_get(r, 2, 1e9 / u["L"]) * u["L"],
                         pcwoc=_get(r, 3, 0.0) * u["p"], goc=_get(r, 4, -1e9 / u["L"]) * u["L"],
                         pcgoc=_get(r, 5, 0.0) * u["p"], rvvd=int(_get(r, 7, 0) or 0) > 0) for r in recs if r]
            elif kw == "RSVD":
                self.region_tables["rsvd"] = [[(r[0] * u["L"], r[1] * u["gas"] / u["liq"]) for r in self._rows(rec, 2)]
                                              for rec in recs if rec]
            elif kw == "PBVD":
                self.region_tables["pbvd"] = [[(r[0] * u["L"], r[1] * u["p"]) for r in self._rows(rec, 2)]
                                              for rec in recs]
            elif kw == "RVVD":
                self.region_tables["rvvd"] = [[(r[0] * u["L"], r[1] * u["liq"] / u["gas"]) for r in self._rows(rec, 2)]
                                              for rec in recs if rec]
            elif kw == "DRSDT":
                # DRSDT и VAPPARS взаимоисключающие: действует заданный последним (сообщение Eclipse
                # «DRSDT CONSTRAINT IS NOW DISABLED BY SPECIFICATION OF VAPPARS KEYWORD»)
                self.drsdt = float(recs[0][0]) * u["gas"] / u["liq"] / DAY
                self.vappars = None
            elif kw == "ENDSCALE":
                r = recs[0] if recs else []
                if str(_get(r, 0, "NODIR")).upper() == "DIRECT":
                    raise NotImplementedError("ENDSCALE DIRECT (направленное масштабирование) не поддерживается")
                self.endscale = True
            elif kw == "SCALECRS":
                self.scalecrs = str(_get(recs[0] if recs else [], 0, "NO")).upper().startswith("Y")
            elif kw == "SATOPTS":
                opts = {str(v).upper() for r in recs for v in r if v is not None}
                if "DIRECT" in opts or "IRREVERS" in opts:
                    raise NotImplementedError(f"SATOPTS {sorted(opts)} не поддерживаются")
                self.hysteresis = "HYSTER" in opts
            elif kw in ("EHYSTR",):
                self.ehystr = recs[0] if recs else []
            elif kw == "FAULTS":
                for r in recs:
                    if r:
                        fbox = tuple(int(v) for v in r[1:7])
                        self.faults.setdefault(str(r[0]), []).append((fbox, str(r[7]).upper()))
            elif kw == "MULTFLT":
                self.multflt.extend((str(r[0]), float(r[1])) for r in recs if r)
            elif kw == "MULTREGT":
                for r in recs:
                    if r:
                        self.multregt.append(dict(r1=_get(r, 0), r2=_get(r, 1), mult=float(r[2]),
                                                  dirs=str(_get(r, 3, "XYZ")).upper(),
                                                  nnc=str(_get(r, 4, "ALL")).upper(),
                                                  region=str(_get(r, 5, "M")).upper()[0]))
            elif kw == "PINCH":
                r = recs[0] if recs else []
                self.pinch = dict(threshold=float(_get(r, 0, 0.001)) * u["L"],
                                  gap=str(_get(r, 1, "GAP")).upper() != "NOGAP",
                                  max_gap=float(_get(r, 2, 1e20)) * u["L"],
                                  trans=str(_get(r, 3, "TOPBOT")).upper(),
                                  multz=str(_get(r, 4, "TOP")).upper())
            elif kw == "MINPV":
                self.minpv = float(recs[0][0]) * u["liq"]   # rm³ / rb
            elif kw == "WELSPECS":
                for r in recs:
                    if not r:
                        continue
                    self.well_group[r[0]] = str(_get(r, 1, "FIELD"))
                    self.wells[r[0]] = dict(name=r[0], i=int(r[2]) - 1, j=int(r[3]) - 1,
                                            ref_depth=None if _get(r, 4) is None else r[4] * u["L"],
                                            phase=str(_get(r, 5, "OIL")).upper(), conns={})
            elif kw == "VAPPARS":
                if self.steps:
                    raise NotImplementedError("VAPPARS после начала расписания не поддерживается")
                r = recs[0]
                self.vappars = (float(_get(r, 0, 0.0) or 0.0), float(_get(r, 1, 0.0) or 0.0))
                self.drsdt = None
            elif kw == "EQLOPTS":
                self.eqlopts = {str(v).upper() for r in recs for v in r if v is not None}
                bad = self.eqlopts - {"THPRES", "IRREVERS"}
                if bad:
                    raise NotImplementedError(f"EQLOPTS: опции {', '.join(sorted(bad))} не поддерживаются")
            elif kw == "THPRES":
                for r in recs:
                    if r:
                        if _get(r, 0) is None or _get(r, 1) is None:
                            raise NotImplementedError("THPRES: регионы по умолчанию не поддерживаются")
                        val = _get(r, 2)
                        self.thpres.append((int(r[0]) - 1, int(r[1]) - 1, None if val is None else float(val) * u["p"]))
            elif kw == "COMPDAT":
                for r in recs:
                    if r:
                        self._compdat(r)
            elif kw in ("WCONPROD", "WCONINJE", "WCONHIST", "WCONINJH"):
                for r in recs:
                    if not r:
                        continue
                    st = str(_get(r, 1 if kw in ("WCONPROD", "WCONHIST") else 2, "OPEN")).upper()
                    for name in self._match_wells(r[0]):
                        controls[name] = (kw, r, st)
                        opened.setdefault(name, len(self.steps) if st == "OPEN" else None)
            elif kw == "GRUPTREE":
                for r in recs:
                    if r:
                        self.group_parent[str(r[0])] = str(_get(r, 1, "FIELD"))
            elif kw == "GCONPROD":
                for r in recs:
                    if r:
                        gcon[(str(r[0]), "prod")] = r
            elif kw == "GCONINJE":
                for r in recs:
                    if r:
                        gcon[(str(r[0]), "inj", str(r[1]).upper()[:3])] = r
            elif kw == "WELOPEN":
                for r in recs:
                    if not r:
                        continue
                    st = str(_get(r, 1, "OPEN")).upper()
                    if any(_get(r, m) is not None for m in range(2, 7)):
                        self._welopen_conns(r, st)
                        continue
                    for name in self._match_wells(r[0]):
                        if st == "OPEN":
                            self._reopen.add(name)
                        if name in controls:
                            controls[name] = controls[name][:2] + (st,)
                        elif st == "OPEN":
                            raise ValueError(f"WELOPEN {name}: скважина открыта без режима")
            elif kw == "VFPPROD":
                from .vfp import VFPProd
                t = VFPProd(recs, u)
                self.vfp[t.number] = t
            elif kw == "WEFAC":
                for r in recs:
                    if r:
                        for name in self._match_wells(str(r[0])):
                            self.wefac[name] = float(_get(r, 1, 1.0) or 1.0)
            elif kw == "BRANPROP":
                for r in recs:
                    if r:
                        vfp = int(_get(r, 2, 0) or 0)
                        if vfp > 0:
                            self.branches[str(r[0])] = dict(up=str(r[1]), vfp=vfp, alq=float(_get(r, 3, 0.0) or 0.0))
                        else:
                            self.branches.pop(str(r[0]), None)
            elif kw == "NODEPROP":
                for r in recs:
                    if r:
                        p = _get(r, 1)
                        self.nodes[str(r[0])] = dict(p=None if p is None else float(p) * u["p"],
                                                     lift=str(_get(r, 3, "NO")).upper().startswith("Y"))
            elif kw == "GRUPNET":
                for r in recs:
                    if r:
                        p = _get(r, 1)
                        self.grupnet[str(r[0])] = dict(p=None if p is None else float(p) * u["p"],
                                                       vfp=int(_get(r, 2, 0) or 0), alq=float(_get(r, 3, 0.0) or 0.0))
            elif kw == "NETBALAN":
                r = recs[0] if recs else []
                self.netbalan = dict(interval=float(_get(r, 0, 0.0) or 0.0),
                                     tol=float(_get(r, 1, 0.1) or 0.1) * u["p"], iters=int(_get(r, 2, 10) or 10))
            elif kw == "GLIFTOPT":
                for r in recs:
                    if r and _get(r, 1) is not None:
                        self.gliftopt[str(r[0])] = float(r[1])
            elif kw == "LIFTOPT":
                r = recs[0]
                self.liftopt = dict(inc=float(_get(r, 0, 0.0) or 0.0), grad=float(_get(r, 1, 0.0) or 0.0)
                                    * u["liq"] / u["gas"], interval=float(_get(r, 2, 0.0) or 0.0))
            elif kw == "WLIFTOPT":
                for r in recs:
                    if not r:
                        continue
                    cfg = dict(opt=str(_get(r, 1, "NO")).upper().startswith("Y"), max=_get(r, 2),
                               weight=float(_get(r, 3, 1.0) or 1.0), min=float(_get(r, 4, 0.0) or 0.0))
                    for name in self._match_wells(str(r[0])):
                        self.wliftopt[name] = cfg
            elif kw == "VFPINJ":
                from .vfp import VFPInj
                t = VFPInj(recs, u)
                self.vfp[("INJ", t.number)] = t
            elif kw == "WTEST":
                for r in recs:
                    if r:
                        self._wtest(r)
            elif kw == "WECON":
                for r in recs:
                    if not r:
                        continue
                    work = str(_get(r, 6, "NONE")).upper()
                    if str(_get(r, 9, "RATE")).upper() != "RATE":
                        raise NotImplementedError(f"WECON {r[0]}: пределы по потенциалу (POTN)")
                    if str(_get(r, 7, "NO")).upper().startswith("Y"):
                        raise NotImplementedError(f"WECON {r[0]}: остановка расчёта (END RUN)")
                    lim = lambda k, s: None if not _get(r, k) else float(_get(r, k)) * s
                    cfg = dict(oil=lim(1, u["liq"]), gas=lim(2, u["gas"]), liq=lim(13, u["liq"]),
                               wct=lim(3, 1.0), gor=lim(4, u["gas"] / u["liq"]), wgr=lim(5, u["liq"] / u["gas"]),
                               glr=lim(12, u["gas"] / u["liq"]), work=work)
                    ratios = any(cfg[k] for k in ("wct", "gor", "wgr", "glr"))
                    if ratios and work not in ("NONE", "WELL"):
                        raise NotImplementedError(f"WECON {r[0]}: процедура {work} (поддержаны NONE и WELL)")
                    for name in self._match_wells(str(r[0])):
                        self.wecon[name] = cfg
            elif kw == "START":
                r = recs[0]
                self.start = _date(r)
                date = self.start
            elif kw == "DATES":
                for r in recs:
                    if not r:
                        continue
                    new = _date(r)
                    if date is None:
                        raise ValueError("DATES без START")
                    d = (new - date).total_seconds() / DAY
                    if d <= 0:
                        raise ValueError(f"DATES {new:%d.%m.%Y}: дата не позже предыдущей")
                    self.steps.append((d, dict(controls)))
                    self.group_steps.append(dict(gcon))
                    self.conn_steps.append(self._conn_snapshot())
                    self._close_step_events()
                    date = new
            elif kw == "TSTEP":
                for rec in recs:
                    for v in rec:
                        self.steps.append((float(v), dict(controls)))
                        self.group_steps.append(dict(gcon))
                        self.conn_steps.append(self._conn_snapshot())
                        self._close_step_events()
                        if date is not None:
                            date = date + datetime.timedelta(days=float(v))
            else:
                unknown.add(kw)
        rt = self.region_tables
        for key in ("pvto", "pvdo", "pvdg", "pvtg", "pvtw", "swof", "sgof"):
            if key in rt:
                self.tables[key] = rt[key][0]
        if "rock" in rt:
            self.rock = rt["rock"][0]
        if "density" in rt:
            self.density = rt["density"][0]
        if "equil" in rt:
            self.equil = rt["equil"][0]
        if "rsvd" in rt:
            self.rsvd = rt["rsvd"][0]
        if unknown:
            print(f"  ⚠ deck: проигнорированы ключевые слова: {', '.join(sorted(unknown))}")

        self.corner_point = "COORD" in self.raw or "ZCORN" in self.raw
        geom = ("COORD", "ZCORN") if self.corner_point else ("DX", "DY", "DZ", "TOPS")
        for req in geom + ("PORO", "PERMX"):
            if req not in self.arrays and req not in self.raw:
                raise ValueError(f"в deck нет {req}")
        if self.grid_only:
            return
        if self.hysteresis:
            r = self.ehystr or []
            model = int(_get(r, 1, 0) or 0)
            what = str(_get(r, 4, "BOTH")).upper()
            if model != 0:
                raise NotImplementedError(f"гистерезис: EHYSTR модель {model} не поддерживается (только 0 — Карлсон)")
            if what != "KR":
                raise NotImplementedError(f"гистерезис Pc (EHYSTR {what}) не поддерживается — только KR")
        if not self.steps:
            raise ValueError("в deck нет TSTEP")
        for name in self.wells:
            if all(name not in c for _, c in self.steps):
                raise ValueError(f"скважина {name}: нет режима (WCONPROD/WCONINJE/WCONHIST/WCONINJH)")
        # начальный тип — по первому режиму с ненулевым управлением; дальше тип может меняться по расписанию
        self.well_type = {}
        for name in self.wells:
            ctl = [c[name] for _, c in self.steps if name in c]
            live = [x for x in ctl if self.control_cfg(name, x)["control_type"] != "shut"]
            kw = (live or ctl)[0][0]
            self.well_type[name] = "producer" if kw in ("WCONPROD", "WCONHIST") else "injector"
        if "pvdg" not in self.tables and "pvtg" not in self.tables:
            raise ValueError("в deck нет PVDG или PVTG")
        if "pvtw" not in self.tables:
            raise ValueError("в deck нет PVTW")
        if self.vapoil and "pvtg" not in self.tables:
            raise ValueError("VAPOIL требует PVTG")
        if "pvto" not in self.tables and "pvdo" not in self.tables:
            raise ValueError("в deck нет PVTO или PVDO")
        if self.equil is None:
            raise NotImplementedError("начальное состояние поддерживается только через EQUIL")

    @property
    def report_days(self):
        return [d for d, _ in self.steps]

    def _compdat(self, r):
        """COMPDAT: соединения (I, J, K1..K2) с CF, диаметром, Kh, скином, направлением и Ro; повтор — замена."""
        name = str(r[0])
        for wn in self._match_wells(name):
            w = self.wells[wn]
            i = int(_get(r, 1, 0) or 0) - 1
            j = int(_get(r, 2, 0) or 0) - 1
            i = w["i"] if i < 0 else i
            j = w["j"] if j < 0 else j
            state = str(_get(r, 5, "OPEN")).upper()
            if state not in ("OPEN", "SHUT"):
                raise NotImplementedError(f"COMPDAT {wn}: состояние соединения {state} (поддерживаются OPEN, SHUT)")
            if int(_get(r, 6, 0) or 0) > 0:
                raise NotImplementedError(f"COMPDAT {wn}: SATNUM соединения не поддерживается")
            if float(_get(r, 11, 0.0) or 0.0) != 0.0:
                raise NotImplementedError(f"COMPDAT {wn}: D-фактор (не-Дарси) не поддерживается")
            direction = str(_get(r, 12, "Z")).upper()[0]
            if direction not in "XYZ":
                raise ValueError(f"COMPDAT {wn}: направление {direction}")
            spec = dict(open=state == "OPEN", cf=_get(r, 7), diam=_get(r, 8), kh=_get(r, 9),
                        skin=float(_get(r, 10, 0.0) or 0.0), dir=direction, ro=_get(r, 13))
            for k in range(int(r[3]) - 1, int(r[4])):
                w["conns"].pop((i, j, k), None)
                w["conns"][(i, j, k)] = dict(spec)

    def _welopen_conns(self, r, st):
        """WELOPEN по соединениям: I, J, K = 0 или по умолчанию — любые; номера заканчиваний не поддерживаются."""
        if _get(r, 5) is not None or _get(r, 6) is not None:
            raise NotImplementedError(f"WELOPEN {r[0]}: выбор по номерам заканчиваний не поддерживается")
        if st not in ("OPEN", "SHUT", "STOP"):
            raise NotImplementedError(f"WELOPEN {r[0]}: статус {st}")
        sel = [int(_get(r, m, 0) or 0) - 1 for m in (2, 3, 4)]
        for wn in self._match_wells(r[0]):
            for ijk, c in self.wells[wn]["conns"].items():
                if all(s < 0 or s == v for s, v in zip(sel, ijk)):
                    c["open"] = st == "OPEN"

    def _wtest(self, r):
        """WTEST: интервал проверки (сут), причины закрытия (P, E, G, D, C), число проверок (0 — без ограничения)."""
        interval = float(_get(r, 1, 0.0) or 0.0)
        reasons = str(_get(r, 2, "") or "").upper()
        bad = set(reasons) - set("PEGDC")
        if bad:
            raise ValueError(f"WTEST {r[0]}: неизвестные причины {''.join(sorted(bad))}")
        tests = int(_get(r, 3, 0) or 0)
        for name in self._match_wells(r[0]):
            if interval <= 0 or not reasons:
                self.wtest.pop(name, None)
            else:
                self.wtest[name] = dict(interval=interval, reasons=reasons, tests=tests)

    def _close_step_events(self):
        self.wtest_steps.append({k: dict(v) for k, v in self.wtest.items()})
        self.wecon_steps.append(dict(self.wecon))
        self.vfp_steps.append(dict(self.vfp))
        self.liftopt_steps.append(self.liftopt)
        self.wliftopt_steps.append(dict(self.wliftopt))
        self.wefac_steps.append(dict(self.wefac))
        self.gliftopt_steps.append(dict(self.gliftopt))
        branches = {k: dict(v) for k, v in self.branches.items()}
        nodes = {k: dict(v) for k, v in self.nodes.items()}
        for g, r in self.grupnet.items():
            if r["vfp"] > 0 and g != "FIELD":
                branches[g] = dict(up=self.group_parent.get(g, "FIELD"), vfp=r["vfp"], alq=r["alq"])
            if r["p"] is not None:
                nodes[g] = dict(p=r["p"], lift=False)
        self.network_steps.append(dict(branches=branches, nodes=nodes, balan=dict(self.netbalan))
                                  if branches else None)
        self.reopen_steps.append(self._reopen)
        self._reopen = set()

    def _conn_snapshot(self):
        return {wn: {ijk: dict(c) for ijk, c in w["conns"].items()} for wn, w in self.wells.items()}

    def well_connections(self, name, step):
        """
        Соединения скважины на отчётном шаге: объединение всех соединений за расписание (фиксированный порядок —
        по первому появлению) с параметрами шага; ещё не вскрытые — закрыты. Возвращает список dict(ijk, ...).
        """
        order, first = [], {}
        for snap in self.conn_steps or [self._conn_snapshot()]:
            for ijk, c in snap.get(name, {}).items():
                if ijk not in first:
                    first[ijk] = c
                    order.append(ijk)
        cur = (self.conn_steps[step] if self.conn_steps else self._conn_snapshot()).get(name, {})
        u = self.u
        out = []
        for ijk in order:
            c = cur.get(ijk)
            spec = dict(c if c is not None else first[ijk])
            if c is None:
                spec["open"] = False
            conv = (u["liq"] / DAY) * 1e-3 / u["p"]
            out.append(dict(ijk=ijk, open=bool(spec["open"]), dir=spec["dir"], skin=spec["skin"],
                            wi=None if spec["cf"] is None else float(spec["cf"]) * conv,
                            rw=0.5 * float(spec["diam"] if spec["diam"] is not None else 1.0 / u["L"] * 0.3048) * u["L"],
                            kh=None if spec["kh"] is None else float(spec["kh"]) * MD * u["L"],
                            ro=None if spec["ro"] is None else float(spec["ro"]) * u["L"]))
        return out

    def _well(self, name):
        if name not in self.wells:
            raise ValueError(f"скважина {name} не объявлена в WELSPECS")
        return self.wells[name]

    def _match_wells(self, pattern):
        pattern = str(pattern)
        names = [n for n in self.wells if fnmatch.fnmatchcase(n, pattern)]
        if not names:
            raise ValueError(f"нет скважин, подходящих под '{pattern}'")
        return names

    def _box(self, rec, start, box):
        nx, ny, nz = self.dims
        vals = [_get(rec, start + m) for m in range(6)]
        if all(v is None for v in vals):
            return box if box is not None else (0, nx, 0, ny, 0, nz)
        b = box if box is not None else (0, nx, 0, ny, 0, nz)
        full = (b[0] + 1, b[1], b[2] + 1, b[3], b[4] + 1, b[5])
        v = [int(full[m] if vals[m] is None else vals[m]) for m in range(6)]
        return (v[0] - 1, v[1], v[2] - 1, v[3], v[4] - 1, v[5])

    def _array(self, name):
        """Массив [nz, ny, nx]; незаданные значения — NaN, для множителей/NTG/регионов — 1."""
        if name not in self.arrays:
            nx, ny, nz = self.dims
            fill = 1.0 if name.startswith("MULT") or name in ("NTG", "FLUXNUM", "MULTNUM", "SATNUM", "PVTNUM", "EQLNUM", "FIPNUM") else np.nan
            self.arrays[name] = np.full((nz, ny, nx), fill)
        return self.arrays[name]

    def _edit(self, op, rec, box):
        """EQUALS / ADD / MULTIPLY: (массив, значение, [бокс]); COPY: (откуда, куда, [бокс])."""
        nx, ny, nz = self.dims
        i0, i1, j0, j1, k0, k1 = self._box(rec, 2, box)
        sl = (slice(k0, k1), slice(j0, j1), slice(i0, i1))
        if op == "COPY":
            src, dst = str(rec[0]).upper(), str(rec[1]).upper()
            if src not in self.arrays:
                raise ValueError(f"COPY: массив {src} не задан")
            arr = self._array(dst)
            arr[sl] = self.arrays[src][sl]
            return
        name, val = str(rec[0]).upper(), float(rec[1])
        if name not in ARRAY_KEYWORDS:
            raise NotImplementedError(f"{op}: массив {name} не поддерживается")
        if op == "EQUALS":
            self._array(name)[sl] = val
            return
        arr = self._array(name)
        if np.isnan(arr[sl]).all():
            raise ValueError(f"{op}: массив {name} не задан")
        if op == "ADD":
            arr[sl] += val
        else:
            arr[sl] *= val

    @staticmethod
    def _rows(rec, width):
        vals = [0.0 if v is None else float(v) for v in rec]
        return [vals[i:i + width] for i in range(0, len(vals) - len(vals) % width, width)]

    def _pvtg(self, recs):
        """PVTG: запись 'P Rv Bg μg [Rv Bg μg ...]' → (P, [(Rv, Bg, μg)]) в СИ (Rv — м³ нефти / м³ газа)."""
        u = self.u
        rv_u = u["liq"] / u["gas"]
        out = []
        for r in recs:
            if not r:
                continue
            rows = [(r[k] * rv_u, r[k + 1] * rv_u, r[k + 2] * 1e-3) for k in range(1, len(r) - 2, 3)]
            out.append((r[0] * u["p"], rows))
        return out

    def _pvto(self, recs):
        u = self.u
        rs_u = u["gas"] / u["liq"]
        out = []
        for r in recs:
            if not r:
                continue
            rs = r[0] * rs_u
            rows = [(r[k] * u["p"], r[k + 1], r[k + 2] * 1e-3) for k in range(1, len(r) - 2, 3)]
            out.append((rs, rows))
        return out

    def _set_array(self, kw, values, box):
        nx, ny, nz = self.dims
        arr = self._array(kw)
        if kw == "TOPS" and box is None:
            box = (0, nx, 0, ny, 0, 1)
        i0, i1, j0, j1, k0, k1 = box if box is not None else (0, nx, 0, ny, 0, nz)
        need = (i1 - i0) * (j1 - j0) * (k1 - k0)
        vals = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
        if vals.size != need:
            raise ValueError(f"{kw}: {vals.size} значений, ожидалось {need}")
        arr[k0:k1, j0:j1, i0:i1] = vals.reshape(k1 - k0, j1 - j0, i1 - i0)

    def field(self, kw):
        """Массив [nx, ny, nz] (порядок тензоров симулятора)."""
        return np.ascontiguousarray(self.arrays[kw].transpose(2, 1, 0))

    # ------------------------------------------------------------ сборка
    def build(self, device=None, sim_overrides=None):
        from .reservoir import Reservoir
        from .fluid import Fluid
        from .well import WellManager

        device = device or torch.device("cpu")
        u = self.u
        nx, ny, nz = self.dims
        res = self._build_grid(Reservoir, device)

        rt = self.region_tables
        regs = {name: self._region_index(name) for name in ("PVTNUM", "SATNUM", "EQLNUM")}
        regs["IMBNUM"] = self._region_index("IMBNUM") if "IMBNUM" in self.arrays else regs["SATNUM"]
        n_pvt, n_sat = int(regs["PVTNUM"].max()) + 1, int(regs["SATNUM"].max()) + 1
        n_tab = max(n_sat, int(regs["IMBNUM"].max()) + 1 if self.hysteresis else 0)
        pick = lambda key, r: rt[key][min(r, len(rt[key]) - 1)] if key in rt else None
        for key, n in (("pvtw", n_pvt), ("pvdg", n_pvt), ("pvtg", n_pvt), ("swof", n_tab), ("sgof", n_tab)):
            if key in rt and len(rt[key]) < n:
                raise ValueError(f"{key.upper()}: таблиц {len(rt[key])}, а регионов {n}")
        oil_key = "pvto" if "pvto" in rt else "pvdo"
        if len(rt[oil_key]) < n_pvt:
            raise ValueError(f"{oil_key.upper()}: таблиц {len(rt[oil_key])}, а регионов PVTNUM {n_pvt}")
        rho = self.density
        pvts, pvts_cpu = [], []
        cpu = torch.device("cpu")
        for r in range(n_pvt):
            tab = {oil_key: rt[oil_key][r], "pvdg": pick("pvdg", r), "pvtw": pick("pvtw", r),
                   "pvtg": pick("pvtg", r), "vapoil": self.vapoil}
            pvts.append(DeckPVT(tab, pick("density", r) or rho, device))
            # равновесие считается по ячейкам/скалярам — на CPU
            pvts_cpu.append(pvts[-1] if device.type == "cpu" else DeckPVT(tab, pick("density", r) or rho, cpu))
        krs = [ThreePhaseRelPerm(tables={k: pick(k, r) for k in ("swof", "sgof") if k in rt}, device=device,
                                 kro_model=self.kro_model) for r in range(n_tab)]
        if n_pvt > 1:
            from .regions import RegionPVT
            pvt = RegionPVT(pvts, torch.as_tensor(regs["PVTNUM"], device=device))
        else:
            pvt = pvts[0]
        ep = self._endpoint_model(krs, regs["SATNUM"], device)
        swatinit = self._cell_array("SWATINIT") if "SWATINIT" in self.arrays else None
        if self.hysteresis:
            from .hysteresis import CarlsonHysteresis
            imb = self._endpoint_model(krs, regs["IMBNUM"], device, prefix="I")
            ep.hyst = CarlsonHysteresis(ep, imb, model=int(_get(self.ehystr or [], 1, 0) or 0))
        if self.endscale or swatinit is not None or self.hysteresis:
            kr = ep
        elif n_sat > 1:
            from .regions import RegionRelPerm
            kr = RegionRelPerm(krs, torch.as_tensor(regs["SATNUM"], device=device))
        else:
            kr = krs[0]
        if self.thpres:
            if "THPRES" not in self.eqlopts:
                raise ValueError("THPRES без EQLOPTS 'THPRES'")
            res.cell_eqlnum = torch.as_tensor(regs["EQLNUM"], device=device)
            res.thpres = dict(records=list(self.thpres), irreversible="IRREVERS" in self.eqlopts)
        swc = ep.endpoint("SWL", slice(None))
        rocks = rt.get("rock", [(1e5, 0.0)])
        rock = np.array([rocks[min(r, len(rocks) - 1)] for r in range(n_pvt)])[regs["PVTNUM"]]
        p_ref = torch.as_tensor(rock[:, 0], dtype=torch.float64, device=device)
        c_r = torch.as_tensor(rock[:, 1], dtype=torch.float64, device=device)

        fluid = Fluid({"pressure": self.equil["p"] / 1e6, "s_w": float(swc[0]), "s_g": 0.0, "rho_oil": rho[0],
                       "rho_water": rho[1], "rho_gas": rho[2], "c_rock": float(c_r[0]) * 1e6}, res, device=device)
        fluid.blackoil_pvt = pvt
        fluid.vappars = self.vappars if self.vapoil else None
        fluid.blackoil_relperm = kr
        fluid.rock_ref_pressure = p_ref
        fluid.rock_compressibility = c_r
        fluid.sw_cr = float(swc.min())
        p, sw, sg, rs, rv = (a.to(device) for a in self.equilibrate(pvts_cpu, ep, regs, cpu, swatinit))
        shp = tuple(res.dimensions)
        fluid.pressure = p.view(shp).clone()
        fluid.s_w = sw.view(shp).clone()
        fluid.s_g = sg.view(shp).clone()
        if ep.hyst is not None:
            ep.hyst.commit(sw, sg)
        fluid.s_o = (1.0 - sw - sg).view(shp).clone()
        fluid.rs = rs.view(shp).clone()
        fluid.rv = rv.view(shp).clone()
        fluid.prev_pressure = fluid.pressure.clone()

        wm = WellManager(self.well_configs(), res)
        sim = {"solver_type": "fully_implicit", "fim_model": "blackoil", "drsdt": self.drsdt,
               "initial_time_step_days": 1.0, "newton_max_iter": 20, "max_time_step_attempts": 10,
               # шаг и сходимость — умолчания OPM Flow
               "dt_control": "opm", "newton_tol_cnv": 1e-2, "newton_max_strict_iter": 0,
               "newton_tol_well_rate": 1e-4}
        sim.update(sim_overrides or {})
        return res, fluid, wm, sim

    def _build_grid(self, Reservoir, device):
        """Reservoir с геометрией deck; для corner-point — с переопределёнными объёмами, глубинами и T."""
        u = self.u
        nx, ny, nz = self.dims
        L = u["L"]
        ntg = self.field("NTG") if "NTG" in self.arrays else np.ones((nx, ny, nz))
        poro = self.field("PORO")
        kx = self.field("PERMX")
        ky = self.field("PERMY") if "PERMY" in self.arrays else kx
        kz = self.field("PERMZ") if "PERMZ" in self.arrays else kx
        for name, arr in (("PORO", poro), ("PERMX", kx), ("PERMY", ky), ("PERMZ", kz)):
            if not self.corner_point and np.isnan(arr).any():
                raise ValueError(f"{name} задан не для всех ячеек")
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)

        if not self.corner_point:
            dx3, dy3, dz3 = (self.field(k) * L for k in ("DX", "DY", "DZ"))
            dx, dy, dz = dx3[:, 0, 0], dy3[0, :, 0], dz3[0, 0, :]
            for name, full, vec, shape in (("DX", dx3, dx, (nx, 1, 1)), ("DY", dy3, dy, (1, ny, 1)),
                                           ("DZ", dz3, dz, (1, 1, nz))):
                if not np.allclose(full, vec.reshape(shape)):
                    raise NotImplementedError(f"{name} меняется не только вдоль своей оси — нужна corner-point сетка")
            tops = self.field("TOPS")[:, :, 0] * L
            if not np.allclose(tops, tops[0, 0]):
                raise NotImplementedError("TOPS не постоянен: для наклонных слоёв используйте COORD/ZCORN")
            if "ACTNUM" in self.arrays and not (self.arrays["ACTNUM"] > 0).all():
                raise NotImplementedError("ACTNUM на декартовой сетке не поддерживается")
            self.depth_top = float(tops[0, 0])
            res = Reservoir({"dimensions": [nx, ny, nz], "grid": {"dx": dx.tolist(), "dy": dy.tolist(),
                                                                   "dz": dz.tolist()},
                             "porosity": (poro * ntg).tolist(), "permeability": (kx * ntg).tolist(),
                             "k_vertical_fraction": 1.0}, device=device)
            res.permeability_y = t(ky * ntg * MD)
            res.permeability_z = t(kz * MD)
            self.cell_depth_abs = self.depth_top + np.broadcast_to(
                res.z_centers.detach().cpu().double().numpy().reshape(1, 1, nz), (nx, ny, nz))
            return res

        s = self.static_model()
        g, act = s["grid"], s["active"]
        self.grid = g
        self.active = act
        # в расчёте только активные ячейки: сетка n_active × 1 × 1, связи и скважины — через cell_map
        self.cells = np.flatnonzero(act.reshape(-1)) if self.compact and not act.all() else None
        shp = (self.n_cells, 1, 1) if self.cells is not None else (nx, ny, nz)
        C = lambda a: self._a(np.broadcast_to(a, (nx, ny, nz))).reshape(shp)
        self.depth_top = float(g.corners[..., 2][act].min())
        self.cell_depth_abs = C(g.depth)
        clean = lambda a: C(np.where(act, np.nan_to_num(a), 0.0))
        if self.cells is not None:
            grid = {"dx": np.ones(shp[0]).tolist(), "dy": [1.0], "dz": [1.0]}
        else:
            grid = {"dx": g.dx.mean(axis=(1, 2)).tolist(), "dy": g.dy.mean(axis=(0, 2)).tolist(),
                    "dz": [g.dz[:, :, k].mean() for k in range(nz)]}
        res = Reservoir({"dimensions": list(shp), "grid": grid,
                         "porosity": clean(poro * ntg).tolist(), "permeability": clean(kx * ntg).tolist(),
                         "k_vertical_fraction": 1.0}, device=device)
        res.permeability_y = t(clean(ky * ntg) * MD)
        res.permeability_z = t(clean(kz) * MD)
        pv = s["pv"]
        pv_floor = 1e-6 * pv[act].mean()
        vol = np.maximum(g.volume, 1e-30)
        # неактивные ячейки (если не исключены) изолированы; минимальный поровый объём — невырожденный Якобиан
        res.porosity = t(C(np.where(act, pv / vol, pv_floor / vol)))
        res.cell_volume = t(C(vol))
        res.cell_volume_flat = res.cell_volume.reshape(-1)
        res.porous_volume = res.cell_volume * res.porosity
        res.cell_depth = t(C(g.depth - self.depth_top))
        res.cell_dims = (t(C(g.dx)), t(C(g.dy)), t(C(g.dz * ntg)))
        c = s["conn"]
        keep = c["T"] > 0
        cmap = np.arange(self.n, dtype=np.int64)
        if self.cells is not None:
            cmap = np.full(self.n, -1, dtype=np.int64)
            cmap[self.cells] = np.arange(self.cells.size)
        res.connections = (torch.as_tensor(cmap[c["c1"][keep]], device=device),
                           torch.as_tensor(cmap[c["c2"][keep]], device=device), t(c["T"][keep]))
        res.actnum = t(C(act.astype(np.float64)))
        res.full_dims = (nx, ny, nz)
        res.cell_map = cmap
        res.active_cells = self.cells
        return res

    @property
    def n_cells(self):
        """Число расчётных ячеек (активных, если сетка сжата)."""
        return self.n if getattr(self, "cells", None) is None else int(self.cells.size)

    def _a(self, a):
        """Ячеечный массив полной сетки (порядок симулятора) → расчётные ячейки."""
        a = np.asarray(a).reshape(-1)
        return a if getattr(self, "cells", None) is None else a[self.cells]

    def to_full(self, a, fill=0.0):
        """Расчётные ячейки → массив [nx, ny, nz] полной сетки (неактивные — fill)."""
        a = np.asarray(a).reshape(-1)
        if getattr(self, "cells", None) is None:
            return a.reshape(self.dims)
        out = np.full(self.n, fill, dtype=a.dtype)
        out[self.cells] = a
        return out.reshape(self.dims)

    def static_model(self):
        """
        Статическая модель corner-point (как в Eclipse): активность (ACTNUM, MINPV, толщина < PINCH),
        поровые объёмы и связи с проводимостями после множителей MULT*, FAULTS/MULTFLT, MULTREGT.
        Возвращает dict: grid, active [nx,ny,nz], pv [nx,ny,nz] (м³), conn = dict(c1, c2, T [м³], axis, nnc).
        """
        from .corner_point import CornerPointGrid
        if not self.corner_point:
            raise NotImplementedError("static_model: только corner-point сетки")
        u = self.u
        nx, ny, nz = self.dims
        L = u["L"]
        actnum = self.arrays["ACTNUM"].reshape(-1) if "ACTNUM" in self.arrays else None
        g = CornerPointGrid(self.raw["COORD"] * L, self.raw["ZCORN"] * L, self.dims, actnum)
        f = lambda name: self.field(name) if name in self.arrays else np.ones((nx, ny, nz))
        ntg = f("NTG")
        poro = np.nan_to_num(self.field("PORO"))
        pv = g.volume * poro * ntg * f("MULTPV")
        active = g.actnum & (pv > 0)
        if self.minpv > 0:
            active &= pv >= self.minpv
        if self.pinch is not None:
            active &= g.dz >= self.pinch["threshold"]
            if self.pinch["trans"] != "TOPBOT" or self.pinch["multz"] != "TOP":
                raise NotImplementedError("PINCH: поддерживаются только TOPBOT и TOP")
        kx = np.nan_to_num(self.field("PERMX")) * MD
        ky = np.nan_to_num(self.field("PERMY") if "PERMY" in self.arrays else self.field("PERMX")) * MD
        kz = np.nan_to_num(self.field("PERMZ") if "PERMZ" in self.arrays else self.field("PERMX")) * MD
        conn = g.connections(kx * ntg, ky * ntg, kz, active=active, pinch=self.pinch)

        mult = {d: f(d).copy() for d in ("MULTX", "MULTY", "MULTZ", "MULTX-", "MULTY-", "MULTZ-")}
        face_kw = {"X": "MULTX", "I": "MULTX", "X-": "MULTX-", "I-": "MULTX-", "Y": "MULTY", "J": "MULTY",
                   "Y-": "MULTY-", "J-": "MULTY-", "Z": "MULTZ", "K": "MULTZ", "Z-": "MULTZ-", "K-": "MULTZ-"}
        for pattern, m in self.multflt:
            names = [n for n in self.faults if fnmatch.fnmatchcase(n, pattern)]
            if not names:
                raise ValueError(f"MULTFLT: разлом {pattern} не задан в FAULTS")
            for n in names:
                for (i1, i2, j1, j2, k1, k2), face in self.faults[n]:
                    mult[face_kw[face]][i1 - 1:i2, j1 - 1:j2, k1 - 1:k2] *= m
        c1, c2, ax = conn["c1"], conn["c2"], conn["axis"]
        m = np.ones_like(conn["T"])
        for a, (plus, minus) in enumerate((("MULTX", "MULTX-"), ("MULTY", "MULTY-"), ("MULTZ", "MULTZ-"))):
            sel = ax == a
            m[sel] = mult[plus].reshape(-1)[c1[sel]] * mult[minus].reshape(-1)[c2[sel]]
        pairs = {}
        for rec in self.multregt:
            pairs[(rec["r1"], rec["r2"], rec["region"])] = rec
        for (r1, r2, reg), rec in pairs.items():
            arr = {"F": "FLUXNUM", "M": "MULTNUM", "O": "OPERNUM"}[reg]
            if arr not in self.arrays:
                raise ValueError(f"MULTREGT: нет массива {arr}")
            regs = self.field(arr).reshape(-1).astype(int)
            ra, rb = regs[c1], regs[c2]
            hit = ra != rb
            if r1 is not None:
                hit &= (ra == int(r1)) | (rb == int(r1))
            if r2 is not None:
                hit &= (ra == int(r2)) | (rb == int(r2))
            dirs = np.array(["X" in rec["dirs"], "Y" in rec["dirs"], "Z" in rec["dirs"]])
            hit &= dirs[ax]
            if rec["nnc"] == "NNC":
                hit &= conn["nnc"]
            elif rec["nnc"] == "NONNC":
                hit &= ~conn["nnc"]
            m[hit] *= rec["mult"]
        conn["T"] = conn["T"] * m
        return {"grid": g, "active": active, "pv": np.where(active, pv, 0.0), "conn": conn}

    def well_configs(self, step=0):
        u = self.u
        out = []
        controls = self.steps[step][1]
        for w in self.wells.values():
            conns = self.well_connections(w["name"], step)
            if not conns:
                raise ValueError(f"скважина {w['name']}: нет COMPDAT")
            cfg = {"name": w["name"], "i": w["i"], "j": w["j"], "k": conns[0]["ijk"][2],
                   "perforations": [list(c["ijk"]) for c in conns], "radius": conns[0]["rw"],
                   "skin": conns[0]["skin"], "connections": conns}
            if w["ref_depth"] is not None:
                cfg["ref_depth"] = w["ref_depth"] - self.depth_top
            cfg.update(self.control_cfg(w["name"], controls.get(w["name"])))
            out.append(cfg)
        return out

    def _cell_array(self, name):
        """Ячеечный массив deck в порядке симулятора (NaN — не задан)."""
        return self._a(np.asarray(self.field(name), dtype=np.float64))

    def _endpoint_model(self, krs, sat, device, prefix=""):
        """
        ScaledRelPerm: концевые точки таблиц региона sat (SATNUM, для пропитки — IMBNUM), поверх них — массивы
        SWL…PCG (prefix "I": ISWL…ISOGCR; вертикальное масштабирование пропитки — по таблицам) — только
        при ENDSCALE, как в Eclipse. Используется как ОФП (ENDSCALE/SWATINIT/HYSTER) и для обращения Pc.
        """
        from .endpoint_scaling import KEYS, ScaledRelPerm, table_endpoints
        if any(k.tables is None or "swof" not in k.tables for k in krs):
            raise NotImplementedError("для равновесия и масштабирования нужны таблицы SWOF")
        tab_ep = [table_endpoints(k.tables["swof"], k.tables.get("sgof")) for k in krs]
        tab_ep = [{key: float(v) for key, v in e.items()} for e in tab_ep]
        cell = {}
        for key in KEYS:
            v = np.array([e[key] for e in tab_ep])[sat]
            name = prefix + key
            if self.endscale and name in self.arrays and not (prefix and key.startswith(("KR", "PC"))):
                a = self._cell_array(name)
                v = np.where(np.isnan(a), v, a)
            cell[key] = torch.as_tensor(v, dtype=torch.float64, device=device)
        act = torch.as_tensor(self._a(self.active) if getattr(self, "active", None) is not None
                              else np.ones(len(sat), bool), device=device)
        bad = act & ((cell["SWL"] > cell["SWCR"] + 1e-12) | (cell["SWCR"] >= cell["SWU"])
                     | (cell["SGL"] > cell["SGCR"] + 1e-12))
        if self.endscale and bool(bad.any()):
            raise ValueError(f"ENDSCALE{' (' + prefix + ')' if prefix else ''}: нарушено SWL ≤ SWCR < SWU / "
                             f"SGL ≤ SGCR в {int(bad.sum())} активных ячейках")
        tables = [(k.tables["swof"], k.tables.get("sgof")) for k in krs]
        return ScaledRelPerm(tables, torch.as_tensor(sat, device=device), cell, tab_ep,
                             three_point=self.scalecrs, kro_model=self.kro_model)

    def _region_index(self, name):
        """Номер региона ячейки (0-based) в порядке симулятора; по умолчанию все в регионе 1."""
        if name not in self.arrays:
            return np.zeros(self.n_cells, dtype=np.int64)
        a = self.field(name).reshape(-1)
        act = getattr(self, "active", None)
        a = np.where(np.isnan(a), 1.0, a)
        if act is not None:
            a = np.where(act.reshape(-1), a, 1.0)
        a = self._a(a)
        if (a < 1).any():
            raise ValueError(f"{name}: номера регионов должны быть ≥ 1")
        return a.astype(np.int64) - 1

    def group_constraints(self, step):
        """Групповые цели шага step для groups.group_targets (дебиты — м³/сут, поверхностные или пластовые)."""
        u = self.u
        out = []
        for key, r in self.group_steps[step].items():
            g = key[0]
            mode = str(_get(r, 1 if key[1] == "prod" else 2, "NONE")).upper()
            if key[1] == "prod" and str(_get(r, 6, "NONE")).upper() == "RATE":
                # ограничения дебитов с действием RATE при превышении — группа переходит на этот дебит
                for m, ph, cv in ((2, "oil", u["liq"]), (3, "water", u["liq"]), (4, "gas", u["gas"]),
                                  (5, "liquid", u["liq"])):
                    lim = _get(r, m)
                    if lim is not None and float(lim) > 0 and {"ORAT": 2, "WRAT": 3, "GRAT": 4, "LRAT": 5}.get(mode) != m:
                        out.append(dict(group=g, kind="prod", phase=ph, target=float(lim) * cv))
            if mode in ("NONE", "FLD", "' '"):
                continue
            if key[1] == "prod":
                idx = {"ORAT": (2, "oil", u["liq"]), "WRAT": (3, "water", u["liq"]), "GRAT": (4, "gas", u["gas"]),
                       "LRAT": (5, "liquid", u["liq"]), "RESV": (13, "resv", u["liq"])}
                if mode not in idx:
                    raise NotImplementedError(f"GCONPROD {g}: режим {mode}")
                m, ph, cv = idx[mode]
                out.append(dict(group=g, kind="prod", phase=ph, target=float(r[m]) * cv))
            else:
                ph = {"WAT": "water", "GAS": "gas"}.get(key[2])
                if ph is None:
                    raise NotImplementedError(f"GCONINJE {g}: фаза {key[2]}")
                cv = u["gas"] if ph == "gas" else u["liq"]
                if mode == "RATE":
                    out.append(dict(group=g, kind="inj", phase=ph, target=float(r[3]) * cv))
                elif mode == "REIN":
                    out.append(dict(group=g, kind="inj", phase=ph, target=0.0, mode="REIN", frac=float(r[5]),
                                    src=str(_get(r, 10, g))))
                elif mode == "VREP":
                    out.append(dict(group=g, kind="inj", phase=ph, target=0.0, mode="VREP", frac=float(r[6]),
                                    src=str(_get(r, 11, g))))
                else:
                    raise NotImplementedError(f"GCONINJE {g}: режим {mode}")
        return out

    def control_cfg(self, name, control):
        """
        Режим скважины из записи WCONPROD/WCONINJE/WCONHIST/WCONINJH (единицы симулятора: м³/сут, МПа).
        Статус SHUT/STOP или отсутствие режима — control_type 'shut' (соединения закрыты, перетока нет).
        WCONHIST RESV: control_type 'resv', исторические поверхностные дебиты — в 'hist_rates'.
        """
        u = self.u
        if control is None:
            typ = self.well_type[name]
        else:
            typ = "producer" if control[0] in ("WCONPROD", "WCONHIST") else "injector"
        if control is None or control[2] in ("SHUT", "STOP"):
            return {"type": typ, "control_type": "shut", "control_value": 0.0}
        kw, r, status = control
        if status != "OPEN":
            raise NotImplementedError(f"{name}: статус {status} не поддерживается")
        p_atm = 14.7 if self.units_name == "FIELD" else 1.01325
        cfg = {"type": typ}
        if kw in ("WCONPROD", "WCONHIST"):
            mode = str(r[2]).upper()
            hist = kw == "WCONHIST"
            # WCONPROD: ORAT WRAT GRAT LRAT RESV BHP (3..8); WCONHIST: ORAT WRAT GRAT (3..5), наблюдённое BHP — 9
            qo, qw, qg = (float(_get(r, m, 0.0) or 0.0) for m in (3, 4, 5))
            ql = qo + qw if hist else float(_get(r, 6, 0.0) or 0.0)
            rates = {"ORAT": (qo, "oil", u["liq"]), "WRAT": (qw, "water", u["liq"]),
                     "GRAT": (qg, "gas", u["gas"]), "LRAT": (ql, "liquid", u["liq"])}
            bhp_idx = 9 if hist else 8
            if mode in rates:
                val, ph, cv = rates[mode]
                cfg.update(control_type="rate", control_value=val * cv, surface_phase=ph)
            elif mode == "RESV" and hist:
                cfg.update(control_type="resv", control_value=0.0, surface_phase="liquid",
                           hist_rates=(qo * u["liq"], qw * u["liq"], qg * u["gas"]))
            elif mode == "RESV":
                cfg.update(control_type="resv", control_value=float(r[7]) * u["liq"], surface_phase="liquid")
            elif mode == "BHP":
                cfg.update(control_type="bhp", control_value=float(r[bhp_idx]) * u["p"] / 1e6)
            elif mode == "GRUP" and not hist:
                cfg.update(control_type="rate", control_value=1e20, surface_phase="liquid", grup=True,
                           rate_limits={ph: v * cv for v, ph, cv in rates.values() if v > 0})
            elif mode == "THP" and not hist:
                # режим THP = управление по дебиту-ограничению, начатое на ограничении по THP (как в OPM)
                lim = next(((v, ph, cv) for key, (v, ph, cv) in rates.items() if v > 0), None)
                val, ph, cv = lim if lim is not None else (1e20, "liquid", 1.0)
                cfg.update(control_type="rate", control_value=val * cv, surface_phase=ph, start_mode="thp_limit")
            else:
                raise NotImplementedError(f"{kw} {name}: режим {mode}")
            if not hist:
                thp, table = _get(r, 9), int(_get(r, 10, 0) or 0)
                if mode == "THP" and (thp is None or table <= 0):
                    raise ValueError(f"WCONPROD {name}: режим THP без THP или таблицы VFP")
                if thp is not None and float(thp) > 0 and table > 0:
                    if table not in self.vfp:
                        raise ValueError(f"WCONPROD {name}: нет таблицы VFPPROD {table}")
                    cfg.update(thp_limit=float(thp) * u["p"] / 1e6, vfp_table=table,
                               alq=float(_get(r, 11, 0.0) or 0.0))
            # нулевая цель по дебиту — скважина остановлена (как stoppedOrZeroRateTarget в OPM)
            zero = max(qo, qw, qg) <= 0 if hist else (cfg["control_type"] in ("rate", "resv")
                                                       and not cfg.get("grup") and cfg["control_value"] <= 0)
            if cfg["control_type"] in ("rate", "resv") and zero:
                return {"type": typ, "control_type": "shut", "control_value": 0.0}
            # в истории ограничение по BHP — 1 атм (WELTARG не поддерживается)
            cfg["bhp_min"] = float(p_atm if hist else _get(r, bhp_idx, p_atm)) * u["p"] / 1e6
        else:
            ph = str(r[1]).upper()
            hist = kw == "WCONINJH"
            # WCONINJE: RATE 4, BHP 6; WCONINJH: RATE 3, BHP 4, режим 11 (RATE по умолчанию)
            mode = str(_get(r, 11, "RATE") if hist else r[3]).upper()
            cfg["injected_phase"] = {"WATER": "water", "WAT": "water", "GAS": "gas"}[ph]
            cv = u["gas"] if ph == "GAS" else u["liq"]
            rate_idx, bhp_idx = (3, 4) if hist else (4, 6)
            if mode == "RATE":
                cfg.update(control_type="rate", control_value=float(_get(r, rate_idx, 0.0) or 0.0) * cv)
                if cfg["control_value"] <= 0:
                    return {"type": typ, "control_type": "shut", "control_value": 0.0}
            elif mode == "BHP":
                cfg.update(control_type="bhp", control_value=float(r[bhp_idx]) * u["p"] / 1e6)
            elif mode == "GRUP" and not hist:
                lim = float(_get(r, rate_idx, 0.0) or 0.0) * cv
                cfg.update(control_type="rate", control_value=1e20, grup=True,
                           rate_limits={cfg["injected_phase"]: lim} if lim > 0 else {})
            elif mode == "THP" and not hist:
                val = float(_get(r, rate_idx, 0.0) or 0.0) * cv
                cfg.update(control_type="rate", control_value=val if val > 0 else 1e20, start_mode="thp_limit")
            else:
                raise NotImplementedError(f"{kw} {name}: режим {mode}")
            if not hist:
                # WCONINJE: THP 8, таблица VFPINJ 9
                thp, table = _get(r, 7), int(_get(r, 8, 0) or 0)
                if mode == "THP" and (thp is None or table <= 0):
                    raise ValueError(f"WCONINJE {name}: режим THP без THP или таблицы VFP")
                if thp is not None and float(thp) > 0 and table > 0:
                    if ("INJ", table) not in self.vfp:
                        raise ValueError(f"WCONINJE {name}: нет таблицы VFPINJ {table}")
                    cfg.update(thp_limit=float(thp) * u["p"] / 1e6, vfp_table=("INJ", table))
            lim = None if hist else _get(r, bhp_idx)
            cfg["bhp_max"] = float(lim if lim is not None else (1e5 if self.units_name == "FIELD" else 6895.0)) \
                * u["p"] / 1e6
        return cfg

    # ------------------------------------------------------------ равновесие
    def _equil_profiles(self, eq, rsvd, pvt, device):
        """Профили давлений фаз региона равновесия: функция profile(фаза, z0, p0, lo, hi) -> (z, p)."""
        if eq["goc"] > eq["woc"]:
            raise ValueError("EQUIL: ГНК глубже ВНК")
        g = 9.80665
        dt = torch.float64
        rho_o_sc, rho_w_sc, rho_g_sc = pvt.rho_sc

        def rs_at(z, p):
            rs_s = float(pvt.rs_sat(torch.tensor([p], dtype=dt, device=device)))
            if rsvd is None:
                return rs_s
            zs = torch.tensor([r[0] for r in rsvd], dtype=dt)
            vs = torch.tensor([r[1] for r in rsvd], dtype=dt)
            from .blackoil_props import interp1
            return min(float(interp1(torch.tensor([z], dtype=dt), zs, vs)), rs_s)

        def rho_phase(ph, z, p):
            pt = torch.tensor([p], dtype=dt, device=device)
            if ph == "o":
                rs = rs_at(z, p)
                sat = torch.tensor([rs >= float(pvt.rs_sat(pt)) - 1e-12], device=device)
                bo, _ = pvt.oil(pt, torch.tensor([rs], dtype=dt, device=device), sat)
                return float((rho_o_sc + rs * rho_g_sc) * bo)
            if ph == "w":
                return float(rho_w_sc * pvt.water(pt)[0])
            return float((rho_g_sc + float(pvt.rv_sat(pt)) * rho_o_sc) * pvt.gas(pt)[0])

        def integrate(ph, z0, p0, z1, step=0.25):
            """Профиль p(z) от z0 до z1 (RK2); возвращает (z, p) по возрастанию z."""
            n = max(1, int(np.ceil(abs(z1 - z0) / step)))
            h = (z1 - z0) / n
            zs, ps = [z0], [p0]
            z, p = z0, p0
            for _ in range(n):
                k1 = rho_phase(ph, z, p) * g
                k2 = rho_phase(ph, z + 0.5 * h, p + 0.5 * h * k1) * g
                p += h * k2
                z += h
                zs.append(z)
                ps.append(p)
            zs, ps = np.array(zs), np.array(ps)
            order = np.argsort(zs)
            return zs[order], ps[order]

        def profile(ph, z0, p0, lo, hi):
            za, pa = integrate(ph, z0, p0, min(lo, z0))
            zb, pb = integrate(ph, z0, p0, max(hi, z0))
            return np.concatenate([za, zb[1:]]), np.concatenate([pa, pb[1:]])

        return profile

    def equilibrate(self, pvts, ep, regs, device, swatinit=None):
        """
        Гидростатическое равновесие по EQUIL для каждого региона EQLNUM (по центрам ячеек, EQUIL item 9 = 0):
        давление нефти интегрируется от опорной глубины, воды — от ВНК (pw = po − Pcow на контакте),
        газа — от ГНК (pg = po + Pcog); плотности — по PVT региона (преобладающий PVTNUM). Насыщенности —
        обращением Pcow(Sw) = po − pw и Pcog(Sg) = pg − po по таблицам SATNUM ячейки; при нулевых Pc —
        резкие контакты. Pc и его обращение — с масштабированием концевых точек ячейки (ep).
        SWATINIT: Sw = SWATINIT, а PCW ячейки пересчитывается так, чтобы Pcow(SWATINIT) = po − pw (в ep), выше ГНК —
        pg − pw − Pcog(1 − SWATINIT), как PPCW Eclipse; где перепад ≤ 0 или Pc(SWATINIT) ≤ 0 — Sw из равновесия
        по исходному Pc.
        """
        rt = self.region_tables
        dt = torch.float64
        z_all = np.asarray(self.cell_depth_abs, dtype=np.float64).reshape(-1)
        n = z_all.size
        p_cell, sw, sg, rs, rv = (np.zeros(n) for _ in range(5))
        eqls = rt.get("equil", [self.equil])
        rsvds = rt.get("rsvd")
        for r in np.unique(regs["EQLNUM"]):
            if r >= len(eqls):
                raise ValueError(f"EQLNUM {r + 1}: нет записи EQUIL")
            eq = eqls[r]
            rsvd = rsvds[r] if rsvds is not None and r < len(rsvds) else None
            rvvds = rt.get("rvvd")
            rvvd = rvvds[r] if rvvds is not None and r < len(rvvds) and eq.get("rvvd") else None
            m = regs["EQLNUM"] == r
            pv_r = int(np.bincount(regs["PVTNUM"][m]).argmax())
            pvt = pvts[pv_r]
            pbvds = rt.get("pbvd")
            if rsvd is None and pbvds is not None and r < len(pbvds):
                # PBVD → RSVD: Rs(z) = Rs_sat(p_b(z)); с min(·, Rs_sat(p)) это Rs_sat(min(p_b, p))
                pb = torch.tensor([q[1] for q in pbvds[r]], dtype=torch.float64, device=device)
                rsvd = list(zip((q[0] for q in pbvds[r]), pvt.rs_sat(pb).tolist()))
            profile = self._equil_profiles(eq, rsvd, pvt, device)
            z = z_all[m]
            lo, hi = z.min() - 1.0, z.max() + 1.0
            has_woc, has_goc = eq["woc"] < 1e8, eq["goc"] > -1e8
            o_lo = min(lo, eq["goc"]) if has_goc else lo
            o_hi = max(hi, eq["woc"]) if has_woc else hi
            zw = zg = None
            # опорная глубина в водяной (газовой) зоне: сначала профиль этой фазы, нефть — от контакта
            if has_woc and eq["datum"] > eq["woc"]:
                zw, pw_n = profile("w", eq["datum"], eq["p"], min(lo, eq["woc"]), hi)
                zo, po_n = profile("o", eq["woc"], float(np.interp(eq["woc"], zw, pw_n)) + eq["pcwoc"], o_lo, o_hi)
            elif has_goc and eq["datum"] < eq["goc"]:
                zg, pg_n = profile("g", eq["datum"], eq["p"], lo, max(hi, eq["goc"]))
                zo, po_n = profile("o", eq["goc"], float(np.interp(eq["goc"], zg, pg_n)) - eq["pcgoc"], o_lo, o_hi)
            else:
                zo, po_n = profile("o", eq["datum"], eq["p"], o_lo, o_hi)
            po = np.interp(z, zo, po_n)
            pw = pg = None
            if has_woc:
                if zw is None:
                    p_w0 = float(np.interp(eq["woc"], zo, po_n)) - eq["pcwoc"]
                    zw, pw_n = profile("w", eq["woc"], p_w0, lo, hi)
                pw = np.interp(z, zw, pw_n)
            if has_goc:
                if zg is None:
                    p_g0 = float(np.interp(eq["goc"], zo, po_n)) + eq["pcgoc"]
                    zg, pg_n = profile("g", eq["goc"], p_g0, lo, hi)
                pg = np.interp(z, zg, pg_n)
            idx = np.flatnonzero(m)
            p_cell[idx], sw[idx], sg[idx] = self._equil_saturations(
                eq, ep, idx, z, po, pw, pg, None if swatinit is None else swatinit[idx])
            for pv in np.unique(regs["PVTNUM"][m]):
                k = regs["PVTNUM"][m] == pv
                pt = torch.tensor(p_cell[idx[k]], dtype=dt, device=device)
                rs_s = pvts[pv].rs_sat(pt).cpu().numpy()
                if rsvd is not None:
                    rr = np.minimum(np.interp(z[k], [q[0] for q in rsvd], [q[1] for q in rsvd]), rs_s)
                else:
                    rr = rs_s
                rs[idx[k]] = np.where(sg[idx[k]] > 0, rs_s, rr)
                # Rv без RVVD (EQUIL item 8 ≤ 0): насыщенный при давлении ячейки в газе (как OPM RvSatAtContact)
                rv_s = pvts[pv].rv_sat(pt).cpu().numpy()
                if rvvd is not None:
                    rvr = np.minimum(np.interp(z[k], [q[0] for q in rvvd], [q[1] for q in rvvd]), rv_s)
                    so = 1.0 - sw[idx[k]] - sg[idx[k]]
                    rv_s = np.where(so > 0, rv_s, rvr)
                rv[idx[k]] = rv_s
        to = lambda a: torch.tensor(a, dtype=dt, device=device)
        return to(p_cell), to(sw), to(sg), to(rs), to(rv)

    @staticmethod
    def _equil_saturations(eq, ep, idx, z, po, pw, pg, swatinit=None):
        """Насыщенности и давление в ячейках idx одного региона равновесия по профилям фаз."""
        swl, swu = ep.endpoint("SWL", idx), ep.endpoint("SWU", idx)
        sw = swl.copy()
        p_cell = po.copy()
        if pw is not None:
            dp = po - pw
            inv = ep.invert_pc(idx, "w", dp)
            sw = np.where(np.isnan(inv), np.where(z > eq["woc"], swu, swl), inv)
            if swatinit is not None:
                s0 = np.clip(np.where(np.isnan(swatinit), sw, swatinit), swl, swu)
                pc0 = ep.pc_value(idx, "w", s0)
                if pg is not None and ep.tables[0][1] is not None:
                    # выше ГНК вода равновесна с газом: Pcow(SWATINIT) = pg − pw − Pcog(1 − SWATINIT)
                    gz = z < eq["goc"]
                    pcg0 = ep.pc_value(idx, "g", np.clip(1.0 - s0, 0.0, None))
                    dp = np.where(gz, pg - pw - pcg0, dp)
                ok = ~np.isnan(swatinit) & (dp > 0) & (pc0 > 0)
                pcw = ep.endpoint("PCW", idx)
                pcw_new = np.where(ok, pcw * dp / np.where(ok, pc0, 1.0), pcw)
                ep.cell["PCW"][torch.as_tensor(idx, device=ep.cell["PCW"].device)] = torch.as_tensor(
                    pcw_new, dtype=torch.float64, device=ep.cell["PCW"].device)
                # где Pc нельзя подстроить (po − pw ≤ 0 или Pc(SWATINIT) ≤ 0) — Sw из равновесия: без перетоков в t=0
                sw = np.where(ok, s0, sw)
            water = sw >= swu - 1e-12
            p_cell = np.where(water, pw + ep.pc_value(idx, "w", sw), p_cell)
        sg = np.zeros_like(z)
        if pg is not None:
            inv = ep.invert_pc(idx, "g", pg - po) if ep.tables[0][1] is not None else np.full(len(idx), np.nan)
            sgu = ep.endpoint("SGU", idx)
            sg = np.where(np.isnan(inv), np.where(z < eq["goc"], np.minimum(sgu, 1.0 - swl), 0.0), inv)
            sg = np.clip(np.minimum(sg, 1.0 - sw), 0.0, None)
            gas = sg >= 1.0 - sw - 1e-12
            pcg = ep.pc_value(idx, "g", sg) if ep.tables[0][1] is not None else np.zeros(len(idx))
            p_cell = np.where(gas & (sg > 0), pg - pcg, p_cell)
        return p_cell, sw, sg


