"""
Чтение deck-файлов Eclipse (*.DATA) и сборка модели для black-oil FIM.

Поддерживается подмножество, достаточное для SPE1-подобных моделей:
  RUNSPEC: DIMENS, FIELD | METRIC, OIL/WATER/GAS/DISGAS
  GRID:    DX, DY, DZ, TOPS, PORO, PERMX/Y/Z, NTG, BOX/ENDBOX (декартова сетка, горизонтальные слои)
  PROPS:   PVTO | PVDO, PVDG, PVTW, ROCK, DENSITY, SWOF, SGOF, STONE2
  SOLUTION: EQUIL (контакты при нулевом Pc), RSVD
  SCHEDULE: WELSPECS, COMPDAT, WCONPROD, WCONINJE, TSTEP, DRSDT
Неподдерживаемые ключевые слова, влияющие на результат, вызывают ошибку, а не игнорируются.
"""
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
IGNORED = {"TITLE", "START", "EQLDIMS", "TABDIMS", "WELLDIMS", "RPTSCHED", "RPTRST", "RPTSOL", "RPTPROPS",
           "RPTGRID", "RPTRUNSP", "NSTACK", "MESSAGES"}
# влияют на физику, но пока не поддерживаются
UNSUPPORTED = {"ACTNUM", "COORD", "ZCORN", "MULTX", "MULTY", "MULTZ", "FAULTS", "PVTG", "VAPOIL", "SWFN",
               "SGFN", "SOF3", "SLGOF", "AQUFETP", "AQUCT", "PBVD", "PDVD", "RVVD", "DATES", "WCONHIST",
               "GCONPROD", "GCONINJE", "TUNING", "EQUALS", "MULTIPLY", "COPY", "ADD", "PRESSURE", "SWAT",
               "SGAS", "RS", "NNC", "DRVDT", "ROCKTAB", "SATNUM", "PVTNUM", "EQLNUM", "WECON", "WTEST"}
ARRAY_KEYWORDS = {"DX", "DY", "DZ", "TOPS", "PORO", "PERMX", "PERMY", "PERMZ", "NTG"}

UNITS = {
    "FIELD": dict(p=6894.757293168, L=0.3048, liq=0.158987294928, gas=28.316846592,
                  rho=16.01846337396, comp=1.0 / 6894.757293168),
    "METRIC": dict(p=1e5, L=1.0, liq=1.0, gas=1.0, rho=1.0, comp=1e-5),
}
MD = 9.869233e-16
DAY = 86400.0

_KW_RE = re.compile(r"^([A-Z][A-Z0-9_]{0,7})\s*$")
_TOK_RE = re.compile(r"'[^']*'|/|[^\s/']+")


def _strip_comment(line):
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


def parse_deck(path):
    """Возвращает список (KEYWORD, [record, ...]); record — список значений (float | str | None)."""
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
        for tok in _TOK_RE.findall("\n".join(body)):
            if tok == "/":
                records.append(cur)
                cur = []
            else:
                cur.extend(_expand(tok))
        if cur:
            records.append(cur)
        while records and records[-1] == [] and len(records) > 1:
            records.pop()
        out.append((kw, records))
    return out


def _get(rec, idx, default=None):
    return rec[idx] if idx < len(rec) and rec[idx] is not None else default


class EclipseCase:
    """Модель, собранная из deck: сетка, свойства, начальное состояние, скважины, расписание (в СИ)."""

    def __init__(self, path):
        self.path = str(path)
        self.deck = parse_deck(path)
        kws = [k for k, _ in self.deck]
        bad = sorted(set(kws) & UNSUPPORTED)
        if bad:
            raise NotImplementedError(f"{Path(path).name}: ключевые слова пока не поддерживаются: {', '.join(bad)}")
        self.units_name = "METRIC"
        for k in kws:
            if k in UNITS:
                self.units_name = k
        if "LAB" in kws:
            raise NotImplementedError("единицы LAB не поддерживаются")
        self.u = UNITS[self.units_name]
        self.kro_model = "stone2" if ("STONE2" in kws or "STONE" in kws) else "default"
        if "STONE1" in kws:
            raise NotImplementedError("STONE1 не поддерживается")
        self._read()

    # ------------------------------------------------------------ разбор
    def _read(self):
        u = self.u
        self.arrays = {}
        box = None
        self.wells = {}
        self.report_days = []
        self.drsdt = None
        self.tables = {}
        self.equil = None
        self.rsvd = None
        schedule_started = False
        section = None
        unknown = set()
        for kw, recs in self.deck:
            if kw in ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE"):
                section = kw
            if section == "SUMMARY" or (kw in FLAG_KEYWORDS and kw != "ENDBOX") or kw in IGNORED:
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
            elif kw == "PVTO":
                self.tables["pvto"] = self._pvto(recs)
            elif kw == "PVDO":
                self.tables["pvdo"] = [(r[0] * u["p"], r[1], r[2] * 1e-3) for r in self._rows(recs[0], 3)]
            elif kw == "PVDG":
                bg = u["liq"] / u["gas"]
                self.tables["pvdg"] = [(r[0] * u["p"], r[1] * bg, r[2] * 1e-3) for r in self._rows(recs[0], 3)]
            elif kw == "PVTW":
                r = recs[0]
                self.tables["pvtw"] = (r[0] * u["p"], r[1], _get(r, 2, 0.0) * u["comp"], _get(r, 3, 0.5) * 1e-3,
                                       _get(r, 4, 0.0) * u["comp"])
            elif kw == "ROCK":
                self.rock = (recs[0][0] * u["p"], _get(recs[0], 1, 0.0) * u["comp"])
            elif kw == "DENSITY":
                r = recs[0]
                self.density = tuple(float(v) * u["rho"] for v in r[:3])
            elif kw == "SWOF":
                rows = self._rows(recs[0], 4)
                self.tables["swof"] = {"sw": [r[0] for r in rows], "krw": [r[1] for r in rows],
                                       "kro": [r[2] for r in rows], "pcow": [r[3] * u["p"] for r in rows]}
            elif kw == "SGOF":
                rows = self._rows(recs[0], 4)
                self.tables["sgof"] = {"sg": [r[0] for r in rows], "krg": [r[1] for r in rows],
                                       "kro": [r[2] for r in rows], "pcog": [r[3] * u["p"] for r in rows]}
            elif kw == "EQUIL":
                r = recs[0]
                self.equil = dict(datum=r[0] * u["L"], p=r[1] * u["p"], woc=_get(r, 2, 1e9 / u["L"]) * u["L"],
                                  pcwoc=_get(r, 3, 0.0) * u["p"], goc=_get(r, 4, -1e9 / u["L"]) * u["L"],
                                  pcgoc=_get(r, 5, 0.0) * u["p"])
            elif kw == "RSVD":
                rows = self._rows(recs[0], 2)
                self.rsvd = [(r[0] * u["L"], r[1] * u["gas"] / u["liq"]) for r in rows]
            elif kw == "DRSDT":
                self.drsdt = float(recs[0][0]) * u["gas"] / u["liq"] / DAY
            elif kw == "WELSPECS":
                for r in recs:
                    if not r:
                        continue
                    self.wells[r[0]] = dict(name=r[0], i=int(r[2]) - 1, j=int(r[3]) - 1,
                                            ref_depth=None if _get(r, 4) is None else r[4] * u["L"],
                                            phase=str(_get(r, 5, "OIL")).upper(), perfs=[], control=None)
            elif kw == "COMPDAT":
                for r in recs:
                    if not r:
                        continue
                    w = self._well(r[0])
                    i = int(_get(r, 1, 0) or 0) - 1
                    j = int(_get(r, 2, 0) or 0) - 1
                    i = w["i"] if i < 0 else i
                    j = w["j"] if j < 0 else j
                    if str(_get(r, 5, "OPEN")).upper() != "OPEN":
                        raise NotImplementedError(f"COMPDAT {r[0]}: закрытые перфорации не поддерживаются")
                    if _get(r, 9) is not None:
                        raise NotImplementedError(f"COMPDAT {r[0]}: Kh пока не поддерживается")
                    for k in range(int(r[3]) - 1, int(r[4])):
                        w["perfs"].append(dict(ijk=(i, j, k), cf=_get(r, 7), diam=_get(r, 8), skin=_get(r, 10, 0.0)))
            elif kw in ("WCONPROD", "WCONINJE"):
                if schedule_started:
                    raise NotImplementedError(f"{kw} после TSTEP: смена режимов по расписанию пока не поддерживается")
                for r in recs:
                    if r:
                        self._well(r[0])["control"] = (kw, r)
            elif kw == "TSTEP":
                schedule_started = True
                self.report_days.extend(float(v) for rec in recs for v in rec)
            else:
                unknown.add(kw)
        if unknown:
            print(f"  ⚠ deck: проигнорированы ключевые слова: {', '.join(sorted(unknown))}")

        for name, req in (("DX", "DX"), ("DY", "DY"), ("DZ", "DZ"), ("TOPS", "TOPS"), ("PORO", "PORO"),
                          ("PERMX", "PERMX")):
            if req not in self.arrays:
                raise ValueError(f"в deck нет {req}")
        for key in ("pvdg", "pvtw"):
            if key not in self.tables:
                raise ValueError(f"в deck нет {key.upper()}")
        if "pvto" not in self.tables and "pvdo" not in self.tables:
            raise ValueError("в deck нет PVTO или PVDO")
        if self.equil is None:
            raise NotImplementedError("начальное состояние поддерживается только через EQUIL")

    def _well(self, name):
        if name not in self.wells:
            raise ValueError(f"скважина {name} не объявлена в WELSPECS")
        return self.wells[name]

    @staticmethod
    def _rows(rec, width):
        vals = [0.0 if v is None else float(v) for v in rec]
        return [vals[i:i + width] for i in range(0, len(vals) - len(vals) % width, width)]

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
        arr = self.arrays.setdefault(kw, np.full((nz, ny, nx), np.nan))
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
        L = u["L"]
        dx3, dy3, dz3 = (self.field(k) * L for k in ("DX", "DY", "DZ"))
        dx, dy, dz = dx3[:, 0, 0], dy3[0, :, 0], dz3[0, 0, :]
        for name, full, vec, shape in (("DX", dx3, dx, (nx, 1, 1)), ("DY", dy3, dy, (1, ny, 1)),
                                       ("DZ", dz3, dz, (1, 1, nz))):
            if not np.allclose(full, vec.reshape(shape)):
                raise NotImplementedError(f"{name} меняется не только вдоль своей оси — нужна corner-point сетка")
        tops = self.field("TOPS")[:, :, 0] * L
        if not np.allclose(tops, tops[0, 0]):
            raise NotImplementedError("TOPS не постоянен: наклонные слои пока не поддерживаются")
        self.depth_top = float(tops[0, 0])

        poro = self.field("PORO")
        if "NTG" in self.arrays:
            poro = poro * self.field("NTG")
        kx = self.field("PERMX")
        ky = self.field("PERMY") if "PERMY" in self.arrays else kx
        kz = self.field("PERMZ") if "PERMZ" in self.arrays else kx
        res = Reservoir({"dimensions": [nx, ny, nz], "grid": {"dx": dx.tolist(), "dy": dy.tolist(),
                                                               "dz": dz.tolist()},
                         "porosity": poro.tolist(), "permeability": kx.tolist(), "k_vertical_fraction": 1.0},
                        device=device)
        res.permeability_y = torch.as_tensor(ky * MD, dtype=torch.float64, device=device)
        res.permeability_z = torch.as_tensor(kz * MD, dtype=torch.float64, device=device)

        rho = self.density
        pvt = DeckPVT(self.tables, rho, device)
        rp_tables = {k: self.tables[k] for k in ("swof", "sgof") if k in self.tables}
        kr = ThreePhaseRelPerm(tables=rp_tables, device=device, kro_model=self.kro_model)
        swc = float(self.tables["swof"]["sw"][0]) if "swof" in self.tables else 0.0
        p_ref, c_r = getattr(self, "rock", (1e5, 0.0))

        fluid = Fluid({"pressure": self.equil["p"] / 1e6, "s_w": swc, "s_g": 0.0, "rho_oil": rho[0],
                       "rho_water": rho[1], "rho_gas": rho[2], "c_rock": c_r * 1e6}, res, device=device)
        fluid.blackoil_pvt = pvt
        fluid.blackoil_relperm = kr
        fluid.rock_ref_pressure = p_ref
        fluid.rock_compressibility = c_r
        fluid.sw_cr = swc
        p, sw, sg, rs = self.equilibrate(res, pvt, kr, swc, device)
        shp = (nx, ny, nz)
        fluid.pressure = p.view(shp).clone()
        fluid.s_w = sw.view(shp).clone()
        fluid.s_g = sg.view(shp).clone()
        fluid.s_o = (1.0 - sw - sg).view(shp).clone()
        fluid.rs = rs.view(shp).clone()
        fluid.prev_pressure = fluid.pressure.clone()

        wm = WellManager(self.well_configs(), res)
        sim = {"solver_type": "fully_implicit", "fim_model": "blackoil", "drsdt": self.drsdt,
               "initial_time_step_days": 1.0, "newton_max_iter": 20, "max_time_step_attempts": 10}
        sim.update(sim_overrides or {})
        return res, fluid, wm, sim

    def well_configs(self):
        u = self.u
        out = []
        for w in self.wells.values():
            if not w["perfs"] or w["control"] is None:
                raise ValueError(f"скважина {w['name']}: нет COMPDAT или режима")
            kw, r = w["control"]
            diam = {pf["diam"] for pf in w["perfs"]}
            skins = {pf["skin"] for pf in w["perfs"]}
            cfs = {pf["cf"] for pf in w["perfs"]}
            if len(diam) > 1 or None in diam or len(skins) > 1:
                raise NotImplementedError(f"{w['name']}: разные диаметры/скины по перфорациям или диаметр по умолчанию")
            cfg = {"name": w["name"], "i": w["i"], "j": w["j"], "k": w["perfs"][0]["ijk"][2],
                   "perforations": [list(pf["ijk"]) for pf in w["perfs"]], "radius": diam.pop() * u["L"] / 2,
                   "skin": skins.pop()}
            if cfs != {None}:
                if len(cfs) > 1 or None in cfs:
                    raise NotImplementedError(f"{w['name']}: CF задан не для всех перфораций одинаково")
                cf = cfs.pop()
                conv = (u["liq"] / DAY) * 1e-3 / u["p"]
                cfg["well_index"] = cf * conv
            if w["ref_depth"] is not None:
                cfg["ref_depth"] = w["ref_depth"] - self.depth_top
            status = str(_get(r, 1 if kw == "WCONPROD" else 2, "OPEN")).upper()
            if status != "OPEN":
                raise NotImplementedError(f"{w['name']}: статус {status} не поддерживается")
            if kw == "WCONPROD":
                mode = str(r[2]).upper()
                cfg["type"] = "producer"
                rates = {"ORAT": (3, "oil", u["liq"]), "WRAT": (4, "water", u["liq"]),
                         "GRAT": (5, "gas", u["gas"]), "LRAT": (6, "liquid", u["liq"])}
                if mode in rates:
                    idx, ph, cv = rates[mode]
                    cfg.update(control_type="rate", control_value=float(r[idx]) * cv, surface_phase=ph)
                elif mode == "BHP":
                    cfg.update(control_type="bhp", control_value=float(r[8]) * u["p"] / 1e6)
                else:
                    raise NotImplementedError(f"WCONPROD {w['name']}: режим {mode}")
                cfg["bhp_min"] = float(_get(r, 8, 14.7 if self.units_name == "FIELD" else 1.01325)) * u["p"] / 1e6
            else:
                ph = str(r[1]).upper()
                mode = str(r[3]).upper()
                cfg["type"] = "injector"
                cfg["injected_phase"] = {"WATER": "water", "WAT": "water", "GAS": "gas"}[ph]
                cv = u["gas"] if ph == "GAS" else u["liq"]
                if mode == "RATE":
                    cfg.update(control_type="rate", control_value=float(r[4]) * cv)
                elif mode == "BHP":
                    cfg.update(control_type="bhp", control_value=float(r[6]) * u["p"] / 1e6)
                else:
                    raise NotImplementedError(f"WCONINJE {w['name']}: режим {mode}")
                cfg["bhp_max"] = float(_get(r, 6, 1e5 if self.units_name == "FIELD" else 6895.0)) * u["p"] / 1e6
            out.append(cfg)
        return out

    # ------------------------------------------------------------ равновесие
    def equilibrate(self, res, pvt, kr, swc, device):
        """
        Гидростатическое равновесие по EQUIL при нулевом капиллярном давлении (резкие контакты):
        давление нефти интегрируется от опорной глубины, воды — от ВНК, газа — от ГНК.
        """
        eq = self.equil
        if eq["pcwoc"] or eq["pcgoc"]:
            raise NotImplementedError("EQUIL: ненулевое Pc на контактах не поддерживается")
        swof, sgof = self.tables.get("swof"), self.tables.get("sgof")
        if (swof and any(swof["pcow"])) or (sgof and any(sgof["pcog"])):
            raise NotImplementedError("EQUIL с ненулевыми капиллярными кривыми пока не поддерживается")
        if not (eq["goc"] <= eq["datum"] <= eq["woc"]):
            raise NotImplementedError("EQUIL: опорная глубина должна лежать в нефтяной зоне")
        g = 9.80665
        dt = torch.float64
        rho_o_sc, rho_w_sc, rho_g_sc = pvt.rho_sc

        def rs_at(z, p):
            rs_s = float(pvt.rs_sat(torch.tensor([p], dtype=dt, device=device)))
            if self.rsvd is None:
                return rs_s
            zs = torch.tensor([r[0] for r in self.rsvd], dtype=dt)
            vs = torch.tensor([r[1] for r in self.rsvd], dtype=dt)
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
            return float(rho_g_sc * pvt.gas(pt)[0])

        def integrate(ph, z0, p0, z1, step=0.25):
            n = max(1, int(abs(z1 - z0) / step))
            h = (z1 - z0) / n
            z, p = z0, p0
            for _ in range(n):
                k1 = rho_phase(ph, z, p) * g
                k2 = rho_phase(ph, z + 0.5 * h, p + 0.5 * h * k1) * g
                p += h * k2
                z += h
            return p

        z = self.depth_top + res.z_centers.detach().cpu().double().numpy()
        p_woc = integrate("o", eq["datum"], eq["p"], eq["woc"]) if eq["woc"] < 1e8 else None
        p_goc = integrate("o", eq["datum"], eq["p"], eq["goc"]) if eq["goc"] > -1e8 else None
        p_l, sw_l, sg_l, rs_l = [], [], [], []
        sw_max = float(swof["sw"][-1]) if swof else 1.0
        for zk in z:
            if p_woc is not None and zk > eq["woc"]:
                p = integrate("w", eq["woc"], p_woc, zk)
                p_l.append(p); sw_l.append(sw_max); sg_l.append(0.0); rs_l.append(rs_at(zk, p))
            elif p_goc is not None and zk < eq["goc"]:
                p = integrate("g", eq["goc"], p_goc, zk)
                p_l.append(p); sw_l.append(swc); sg_l.append(1.0 - swc)
                rs_l.append(float(pvt.rs_sat(torch.tensor([p], dtype=dt, device=device))))
            else:
                p = integrate("o", eq["datum"], eq["p"], zk)
                p_l.append(p); sw_l.append(swc); sg_l.append(0.0); rs_l.append(rs_at(zk, p))
        nx, ny, nz = self.dims
        col = lambda v: torch.tensor(v, dtype=dt, device=device).view(1, 1, nz).expand(nx, ny, nz).reshape(-1)
        return col(p_l), col(sw_l), col(sg_l), col(rs_l)

