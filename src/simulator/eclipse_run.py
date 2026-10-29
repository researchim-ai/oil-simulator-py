"""Расчёт deck Eclipse по расписанию TSTEP со сводкой (векторы SUMMARY) в единицах deck."""
import contextlib
import io
import os
import time

import numpy as np
import torch

from .eclipse_deck import EclipseCase, DAY
from .simulation import Simulator

_PHASE = {"O": "oil", "W": "water", "G": "gas"}


def apply_controls(case, step, wm, solver):
    """Переводит скважины на режимы отчётного шага step (смена WCONPROD/WCONINJE по расписанию)."""
    from .groups import GroupTree
    controls = case.steps[step][1]
    fw = getattr(solver, "wells", None)
    if fw is not None and hasattr(fw, "groups"):
        fw.groups = (GroupTree(case.group_parent, case.well_group), case.group_constraints(step))
    if fw is not None and hasattr(fw, "vfp"):
        fw.vfp, fw.depth_top = case.vfp_steps[step], getattr(case, "depth_top", 0.0)
        fw.liftopt, fw.wliftopt = case.liftopt_steps[step], case.wliftopt_steps[step]
        fw.alq_si = case.u["gas"] / 86400.0
        fw.gliftopt = case.gliftopt_steps[step]
        fw.network = case.network_steps[step]
    for idx, w in enumerate(wm.wells):
        w.connection_order = case.compord_steps[step].get(w.name, "TRACK")
        cfg = case.control_cfg(w.name, controls.get(w.name))
        w.stopped = bool(cfg.get("stopped", False))
        w.thp_limit, w.vfp_table, w.alq = cfg.get("thp_limit"), cfg.get("vfp_table"), cfg.get("alq", 0.0)
        w.eff = case.wefac_steps[step].get(w.name, 1.0)
        retype = cfg["type"] != w.type or (
            cfg["type"] == "injector" and cfg.get("injected_phase", w.injected_phase) != w.injected_phase)
        if retype and cfg["control_type"] != "shut":
            if fw is None or not hasattr(fw, "set_type"):
                raise NotImplementedError(f"{w.name}: смена типа скважины или фазы закачки не поддерживается")
            fw.set_type(fw.wells.index(w), cfg["type"], cfg.get("injected_phase"), cfg.get("surface_phase"))
        new = (cfg["control_type"], cfg["control_value"], cfg.get("surface_phase"), cfg.get("bhp_min"),
               cfg.get("bhp_max"), cfg.get("hist_rates"), cfg.get("start_mode"))
        old = (w.control_type, w.control_value, w.surface_phase, w.bhp_min, w.bhp_max,
               getattr(w, "hist_rates", None), getattr(w, "start_mode", None))
        w.grup = bool(cfg.get("grup", False))
        w.rate_limits = cfg.get("rate_limits") or {}
        if fw is not None and getattr(w, "connections", None) is not None and hasattr(fw, "set_connections"):
            fw.set_connections(fw.wells.index(w), case.well_connections(w.name, step))
        if fw is not None and hasattr(fw, "set_open"):
            # скважина без открытых соединений закрыта (как в Eclipse/OPM)
            is_open = cfg["control_type"] != "shut"
            if hasattr(fw, "has_open_connections"):
                is_open = is_open and fw.has_open_connections(fw.wells.index(w))
            fw.set_open(fw.wells.index(w), is_open)
        if new == old:
            continue
        if cfg.get("surface_phase") != w.surface_phase and fw is not None and hasattr(fw, "rate_w"):
            fw.set_surface_phase(fw.wells.index(w), cfg.get("surface_phase"))
        w.control_type, w.control_value, w.surface_phase, w.bhp_min, w.bhp_max, w.hist_rates, w.start_mode = new
        if fw is not None and hasattr(fw, "mode"):
            fw.mode[fw.wells.index(w)] = w.start_mode or w.control_type


class WellTestState:
    """
    Закрытия скважин симулятором и их проверки (WTEST), как в Eclipse/OPM:
    открытая скважина на контроле по забойному давлению, которая за отчётный шаг ничего не отобрала
    (нагнетательная — не закачала), закрывается по причине 'P'. Закрытую скважину раз в интервал WTEST
    открывают на пробу (не больше заданного числа проверок); если она снова не работает — закрывают
    до следующей проверки. Закрытие снимает только явный WELOPEN OPEN, повтор WCONHIST/WCONPROD — нет.
    """

    def __init__(self):
        self.closed = {}     # {скважина: dict(reason, t, tests)}
        self.testing = set()
        self.events = []     # [(t сут, скважина, 'close' | 'test' | 'reopen')]

    def substep(self, case, step, t, wm, fw):
        """
        После каждого шага по времени (как OPM): экономические пределы WECON — добывающая с дебитом ниже
        минимального (или с превышением отношения при процедуре WELL) закрывается по причине 'E'; закрытая
        по 'E' открывается на пробу, когда прошёл интервал WTEST, и остаётся открытой, если пределы выполнены.
        """
        wecon, wtest = case.wecon_steps[step], case.wtest_steps[step]
        for w in wm.wells:
            cfg = wecon.get(w.name)
            if cfg is None or w.type != "producer" or w.control_type == "shut":
                continue
            if w.name in self.closed and w.name not in self.testing:
                continue
            k = fw.wells.index(w)
            if not bool((fw.conn_open[fw.conn_well == k] > 0).any()):
                continue
            q = getattr(w, "last_surface_rates", None) or {}
            if self._violates(cfg, q):
                prev = self.closed.get(w.name)
                self.closed[w.name] = dict(reason="E", t=t, tests=prev["tests"] if prev else 0)
                self.testing.discard(w.name)
                fw.set_open(k, False)
                self.events.append((t, w.name, "close"))
            elif w.name in self.testing:
                self.testing.discard(w.name)
                self._reopen(t, w.name)
        for w in wm.wells:
            st, wt = self.closed.get(w.name), wtest.get(w.name)
            if st is None or st["reason"] != "E" or w.name in self.testing or wt is None or "E" not in wt["reasons"] \
                    or w.control_type == "shut":
                continue
            if (wt["tests"] == 0 or st["tests"] < wt["tests"]) and t - st["t"] >= wt["interval"] - 1e-9:
                st["tests"] += 1
                self.testing.add(w.name)
                fw.set_open(fw.wells.index(w), True)
                self.events.append((t, w.name, "test"))

    def apply_cutbacks(self, case, step, wm, fw):
        """Сокращения WCUTBACK, накопленные раньше, переживают смену шага (сброс — новым WCONPROD/WCUTBACK)."""
        cut = self.__dict__.setdefault("cut", {})
        for name in case.cutback_reset_steps[step]:
            cut.pop(name, None)
        for w in wm.wells:
            c = cut.get(w.name)
            if c is not None and w.control_type != "shut":
                self._rate_control(w, fw, c["phase"], c["value"])
                w.rate_limits = {ph: v * c["mult"] for ph, v in w.rate_limits.items()}

    def cutback(self, case, step, t, wm, fw):
        """
        WCUTBACK после каждого шага по времени (как Eclipse): добывающая, у которой отношение (WCT, GOR, GLR, WGR)
        выше предела или BHP/THP ниже предела, переводится на дебит = текущий × коэффициент; на контроле по
        дебиту — по своей фазе, на контроле по давлению — по фазе из WCUTBACK. Сокращение не отменяется.
        """
        cut = self.__dict__.setdefault("cut", {})
        for w in wm.wells:
            cfg = case.wcutback_steps[step].get(w.name)
            if cfg is None or w.type != "producer" or w.control_type == "shut":
                continue
            k = fw.wells.index(w)
            if not bool((fw.conn_open[fw.conn_well == k] > 0).any()):
                continue
            q = {ph: max(v, 0.0) for ph, v in (getattr(w, "last_surface_rates", None) or {}).items()}
            qo, qw, qg = q.get("oil", 0.0), q.get("water", 0.0), q.get("gas", 0.0)
            ratio = lambda a, b: a / b if b > 0 else 0.0
            bhp, thp = getattr(w, "bhp_mpa", None), getattr(w, "thp_mpa", None)
            hit = any(cfg[k_] and v > cfg[k_] for k_, v in (
                ("wct", ratio(qw, qo + qw)), ("gor", ratio(qg, qo)), ("glr", ratio(qg, qo + qw)),
                ("wgr", ratio(qw, qg))))
            hit = hit or (cfg["bhp"] and bhp is not None and bhp < cfg["bhp"]) \
                or (cfg["thp"] and thp is not None and thp < cfg["thp"])
            if not hit:
                continue
            if w.grup:
                raise NotImplementedError(f"WCUTBACK {w.name}: скважина под групповым управлением (GRUP)")
            mode = fw.mode[k] if hasattr(fw, "mode") else w.control_type
            own = w.surface_phase if w.control_type == "rate" else None
            phase = (own if mode == "rate" else cfg["phase"] or own) or "liquid"
            prev = cut.get(w.name, {"mult": 1.0})
            cut[w.name] = dict(phase=phase, value=q.get(phase, 0.0) * cfg["factor"],
                               mult=prev["mult"] * cfg["factor"])
            self._rate_control(w, fw, phase, cut[w.name]["value"])
            w.rate_limits = {ph: v * cfg["factor"] for ph, v in w.rate_limits.items()}
            self.events.append((t, w.name, "cutback"))

    @staticmethod
    def _rate_control(w, fw, phase, value):
        k = fw.wells.index(w)
        if phase != w.surface_phase and hasattr(fw, "rate_w"):
            fw.set_surface_phase(k, phase)
        w.control_type, w.control_value, w.surface_phase, w.start_mode = "rate", value, phase, None
        if hasattr(fw, "mode"):
            fw.mode[k] = "rate"

    @staticmethod
    def _violates(cfg, q):
        qo, qw, qg = (max(q.get(ph, 0.0), 0.0) for ph in ("oil", "water", "gas"))
        if any(cfg[k] and v < cfg[k] for k, v in (("oil", qo), ("gas", qg), ("liq", qo + qw))):
            return True
        if cfg["work"] != "WELL":
            return False
        ratio = lambda a, b: a / b if b > 0 else 0.0
        return any(cfg[k] and v > cfg[k] for k, v in (("wct", ratio(qw, qo + qw)), ("gor", ratio(qg, qo)),
                                                     ("wgr", ratio(qw, qg)), ("glr", ratio(qg, qo + qw))))

    def before_step(self, case, step, t, wm, fw):
        for name in case.reopen_steps[step]:
            self.closed.pop(name, None)
        self.testing = set()
        wtest = case.wtest_steps[step]
        for idx, w in enumerate(wm.wells):
            st = self.closed.get(w.name)
            if st is None:
                continue
            wt = wtest.get(w.name)
            if (wt is not None and st["reason"] in wt["reasons"] and (wt["tests"] == 0 or st["tests"] < wt["tests"])
                    and t - st["t"] >= wt["interval"] - 1e-9):
                st["tests"] += 1
                self.testing.add(w.name)
                self.events.append((t, w.name, "test"))
            else:
                fw.set_open(fw.wells.index(w), False)

    def after_step(self, case, step, t, wm, fw):
        for idx, w in enumerate(wm.wells):
            if w.name in self.closed and w.name not in self.testing:
                continue
            k = fw.wells.index(w)
            if not bool((fw.conn_open[fw.conn_well == k] > 0).any()) or fw.step_mode[k] not in ("bhp", "bhp_limit", "thp_limit"):
                if w.name in self.testing:
                    self._reopen(t, w.name)
                continue
            q = getattr(w, "last_surface_rates", None) or {}
            if w.type == "producer":
                alive = max(q.get(ph, 0.0) for ph in ("oil", "water", "gas")) > 0.0
            else:
                alive = -q.get((w.injected_phase or "water").lower(), 0.0) > 0.0
            if alive:
                if w.name in self.testing:
                    self._reopen(t, w.name)
                continue
            prev = self.closed.get(w.name)
            self.closed[w.name] = dict(reason="P", t=t, tests=prev["tests"] if prev else 0)
            self.events.append((t, w.name, "close"))

    def _reopen(self, t, name):
        self.closed.pop(name, None)
        self.events.append((t, name, "reopen"))


def _requests(case, cells):
    """Список векторов: (ключ, тип, аргумент). Берётся из SUMMARY deck и дополняется cells."""
    out = []
    names = [w for w in case.wells]
    for kw, recs in case.summary_requests:
        if kw[0] == "F" and kw in ("FOPR", "FWPR", "FGPR", "FGOR", "FWCT", "FOPT", "FWPT", "FGPT", "FWIR", "FGIR", "FGLIR"):
            out.append((kw, "F", None))
        elif kw[0] == "W" and kw[1:] in ("BHP", "THP", "GLIR", "OPR", "WPR", "GPR", "WIR", "GIR", "OIR", "OPT", "WPT", "GPT",
                                         "OIT", "WIT", "GIT", "GOR"):
            listed = [str(v) for r in recs for v in r if v is not None] or names
            for n in listed:
                for m in case._match_wells(n):
                    out.append((f"{kw}:{m}", "W", (kw, m)))
        elif kw == "GPR":
            listed = [str(v) for r in recs for v in r if v is not None] or \
                ["FIELD"] + sorted(set(case.group_parent) - {"FIELD"})
            for g in listed:
                out.append((f"{kw}:{g}", "N", g))
        elif kw[0] == "G" and kw[1:] in ("OPR", "WPR", "GPR", "WIR", "GIR", "OPT", "WPT", "GPT", "WIT", "GIT"):
            listed = [str(v) for r in recs for v in r if v is not None] or ["FIELD"]
            for g in listed:
                out.append((f"{kw}:{g}", "G", (kw, g)))
        elif kw in ("BPR", "BGSAT", "BWSAT", "BOSAT"):
            for r in recs:
                if r:
                    ijk = tuple(int(v) for v in r[:3])
                    out.append((f"{kw}:{ijk[0]},{ijk[1]},{ijk[2]}", "B", (kw, ijk)))
    for kw in ("BPR", "BGSAT"):
        for ijk in cells:
            key = f"{kw}:{ijk[0]},{ijk[1]},{ijk[2]}"
            if all(k != key for k, _, _ in out):
                out.append((key, "B", (kw, tuple(ijk))))
    for kw in ("FOPR", "FWPR", "FGPR", "FGOR", "FWIR", "FGIR"):
        if all(k != kw for k, _, _ in out):
            out.append((kw, "F", None))
    seen = set()
    return [r for r in out if not (r[0] in seen or seen.add(r[0]))]


def run_deck(path, device=None, sim_overrides=None, cells=(), quiet=True, max_days=None, progress=None,
             checkpoint=None, checkpoint_every=1, resume=None, control_hook=None):
    """
    Возвращает (summary, info): summary — словарь массивов по отчётным шагам (t = 0 — NaN для дебитов);
    ключи как в Eclipse: 'TIME', 'FOPR', 'WBHP:PROD', 'BPR:i,j,k' (i,j,k с единицы), ...

    checkpoint — файл, куда после каждых checkpoint_every отчётных шагов сохраняется полное состояние
    расчёта; resume — такой файл, с которого расчёт продолжается (deck тот же). progress(n, t, summary) —
    после каждого отчётного шага; control_hook — см. iter_deck.
    """
    steps = iter_deck(path, device, sim_overrides, cells, quiet, max_days, checkpoint, checkpoint_every, resume,
                      control_hook)
    while True:
        try:
            n, t, summary = next(steps)
        except StopIteration as stop:
            return stop.value
        if progress:
            progress(n, t, summary)


def iter_deck(path, device=None, sim_overrides=None, cells=(), quiet=True, max_days=None,
              checkpoint=None, checkpoint_every=1, resume=None, control_hook=None, context=None, yield_start=False):
    """
    Пошаговый run_deck: генератор, отдающий (n, t, summary) после каждого отчётного шага; по окончании
    StopIteration.value = (summary, info). control_hook(n, t, wm, solver) вызывается перед расчётом шага n,
    после режимов расписания и WTEST, — может менять режимы скважин (среда для агента). context (dict)
    заполняется объектами расчёта (case, sim, wm, solver, fluid); yield_start — первым отдать (None, t, summary)
    до первого шага.
    """
    case = EclipseCase(path)
    u = case.u
    state = None
    if resume is not None:
        # CSR-матрицы решателя хранят столбцы строки в порядке сборки, а не отсортированными
        validate = torch._utils._validate_loaded_sparse_tensors
        torch._utils._validate_loaded_sparse_tensors = lambda: torch._utils._sparse_tensors_to_validate.clear()
        try:
            state = torch.load(resume, map_location=device, weights_only=False)
        finally:
            torch._utils._validate_loaded_sparse_tensors = validate
        sim, wm, fluid = state["sim"], state["wm"], state["fluid"]
    else:
        out = io.StringIO()
        with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
            res, fluid, wm, sim_params = case.build(device=device, sim_overrides=sim_overrides)
            sim = Simulator(res, fluid, wm, sim_params, device=device or torch.device("cpu"))
    solver = sim.fim_solver
    rate_u = {"oil": u["liq"], "water": u["liq"], "gas": u["gas"]}
    requests = _requests(case, cells)
    from .groups import GroupTree
    tree = GroupTree(case.group_parent, case.well_group)
    wells = {w.name: w for w in wm.wells}
    summary = {"TIME": [0.0]}
    cmap = getattr(solver.reservoir, "cell_map", None)

    def block_value(kw, ijk):
        nx, ny, nz = case.dims
        i, j, k = (v - 1 for v in ijk)
        c = (i * ny + j) * nz + k
        c = int(cmap[c]) if cmap is not None else c
        if c < 0:
            return float("nan")
        sw, sg = float(fluid.s_w.reshape(-1)[c]), float(fluid.s_g.reshape(-1)[c])
        return {"BPR": lambda: float(fluid.pressure.reshape(-1)[c]) / u["p"], "BGSAT": lambda: sg,
                "BWSAT": lambda: sw, "BOSAT": lambda: 1.0 - sw - sg}[kw]()

    def rates(w):
        r = getattr(w, "last_surface_rates", None) or {}
        return {ph: r.get(ph, 0.0) / rate_u[ph] for ph in ("oil", "water", "gas")}

    def record():
        f = {"oil": 0.0, "water": 0.0, "gas": 0.0}
        fi = {"water": 0.0, "gas": 0.0}
        wr = {}
        for w in wm.wells:
            q = rates(w)
            wr[w.name] = q
            e = float(getattr(w, "eff", 1.0) or 1.0)          # WEFAC: групповые дебиты — ε·q
            for ph in f:
                if w.type == "producer":
                    f[ph] += e * q[ph]
                elif ph in fi:
                    fi[ph] += -e * q[ph]
        lift = lambda w: float(getattr(w, "alq", 0.0) or 0.0) if (
            w.type == "producer" and w.control_type != "shut" and getattr(w, "vfp_table", None) is not None) else 0.0
        vals = {"FGLIR": sum(lift(w) * float(getattr(w, "eff", 1.0) or 1.0) for w in wm.wells), "FOPR": f["oil"], "FWPR": f["water"], "FGPR": f["gas"], "FWIR": fi["water"], "FGIR": fi["gas"],
                "FGOR": f["gas"] / f["oil"] if f["oil"] > 0 else np.nan,
                "FWCT": f["water"] / (f["water"] + f["oil"]) if f["water"] + f["oil"] > 0 else np.nan}
        cum = {w.name: {ph: (getattr(w, "cum_prod", None) or {}).get(ph, 0.0) / rate_u[ph]
                        for ph in ("oil", "water", "gas")} for w in wm.wells}
        cum_i = {w.name: {ph: (getattr(w, "cum_inj", None) or {}).get(ph, 0.0) / rate_u[ph]
                          for ph in ("oil", "water", "gas")} for w in wm.wells}
        for key, kind, arg in requests:
            if kind == "F":
                if key.endswith("PT"):
                    ph = _PHASE[key[1]]
                    v = sum(cum[w.name][ph] for w in wm.wells)
                else:
                    v = vals[key]
            elif kind == "W":
                kw, name = arg
                w = wells[name]
                if kw == "WBHP":
                    v = np.nan if getattr(w, "bhp_mpa", None) is None else w.bhp_mpa * 1e6 / u["p"]
                    if w.control_type == "shut":
                        v = 0.0
                elif kw == "WGLIR":
                    v = lift(w)
                elif kw == "WTHP":
                    v = 0.0 if getattr(w, "thp_mpa", None) is None else w.thp_mpa * 1e6 / u["p"]
                elif kw == "WGOR":
                    q = wr[name]
                    v = q["gas"] / q["oil"] if w.type == "producer" and q["oil"] > 0 else 0.0
                else:
                    ph = _PHASE[kw[1]]
                    prod = kw[2] == "P"
                    q = wr[name][ph]
                    rate = (q if w.type == "producer" else 0.0) if prod else (-q if w.type == "injector" else 0.0)
                    if kw.endswith("T"):
                        v = cum[name][ph] if prod else cum_i[name][ph]
                    else:
                        v = rate
            elif kind == "G":
                kw, g = arg
                ph = _PHASE[kw[1]]
                prod = kw[2] == "P"
                v = 0.0
                for w in wm.wells:
                    if g not in tree.ancestors(case.well_group.get(w.name, "FIELD")):
                        continue
                    if kw.endswith("T"):
                        v += cum[w.name][ph] if prod else cum_i[w.name][ph]
                        continue
                    if (w.type == "producer") != prod:
                        continue
                    v += float(getattr(w, "eff", 1.0) or 1.0) * (wr[w.name][ph] if prod else -wr[w.name][ph])
            elif kind == "N":
                p = (getattr(getattr(solver, "wells", None), "node_p", None) or {}).get(arg)
                v = np.nan if p is None else p / u["p"]
            else:
                v = block_value(*arg)
            summary.setdefault(key, []).append(v)

    def record_initial():
        for key, kind, arg in requests:
            if kind == "B":
                v = block_value(*arg)
            elif key.split(":")[0].endswith(("PT", "IT")):
                v = 0.0
            else:
                v = np.nan
            summary[key] = [v]

    t0 = time.perf_counter()
    fw = getattr(solver, "wells", None)
    if state is None:
        record_initial()
        newton = substeps = n0 = 0
        t = 0.0
        wtest = WellTestState() if fw is not None and hasattr(fw, "set_open") and hasattr(fw, "conn_open") else None
    else:
        summary, wtest = state["summary"], state["wtest"]
        newton, substeps, n0, t = state["newton"], state["substeps"], state["n"], state["t"]
        del state
    if context is not None:
        context.update(case=case, sim=sim, wm=wm, solver=solver, fluid=fluid, n0=n0, t0=t)
    if yield_start:
        yield None, t, summary

    def save_checkpoint(n_next):
        tmp = f"{checkpoint}.tmp"
        torch.save({"sim": sim, "wm": wm, "fluid": fluid, "summary": summary, "wtest": wtest,
                    "newton": newton, "substeps": substeps, "n": n_next, "t": t}, tmp)
        os.replace(tmp, checkpoint)

    for n, d in enumerate(case.report_days):
        if n < n0:
            continue
        if max_days is not None and t + d > max_days + 1e-9:
            break
        # Scheduled time controls persist, while explicit build overrides win.
        time_overrides = set(sim.sim_params.get("_deck_tuning_overrides", ()))
        for key, value in case.tuning_steps[n].items():
            if key not in time_overrides:
                sim.sim_params[key] = value
        restart = case.tuning_restart_steps[n]
        if restart is not None and "initial_time_step_days" not in time_overrides:
            sim._fim_dt = restart * DAY
        well_event = (n == 0 or case.steps[n][1] != case.steps[n - 1][1]
                      or case.conn_steps[n] != case.conn_steps[n - 1])
        after_event = sim.sim_params.get("time_step_after_well_event_days")
        if well_event and after_event is not None and after_event > 0:
            sim._fim_dt = after_event * DAY
        apply_controls(case, n, wm, solver)
        if wtest is not None:
            wtest.before_step(case, n, t, wm, fw)
            wtest.apply_cutbacks(case, n, wm, fw)
            hooks = ([wtest.substep] if case.wecon_steps[n] else []) + \
                ([wtest.cutback] if case.wcutback_steps[n] else [])
            sim.substep_hook = (lambda tt, n=n, hooks=hooks: [h(case, n, tt, wm, fw) for h in hooks]) if hooks else None
        if control_hook is not None:
            control_hook(n, t, wm, solver)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
            ok = sim.run_step(d * DAY)
        sim.substep_hook = None
        log = buf.getvalue()
        newton += log.count("Iter ") - log.count("Converged!")
        substeps += log.count("Converged!")
        if not ok:
            raise RuntimeError(f"расчёт остановлен на шаге {n} (t = {t:.1f} сут):\n{log[-4000:]}")
        t += d
        if wtest is not None:
            wtest.after_step(case, n, t, wm, fw)
        summary["TIME"].append(t)
        record()
        yield n, t, summary
        if checkpoint is not None and (n + 1) % max(1, int(checkpoint_every)) == 0:
            save_checkpoint(n + 1)
    summary = {k: np.array(v, dtype=np.float64) for k, v in summary.items()}
    info = {"wall_s": time.perf_counter() - t0, "newton_iters": newton, "substeps": substeps,
            "units": case.units_name, "case": case, "solver": solver,
            "convergence": {"tol_cnv": solver.tol_cnv, "tol_mb": solver.tol_mb,
                            "relaxed_max_pv_fraction": solver.relaxed_max_pv_fraction,
                            "max_strict_iter": solver.max_strict_iter,
                            "tol_cnv_relaxed": solver.tol_cnv_relaxed,
                            "tol_mb_relaxed": solver.tol_mb_relaxed},
            "well_events": wtest.events if wtest is not None else []}
    return summary, info
