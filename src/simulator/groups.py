"""
Групповые контроли (GRUPTREE / GCONPROD / GCONINJE) — явное распределение целей группы по скважинам
в начале каждого шага по времени, как в Eclipse/OPM при управлении по guide rate = потенциалу скважины.

Скважина участвует в распределении цели группы, если она открыта, принадлежит поддереву группы, нужного типа
и находится под групповым управлением (режим GRUP) либо её собственная цель/потенциал не позволяют группе
уложиться в цель (Eclipse переводит такие скважины под управление группы). Доля скважины ограничена её
потенциалом (дебит при предельном BHP); недобор перераспределяется между остальными (water filling).
"""
import numpy as np

PHASES = ("oil", "water", "gas")


def allocate(target, guide, cap):
    """Делит target пропорционально guide, не превышая cap; возвращает доли (сумма = min(target, Σcap))."""
    guide = np.maximum(np.asarray(guide, dtype=np.float64), 0.0)
    cap = np.maximum(np.asarray(cap, dtype=np.float64), 0.0)
    share = np.zeros_like(cap)
    free = cap > 0
    rest = float(target)
    for _ in range(cap.size + 1):
        if rest <= 0 or not free.any():
            break
        g = np.where(free, guide, 0.0)
        if g.sum() <= 0:
            g = free.astype(np.float64)
        trial = rest * g / g.sum()
        over = free & (share + trial >= cap)
        if not over.any():
            share = share + trial
            break
        rest -= float((cap - share)[over].sum())
        share[over] = cap[over]
        free &= ~over
    return share


class GroupTree:
    """Дерево групп и принадлежность скважин."""

    def __init__(self, parent, well_group):
        self.parent = dict(parent)
        self.well_group = dict(well_group)

    def ancestors(self, g):
        out = [g]
        while g in self.parent and g != "FIELD":
            g = self.parent[g]
            out.append(g)
        if out[-1] != "FIELD":
            out.append("FIELD")
        return out

    def wells_of(self, group, names):
        return [n for n in names if group in self.ancestors(self.well_group.get(n, "FIELD"))]

    def depth(self, g):
        return len(self.ancestors(g))


def group_targets(tree, constraints, wells, pot, current, prod_res=None):
    """
    Распределение групповых целей.

    wells: список dict(name, type, open, grup (bool), phase — фаза собственного контроля или None (BHP),
           value — собственная цель в этой фазе или None);
    pot[name] = {фаза: потенциал при предельном BHP} (поверхностные м³/с; для RESV ключ 'resv');
    current[name] = {фаза: текущий дебит} — для REIN/VREP (продукция группы);
    constraints: список dict(group, kind 'prod'|'inj', phase ('oil'|'water'|'gas'|'liquid'|'resv'),
                 target (м³/с), [mode REIN/VREP: frac, src_phase]).
    Возвращает {скважина: (фаза, цель)} — новые цели скважин под групповым управлением.
    """
    names = [w["name"] for w in wells]
    by = {w["name"]: w for w in wells}
    out = {}

    def rate_in(name, ph):
        p = pot[name]
        if ph == "liquid":
            return p["oil"] + p["water"]
        return p[ph]

    for con in sorted(constraints, key=lambda c: -tree.depth(c["group"])):
        kind = con["kind"]
        members = [n for n in tree.wells_of(con["group"], names)
                   if by[n]["open"] and by[n]["type"] == ("producer" if kind == "prod" else "injector")
                   and (kind == "prod" or by[n].get("inj_phase") == con["phase"])]
        if not members:
            continue
        ph = con["phase"]
        target = con["target"]
        if con.get("mode") == "REIN":
            src = tree.wells_of(con.get("src") or con["group"], names)
            target = con["frac"] * sum(current[n].get(ph, 0.0) for n in src if by[n]["type"] == "producer")
        elif con.get("mode") == "VREP":
            src = tree.wells_of(con.get("src") or con["group"], names)
            target = con["frac"] * sum((prod_res or {}).get(n, 0.0) for n in src if by[n]["type"] == "producer")
        cap = []
        for n in members:
            c = rate_in(n, ph)
            if n in out:                                  # уже ограничена подгруппой: та же доля потенциала
                oph, oval = out[n]
                base = rate_in(n, oph)
                c = min(c, c * oval / base) if base > 0 else 0.0
            elif not by[n]["grup"] and by[n]["phase"] is not None:
                own = rate_in(n, by[n]["phase"])
                if own > 0 and by[n]["value"] is not None:
                    c = min(c, c * by[n]["value"] / own)
            # пределы дебитов WCONPROD у скважины под групповым управлением (GRUP)
            for lph, lv in (by[n].get("limits") or {}).items():
                own = rate_in(n, lph)
                if own > lv:
                    c = c * lv / own
            cap.append(c)
        cap = np.array(cap)
        if not any(by[n]["grup"] for n in members) and cap.sum() <= target * (1 + 1e-9):
            continue                                      # цель группы не нарушена — скважины на своих режимах
        guide = np.array([rate_in(n, ph) for n in members])
        share = allocate(target, guide, cap)
        for n, s in zip(members, share):
            out[n] = (ph, float(s))
    return out
