"""
Регионы свойств (PVTNUM, SATNUM): набор таблиц и номер региона для каждой ячейки.

Обёртки повторяют интерфейс DeckPVT / ThreePhaseRelPerm: значения считаются по таблицам всех регионов и
выбираются по номеру региона (дифференцируемо, для torch.func.jvp). Для вычислений не по ячейкам
(соединения скважин, отдельные давления) — pvt.at(индексы ячеек).
"""
import torch


def _select(reg, outs):
    """outs[r] — результат региона r (тензор или кортеж); выбор по reg (0-based)."""
    first = outs[0]
    if isinstance(first, tuple):
        return tuple(_select(reg, [o[m] for o in outs]) for m in range(len(first)))
    res = first
    for r in range(1, len(outs)):
        res = torch.where(reg == r, outs[r], res)
    return res


class _Bound:
    """PVT с фиксированными регионами (для массивов не по ячейкам)."""

    def __init__(self, parent, reg):
        self._p, self.reg = parent, reg
        self.rho_sc, self.has_pvt, self.live, self.rs_max = parent.rho_sc, parent.has_pvt, parent.live, parent.rs_max
        self.wet, self.rv_max = parent.wet, parent.rv_max

    def _reg(self, x):
        if self.reg.numel() == x.numel():
            return self.reg.reshape(x.shape)
        if self.reg.numel() == 1:
            return self.reg.reshape(-1).expand_as(x)
        raise ValueError(f"PVT по регионам: размер {tuple(x.shape)} не совпадает с числом регионов-ячеек "
                         f"{self.reg.numel()} — используйте pvt.at(ячейки)")

    def rs_sat(self, p):
        return _select(self._reg(p), [t.rs_sat(p) for t in self._p.tables])

    def oil(self, p, rs, sat):
        return _select(self._reg(p), [t.oil(p, rs, sat) for t in self._p.tables])

    def water(self, p):
        return _select(self._reg(p), [t.water(p) for t in self._p.tables])

    def rv_sat(self, p):
        return _select(self._reg(p), [t.rv_sat(p) for t in self._p.tables])

    def gas(self, p, rv=None, sat=None):
        return _select(self._reg(p), [t.gas(p, rv, sat) for t in self._p.tables])

    def at(self, idx):
        return _Bound(self._p, self.reg.reshape(-1)[idx])


class RegionPVT(_Bound):
    """
    PVT по PVTNUM: tables — список DeckPVT, reg [n_cells] (0-based). Массы компонентов считаются с плотностями
    первого региона (rho_sc): это постоянный множитель к поверхностным объёмам, которые и сохраняются. При разных
    DENSITY плотности ячеек в пласте — по своему региону (rho_cell [3, n_cells]).
    """

    def __init__(self, tables, reg):
        self.tables = tables
        self.rho_sc = tables[0].rho_sc
        self.rho_cell = None
        if len({tuple(t.rho_sc) for t in tables}) > 1:
            rho = torch.tensor([list(t.rho_sc) for t in tables], dtype=torch.float64, device=reg.device)
            self.rho_cell = rho[reg.long()].T.contiguous()
        self.has_pvt = any(t.has_pvt for t in tables)
        self.live = any(getattr(t, "live", False) for t in tables)
        self.rs_max = max(t.rs_max for t in tables)
        self.wet = any(t.wet for t in tables)
        if self.wet and not all(t.wet for t in tables):
            raise NotImplementedError("VAPOIL: PVTG нужен во всех регионах PVTNUM")
        self.rv_max = max(t.rv_max for t in tables)
        self._p, self.reg = self, reg.long()

    def region(self, r):
        return self.tables[r]


class RegionRelPerm:
    """ОФП и Pc по SATNUM: tables — список ThreePhaseRelPerm, reg [n_cells] (0-based)."""

    def __init__(self, tables, reg):
        self.tables = tables
        self.reg = reg.long()
        self.swc = tables[0].swc
        self.kro_model = tables[0].kro_model

    def __call__(self, sw, sg):
        return _select(self.reg, [t(sw, sg) for t in self.tables])

    def pc(self, sw, sg):
        return _select(self.reg, [t.pc(sw, sg) for t in self.tables])

    def region(self, r):
        return self.tables[r]
