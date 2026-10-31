"""Conservative stock-component accounting and restricted isothermal black-oil separation.

This is ideal phase separation, not compositional EOS, separator sizing, or a
multiphase pipeline model. Gas dissolution in oil is supported; vaporized oil
(Rv) is rejected. Stage PVT is never extrapolated or silently clamped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .liquid import PropertyModelError, _name, _number


@dataclass(frozen=True)
class StockDensities:
    oil_kg_m3: float
    water_kg_m3: float
    gas_kg_m3: float

    def __post_init__(self):
        for name, value in asdict(self).items():
            _number(value, name, positive=True)


@dataclass(frozen=True)
class ComponentStream:
    """Standard-condition component equivalents; gas includes dissolved gas."""

    oil_m3_s: float = 0.0
    water_m3_s: float = 0.0
    gas_m3_s: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            _number(value, name, nonnegative=True)

    def mass_rates(self, densities):
        return {
            "oil": self.oil_m3_s * densities.oil_kg_m3,
            "water": self.water_m3_s * densities.water_kg_m3,
            "gas": self.gas_m3_s * densities.gas_kg_m3,
        }

    def total_mass_rate(self, densities):
        return math.fsum(self.mass_rates(densities).values())

    @classmethod
    def mix(cls, streams):
        streams = tuple(streams)
        if not all(isinstance(s, cls) for s in streams):
            raise TypeError("Can only mix ComponentStreams")
        return cls(
            *(
                math.fsum(getattr(s, key) for s in streams)
                for key in ("oil_m3_s", "water_m3_s", "gas_m3_s")
            )
        )

    def split(self, fraction):
        f = _number(fraction, "fraction", nonnegative=True)
        if f > 1:
            raise ValueError("Split fraction must be <= 1")
        values = (self.oil_m3_s, self.water_m3_s, self.gas_m3_s)
        return ComponentStream(*(v * f for v in values)), ComponentStream(
            *(v * (1 - f) for v in values)
        )

    @classmethod
    def from_mass_rates(cls, oil_kg_s, water_kg_s, gas_kg_s, densities):
        return cls(
            oil_kg_s / densities.oil_kg_m3,
            water_kg_s / densities.water_kg_m3,
            gas_kg_s / densities.gas_kg_m3,
        )


def component_balance(inputs, outputs, densities):
    incoming, outgoing = ComponentStream.mix(inputs), ComponentStream.mix(outputs)
    a, b = incoming.mass_rates(densities), outgoing.mass_rates(densities)
    return {name: a[name] - b[name] for name in a}


@dataclass(frozen=True)
class SeparationResult:
    name: str
    pressure_pa: float
    oil: ComponentStream
    water: ComponentStream
    gas: ComponentStream
    phase_volume_rates_m3_s: dict
    phase_density_kg_m3: dict
    component_mass_residuals_kg_s: dict
    dissolved_gas_ratio_m3_m3: float

    def to_dict(self):
        return asdict(self)


class BlackOilSeparator:
    """Ideal, no-entrainment stage; existing PVTTable supplies Bo/Bw/Bg/Rs.

    Undersaturated oil requires extended PVTO tables that cover pressure and
    dissolved gas ratio. Saturated Bo is never reused as undersaturated Bo.
    """

    def __init__(self, name, pressure_pa, pvt, densities):
        _name(name)
        self.name = name
        self.pressure_pa = _number(pressure_pa, "pressure_pa", positive=True)
        if not isinstance(densities, StockDensities):
            raise TypeError("Separator requires StockDensities")
        self.densities = densities
        if isinstance(pvt, dict):
            try:
                from ..simulator.pvt import PVTTable
            except ImportError:
                from simulator.pvt import PVTTable
            pvt = PVTTable(pvt)
        self.pvt = pvt
        p = np.asarray(pvt.P)
        if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or np.any(np.diff(p) <= 0):
            raise ValueError("Separator PVT pressure must be finite and strictly increasing")
        if not p[0] <= self.pressure_pa / 1e6 <= p[-1]:
            raise ValueError("Separator pressure outside PVT range")
        for column in ("Bo", "Bw", "Bg"):
            values = np.asarray(getattr(pvt, column))
            if values.shape != p.shape or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"Invalid separator PVT {column}")
        rs, rv = np.asarray(pvt.Rs), np.asarray(pvt.Rv)
        if (
            rs.shape != p.shape
            or not np.isfinite(rs).all()
            or np.any(rs < 0)
            or np.any(np.diff(rs) < 0)
        ):
            raise ValueError("Invalid separator Rs")
        if rv.shape != p.shape or not np.isfinite(rv).all() or np.any(rv != 0):
            raise ValueError("Vaporized oil Rv is not supported by this separator")

    def _oil_volume_factor(self, rs, saturated_rs, base):
        if rs >= saturated_rs:
            return base
        pvt, p = self.pvt, self.pressure_pa / 1e6
        axis = np.asarray(pvt.oil_rs)
        if (
            len(axis) < 2
            or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)
            or not axis[0] <= rs <= axis[-1]
        ):
            raise PropertyModelError("Undersaturated separator oil requires covering extended PVTO")
        upper = min(int(np.searchsorted(axis, rs, side="right")), len(axis) - 1)
        lower = max(0, upper - 1)
        for ratio in axis[lower : upper + 1]:
            table = pvt.oil_tables.get(float(ratio))
            if table is None:
                raise ValueError("Missing extended PVTO table")
            if np.any(np.asarray(table.get("Rv_m3m3", [])) != 0):
                raise ValueError("Vaporized oil Rv is not supported in extended PVTO")
            xp, bo = np.asarray(table["pressure_MPa"]), np.asarray(table["Bo"])
            if (
                len(xp) < 2
                or xp.shape != bo.shape
                or not np.isfinite(xp).all()
                or np.any(np.diff(xp) <= 0)
                or not xp[0] <= p <= xp[-1]
                or not np.isfinite(bo).all()
                or np.any(bo <= 0)
            ):
                raise PropertyModelError(
                    "Undersaturated separator pressure outside valid PVTO coverage"
                )
        value = float(pvt.eval_oil(np.array([p]), np.array([rs]))["Bo"][0])
        return _number(value, "undersaturated Bo", positive=True)

    def separate(self, feed):
        if not isinstance(feed, ComponentStream):
            raise TypeError("Separator feed must be ComponentStream")
        state = {
            k: float(v[0]) for k, v in self.pvt.eval(np.array([self.pressure_pa / 1e6])).items()
        }
        rs = min(state["Rs"], feed.gas_m3_s / feed.oil_m3_s) if feed.oil_m3_s else 0.0
        dissolved = min(feed.gas_m3_s, rs * feed.oil_m3_s)
        oil = ComponentStream(oil_m3_s=feed.oil_m3_s, gas_m3_s=dissolved)
        water = ComponentStream(water_m3_s=feed.water_m3_s)
        gas = ComponentStream(gas_m3_s=feed.gas_m3_s - dissolved)
        bo = self._oil_volume_factor(rs, state["Rs"], state["Bo"]) if feed.oil_m3_s else state["Bo"]
        volumes = {
            "oil": feed.oil_m3_s * bo,
            "water": feed.water_m3_s * state["Bw"],
            "gas": gas.gas_m3_s * state["Bg"],
        }
        phase_density = {
            name: stream.total_mass_rate(self.densities) / volumes[name]
            for name, stream in (("oil", oil), ("water", water), ("gas", gas))
            if volumes[name] > 0
        }
        residual = component_balance([feed], [oil, water, gas], self.densities)
        return SeparationResult(
            self.name, self.pressure_pa, oil, water, gas, volumes, phase_density, residual, rs
        )


def run_separator_train(feed, stages):
    """Oil advances to the next lower-pressure stage; free gas/water leave each stage."""
    stages = tuple(stages)
    if not stages:
        raise ValueError("Separator train must contain stages")
    if len({s.name for s in stages}) != len(stages):
        raise ValueError("Duplicate separator names")
    if any(b.pressure_pa >= a.pressure_pa for a, b in zip(stages, stages[1:])):
        raise ValueError("Separator train pressures must strictly decrease")
    if any(s.densities != stages[0].densities for s in stages):
        raise ValueError("Separator stages must share standard-condition densities")
    results = []
    current = feed
    for stage in stages:
        result = stage.separate(current)
        results.append(result)
        current = result.oil
    outputs = {
        "oil": current,
        "water": ComponentStream.mix(r.water for r in results),
        "gas": ComponentStream.mix(r.gas for r in results),
    }
    return results, outputs, component_balance([feed], outputs.values(), stages[0].densities)
