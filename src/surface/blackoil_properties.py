"""Strict differentiable Rs-only surface flash in stock-component mass coordinates.

Float64 on CPU or CUDA. Pressure in Pa, viscosities in Pa s, stock masses in kg/s.
This snapshots PVTTable data and never invokes its endpoint clamping.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from .liquid import PropertyModelError, _number
from .process import StockDensities


class BlackOilMixture:
    def __init__(self, pvt, densities):
        if not isinstance(densities, StockDensities):
            raise TypeError("Expected StockDensities")
        if isinstance(pvt, dict):
            try:
                from ..simulator.pvt import PVTTable
            except ImportError:
                from simulator.pvt import PVTTable
            pvt = PVTTable(pvt)
        self.pvt = copy.deepcopy(pvt)
        self.densities = densities
        self.rho = torch.tensor(
            [densities.oil_kg_m3, densities.water_kg_m3, densities.gas_kg_m3], dtype=torch.float64
        )
        p = np.asarray(pvt.P, dtype=float)
        self._validate_axis(p, "PVT pressure")
        if p[0] <= 0:
            raise PropertyModelError("PVT pressure must be positive")
        self.pressures = torch.tensor(p * 1e6, dtype=torch.float64)
        self.columns = {}
        for key in ("Bo", "Bw", "Bg", "mu_o", "mu_w", "mu_g", "Rs", "Rv"):
            values = np.asarray(getattr(pvt, key), dtype=float)
            if values.shape != p.shape or not np.isfinite(values).all():
                raise PropertyModelError(f"Invalid PVT {key}")
            if key not in ("Rs", "Rv") and np.any(values <= 0):
                raise PropertyModelError(f"Nonpositive PVT {key}")
            self.columns[key] = torch.tensor(values, dtype=torch.float64)
        if bool((self.columns["Rv"] != 0).any()):
            raise PropertyModelError("Surface mixture does not support vaporized oil Rv")
        if bool((self.columns["Rs"] < 0).any()) or bool((torch.diff(self.columns["Rs"]) < 0).any()):
            raise PropertyModelError("Rs must be nonnegative and nondecreasing")
        self.oil_rs = torch.tensor(np.asarray(pvt.oil_rs), dtype=torch.float64)
        self.oil_tables = {}
        if self.oil_rs.numel():
            self._validate_axis(self.oil_rs.numpy(), "PVTO Rs")
            if float(self.oil_rs[0]) < 0:
                raise PropertyModelError("PVTO Rs must be nonnegative")
            for ratio in self.oil_rs.tolist():
                table = pvt.oil_tables.get(ratio)
                if table is None:
                    raise PropertyModelError("Missing extended PVTO table")
                xp = np.asarray(table["pressure_MPa"], dtype=float)
                self._validate_axis(xp, "PVTO pressure")
                if xp[0] > p[0] or xp[-1] < p[-1]:
                    raise PropertyModelError(
                        "Extended PVTO must cover the entire surface PVT pressure range"
                    )
                if np.any(np.asarray(table.get("Rv_m3m3", [])) != 0):
                    raise PropertyModelError("Extended PVTO Rv is not supported")
                columns = []
                for key in ("Bo", "mu_o_cP"):
                    values = np.asarray(table[key], dtype=float)
                    if (
                        values.shape != xp.shape
                        or not np.isfinite(values).all()
                        or np.any(values <= 0)
                    ):
                        raise PropertyModelError(f"Invalid extended PVTO {key}")
                    columns.append(torch.tensor(values, dtype=torch.float64))
                self.oil_tables[ratio] = (torch.tensor(xp * 1e6, dtype=torch.float64), *columns)

        self._pressure_bounds = (float(p[0] * 1e6), float(p[-1] * 1e6))
        self._fixed_values = tuple(
            sorted(
                set(
                    self.pressures.tolist()
                    + [float(x) for table in self.oil_tables.values() for x in table[0]]
                )
            )
        )
        self._fixed_pressures = torch.tensor(self._fixed_values, dtype=torch.float64)

    def to(self, device):
        """Return an independent validated property snapshot on the target device."""
        result = BlackOilMixture(self.pvt, self.densities)
        for name in ("rho", "pressures", "oil_rs", "_fixed_pressures"):
            setattr(result, name, getattr(result, name).to(device))
        result.columns = {k: v.to(device) for k, v in result.columns.items()}
        result.oil_tables = {
            k: tuple(v.to(device) for v in table) for k, table in result.oil_tables.items()
        }
        return result

    @staticmethod
    def _validate_axis(values, name):
        if (
            values.ndim != 1
            or len(values) < 2
            or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)
        ):
            raise PropertyModelError(f"{name} must be a finite strictly increasing axis")

    @staticmethod
    def interpolate(p, axis, values):
        i = (torch.searchsorted(axis, p.contiguous(), right=True) - 1).clamp(0, len(axis) - 2)
        return values[i] + (p - axis[i]) * (values[i + 1] - values[i]) / (axis[i + 1] - axis[i])

    @property
    def pressure_bounds_pa(self):
        return self._pressure_bounds

    def validate(self, pressure_pa, component_mass_rates):
        p = _number(pressure_pa, "pressure_pa", positive=True)
        if not self.pressure_bounds_pa[0] <= p <= self.pressure_bounds_pa[1]:
            raise PropertyModelError("Surface pressure outside PVT coverage")
        m = torch.as_tensor(component_mass_rates, dtype=torch.float64, device=self.rho.device)
        if (
            m.shape != (3,)
            or not bool(torch.isfinite(m).all())
            or bool((m < 0).any())
            or not bool(m.sum() > 0)
        ):
            raise PropertyModelError(
                "Expected nonnegative oil/water/gas component masses with positive total"
            )
        q = m / self.rho
        if float(q[0]) > 0 and float(q[2] / q[0]) < float(self.columns["Rs"][-1]):
            ratio = float(q[2] / q[0])
            if len(self.oil_rs) < 2 or not float(self.oil_rs[0]) <= ratio <= float(self.oil_rs[-1]):
                raise PropertyModelError(
                    "Undersaturated surface oil requires covering extended PVTO Rs"
                )
        if float(q[0]) > 0:
            ratio = float(q[2] / q[0])
            rs_axis = self.columns["Rs"]
            if float(rs_axis[0]) < ratio < float(rs_axis[-1]):
                i = int(torch.searchsorted(rs_axis, self.rho.new_tensor(ratio), right=True)) - 1
                bubble = self.pressures[i] + (ratio - rs_axis[i]) * (
                    self.pressures[i + 1] - self.pressures[i]
                ) / (rs_axis[i + 1] - rs_axis[i])
                j = int(
                    (
                        torch.searchsorted(self.oil_rs, self.rho.new_tensor(ratio), right=True) - 1
                    ).clamp(0, len(self.oil_rs) - 2)
                )
                lo, hi = self.oil_rs[j], self.oil_rs[j + 1]
                t0, t1 = self.oil_tables[float(lo)], self.oil_tables[float(hi)]
                weight = (ratio - lo) / (hi - lo)
                for column, index in (("Bo", 1), ("mu_o", 2)):
                    saturated = self.interpolate(bubble, self.pressures, self.columns[column])
                    under = (
                        self.interpolate(bubble, t0[0], t0[index]) * (1 - weight)
                        + self.interpolate(bubble, t1[0], t1[index]) * weight
                    )
                    if not bool(torch.isclose(saturated, under, rtol=1e-7, atol=1e-10)):
                        raise PropertyModelError(
                            "Saturated and undersaturated PVTO must be continuous at the stream bubble pressure"
                        )
        return m

    def split_pressures(self, lower, upper, masses):
        """Differentiable bubble pressure and fixed interpolation breakpoints."""
        low, high = float(lower.detach()), float(upper.detach())
        points = [(low, lower), (high, upper)]
        for i, value in enumerate(self._fixed_values):
            if low < value < high:
                points.append((value, self._fixed_pressures[i]))
        q = masses / self.rho
        if float(q[0].detach()) > 0:
            ratio = q[2] / q[0]
            value = float(ratio.detach())
            rs = self.columns["Rs"]
            if self.pvt.Rs[0] < value < self.pvt.Rs[-1]:
                i = int(np.searchsorted(self.pvt.Rs, value, side="right")) - 1
                bubble = self.pressures[i] + (ratio - rs[i]) * (
                    self.pressures[i + 1] - self.pressures[i]
                ) / (rs[i + 1] - rs[i])
                value = float(bubble.detach())
                if low < value < high:
                    points.append((value, bubble))
        points.sort(key=lambda t: t[0])
        unique = []
        for value, p in points:
            if not unique or value != unique[-1][0]:
                unique.append((value, p))
            elif p.requires_grad:
                unique[-1] = (value, p)
        return [p for value, p in unique]

    def tensor_state(self, pressure, masses):
        """Differentiable inside validated pressure/composition regions; no clamping."""
        q = masses / self.rho
        interp = lambda key: self.interpolate(pressure, self.pressures, self.columns[key])
        sat_rs = interp("Rs")
        if float(q[0].detach()) > 0:
            ratio = q[2] / q[0]
            rs = torch.minimum(sat_rs, ratio)
            saturated = sat_rs <= ratio
        else:
            rs = pressure * 0
            ratio = q[0] * 0
            # At pure-water composition use the one-sided zero-GOR oil
            # endpoint when it exists, so an oil perturbation is meaningful.
            saturated = torch.full_like(
                pressure, bool(float(q[2].detach()) > 0 or len(self.oil_rs) < 2), dtype=torch.bool
            )
        bo, muo = interp("Bo"), interp("mu_o")
        if not bool(saturated.all()):
            if len(self.oil_rs) < 2:
                raise PropertyModelError("Missing undersaturated PVTO")
            i = int(
                (torch.searchsorted(self.oil_rs, ratio.detach(), right=True) - 1).clamp(
                    0, len(self.oil_rs) - 2
                )
            )
            lo, hi = self.oil_rs[i], self.oil_rs[i + 1]
            t0, t1 = self.oil_tables[float(lo)], self.oil_tables[float(hi)]
            w = (ratio - lo) / (hi - lo)
            unsat_bo = (
                self.interpolate(pressure, t0[0], t0[1]) * (1 - w)
                + self.interpolate(pressure, t1[0], t1[1]) * w
            )
            unsat_mu = (
                self.interpolate(pressure, t0[0], t0[2]) * (1 - w)
                + self.interpolate(pressure, t1[0], t1[2]) * w
            )
            bo, muo = torch.where(saturated, bo, unsat_bo), torch.where(saturated, muo, unsat_mu)
        dissolved = q[0] * rs
        volumes = torch.stack(
            [q[0] * bo, q[1] * interp("Bw"), (q[2] - dissolved) * interp("Bg")], dim=-1
        )
        phase_mass = torch.stack(
            [
                masses[0] + dissolved * self.rho[2],
                masses[1] + pressure * 0,
                masses[2] - dissolved * self.rho[2],
            ],
            dim=-1,
        )
        volume = volumes.sum(-1)
        viscosities = torch.stack([muo, interp("mu_w"), interp("mu_g")], dim=-1) * 0.001
        return {
            "specific_volume": volume / masses.sum(),
            "viscosity_pa_s": (volumes * viscosities).sum(-1) / volume,
            "phase_volume_rates_m3_s": volumes,
            "phase_mass_rates_kg_s": phase_mass,
            "rs_m3_m3": rs,
        }

    @torch.enable_grad()
    def state(self, pressure_pa, component_mass_rates, derivatives=False):
        m = self.validate(pressure_pa, component_mass_rates)
        if (
            derivatives
            and float(m[0]) == 0
            and float(m[2]) == 0
            and (len(self.oil_rs) < 2 or not float(self.oil_rs[0]) <= 0 <= float(self.oil_rs[-1]))
        ):
            raise PropertyModelError(
                "Pure-water composition derivatives require covering zero-GOR oil PVTO"
            )
        x = torch.cat(
            [torch.tensor([pressure_pa], dtype=torch.float64, device=self.rho.device), m]
        ).requires_grad_(derivatives)
        state = self.tensor_state(x[0], x[1:])
        result = {key: value.detach().tolist() for key, value in state.items()}
        if derivatives:
            rows = []
            for key in ("specific_volume", "viscosity_pa_s"):
                rows.append(torch.autograd.grad(state[key], x, retain_graph=True)[0].tolist())
            result["derivative_columns"] = ["pressure_pa", "oil_kg_s", "water_kg_s", "gas_kg_s"]
            result["derivatives"] = dict(zip(("specific_volume", "viscosity_pa_s"), rows))
        return result
