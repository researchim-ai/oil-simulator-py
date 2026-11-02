"""Explicit single-phase or homogeneous-equilibrium PR network restrictions."""

import math
from dataclasses import dataclass

import numpy as np

from .liquid import PropertyModelError, _name, _number
from .mixture_nozzle import MixtureEquilibriumNozzle
from .single_phase_nozzle import PRSinglePhaseNozzle


class _NozzleConnection:
    def __post_init__(self):
        for key in ("name", "source", "target"):
            _name(getattr(self, key))
        if self.source == self.target:
            raise ValueError("Nozzle connection cannot be a self loop")
        if not isinstance(self.allow_reverse, bool):
            raise TypeError("allow_reverse must be boolean")
        if any(
            p not in ("bulk", "phase_0", "phase_1") for p in (self.source_port, self.target_port)
        ):
            raise ValueError("Unknown nozzle phase port")
        for key in ("max_pressure_difference_pa", "max_molar_rate_mol_s"):
            _number(getattr(self, key), key, positive=True)
        self._model()

    def flow(self, pressures, validate=True, factor=1.0, *, values=None, fluid=None, cache=None):
        factor = _number(factor, "area_factor", nonnegative=True)
        if factor > 1:
            raise ValueError("Area factor must be in [0,1]")
        drop = pressures[self.source] - pressures[self.target]
        report_cache = {} if cache is None else cache
        if factor == 0 or drop == 0 or drop < 0 and not self.allow_reverse:
            report_cache[("diagnostic", self.name)] = {
                "regime": "closed" if factor == 0 else "no_flow"
            }
            return 0.0
        if validate and abs(drop) > self.max_pressure_difference_pa:
            raise PropertyModelError("Nozzle connection outside pressure-difference domain")
        if values is None or fluid is None:
            raise ValueError("PR nozzle connection needs current vessel states and EOS")
        source, target, port = (
            (self.source, self.target, self.source_port)
            if drop > 0
            else (self.target, self.source, self.target_port)
        )
        from .vessel_network import _port

        n, p, t, q, _, _, _ = values[source]
        z, _, _ = _port(q, n, port)
        result = self._model().run(
            p,
            t,
            pressures[target],
            z,
            fluid,
            opening=factor,
            with_outlet=False,
            _cache=report_cache,
        )
        mw = sum(x * c.molar_mass_kg_mol for x, c in zip(z, fluid.components))
        rate = result["mass_rate_kg_s"] / mw
        if validate and rate > self.max_molar_rate_mol_s:
            raise PropertyModelError("Nozzle connection outside molar-flow domain")
        report_cache[("diagnostic", self.name)] = self._diagnostic(result)
        report_cache[("result", self.name)] = result
        return rate if drop > 0 else -rate


@dataclass(frozen=True)
class PRNozzleConnection(_NozzleConnection):
    name: str
    source: str
    target: str
    throat_area_m2: float
    max_pressure_difference_pa: float
    max_molar_rate_mol_s: float
    discharge_coefficient: float = 1.0
    minimum_pressure_ratio: float = 0.4
    source_port: str = "bulk"
    target_port: str = "bulk"
    allow_reverse: bool = False
    pressure_tolerance_pa: float = 0.001
    flow_model = "single_phase_PR_sonic_nozzle"

    def _model(self):
        return PRSinglePhaseNozzle(
            self.name,
            self.throat_area_m2,
            self.minimum_pressure_ratio,
            self.discharge_coefficient,
            self.pressure_tolerance_pa,
        )

    def _diagnostic(self, result):
        return {k: result[k] for k in ("regime", "critical_pressure_pa", "throat_mach")}


@dataclass(frozen=True)
class HEMNozzleConnection(_NozzleConnection):
    name: str
    source: str
    target: str
    throat_area_m2: float
    minimum_pressure_pa: float
    max_pressure_difference_pa: float
    max_molar_rate_mol_s: float
    discharge_coefficient: float = 1.0
    scan_points: int = 9
    source_port: str = "bulk"
    target_port: str = "bulk"
    allow_reverse: bool = False
    pressure_tolerance_pa: float = 0.01
    flow_model = "PR_homogeneous_equilibrium_nozzle"
    relative_flow_difference_step = 1e-5

    def _model(self):
        return MixtureEquilibriumNozzle(
            self.name,
            self.throat_area_m2,
            self.minimum_pressure_pa,
            self.discharge_coefficient,
            self.scan_points,
            self.pressure_tolerance_pa,
        )

    def linearized_flow(self, pressures, values, fluid, factor, cache):
        """Local Jacobian only; nonlinear residuals always use the full HEM.

        Envelope theorem at a smooth interior maximum: dG/dPcrit=0, so source
        derivatives do not require searching a perturbed critical pressure.
        Non-smooth/phase-boundary cases fall back to the full numerical Jacobian.
        """
        from .liquid import ConservationError, HydraulicConvergenceError
        from .vessel_network import _port

        drop = pressures[self.source] - pressures[self.target]
        if not drop or not factor or drop < 0 and not self.allow_reverse:
            return None
        source, target, port = (
            (self.source, self.target, self.source_port)
            if drop > 0
            else (self.target, self.source, self.target_port)
        )
        n, p, t, q, _, _, _ = values[source]
        z, _, _ = _port(q, n, port)
        z = np.array(z)
        base = cache.get(("result", self.name))
        if base is None:
            return None
        throat = base["throat"]
        beta = throat.get("vapor_molar_fraction")
        if beta is not None and not 1e-4 < beta < 1 - 1e-4:
            return None
        critical = base["regime"] == "choked"
        pc = base["critical_pressure_pa"] if critical else pressures[target]
        if critical and pc <= self.minimum_pressure_pa + 2 * self.pressure_tolerance_pa:
            return None
        sign = 1.0 if drop > 0 else -1.0

        def fixed_flux(ps, ts, zs, pt):
            inlet = fluid.flash_pt(ps, ts, zs)
            state = fluid.flash_ps(
                pt, inlet["molar_entropy_j_mol_k"], zs, temperature_guess_k=throat["temperature_k"]
            )
            if (
                state["phase_count"] != throat["phase_count"]
                or inlet["phase_count"] != base["inlet"]["phase_count"]
            ):
                raise PropertyModelError("HEM tangent crosses a phase boundary")
            mw = sum(x * c.molar_mass_kg_mol for x, c in zip(zs, fluid.components))
            drop_h = (inlet["molar_enthalpy_j_mol"] - state["molar_enthalpy_j_mol"]) / mw
            if drop_h <= 0:
                raise PropertyModelError("HEM tangent has no positive acceleration")
            return (
                self.discharge_coefficient
                * self.throat_area_m2
                * factor
                * state["density_kg_m3"]
                * math.sqrt(2 * drop_h)
                / mw
            )

        try:
            base_rate = fixed_flux(p, t, z, pc)
            if critical:
                dp = max(pc * 1e-5, 10 * self.pressure_tolerance_pa)
                if pc - dp <= self.minimum_pressure_pa or pc + dp >= p:
                    return None
                plus = fixed_flux(p, t, z, pc + dp)
                minus = fixed_flux(p, t, z, pc - dp)
                stationarity = (plus - minus) / (2 * dp)
                scale = pc / max(base_rate, 1e-30)
                if (
                    abs(stationarity * scale) > 1e-4
                    or max(abs((plus - base_rate) / dp), abs((base_rate - minus) / dp)) * scale
                    > 1e-3
                ):
                    return None
            dp = p * 1e-5
            dt = t * 1e-5
            gp = (fixed_flux(p + dp, t, z, pc) - fixed_flux(p - dp, t, z, pc)) / (2 * dp)
            gt = (fixed_flux(p, t + dt, z, pc) - fixed_flux(p, t - dt, z, pc)) / (2 * dt)
            gz = []
            for i in range(len(z) - 1):
                dz = min(1e-5, z[i] * 0.01, z[-1] * 0.01)
                plus = z.copy()
                minus = z.copy()
                plus[i] += dz
                plus[-1] -= dz
                minus[i] -= dz
                minus[-1] += dz
                gz.append((fixed_flux(p, t, plus, pc) - fixed_flux(p, t, minus, pc)) / (2 * dz))
            gb = 0.0
            if not critical:
                dpb = pressures[target] * 1e-5
                if pc - dpb <= self.minimum_pressure_pa or pc + dpb >= p:
                    return None
                gb = (fixed_flux(p, t, z, pc + dpb) - fixed_flux(p, t, z, pc - dpb)) / (2 * dpb)
        except (PropertyModelError, HydraulicConvergenceError, ConservationError):
            return None
        if not all(math.isfinite(v) for v in [gp, gt, gb, *gz]):
            return None
        # Anchor the tangent to the full-search rate, not the diagnostic
        # fixed-pressure reconstruction, so the physical base residual is exact.
        mw = sum(x * c.molar_mass_kg_mol for x, c in zip(z, fluid.components))
        anchored = base["mass_rate_kg_s"] / mw

        def linear(pressures_new, values_new):
            nn, pp, tt, qq, _, _, _ = values_new[source]
            zz, _, _ = _port(qq, nn, port)
            return sign * (
                anchored
                + gp * (pp - p)
                + gt * (tt - t)
                + np.dot(gz, np.array(zz[:-1]) - z[:-1])
                + gb * (pressures_new[target] - pressures[target])
            )

        return linear

    def _diagnostic(self, result):
        return {
            k: result[k] for k in ("regime", "critical_pressure_pa", "critical_pressure_resolved")
        } | {
            "upstream_port_phase_count": result["inlet"]["phase_count"],
            "throat_phase_count": result["throat"]["phase_count"],
            "throat_vapor_molar_fraction": result["throat"].get("vapor_molar_fraction"),
        }


def vessel_connection_from_dict(spec):
    from .vessel_network import VesselConnection

    spec = dict(spec)
    model = spec.pop("model", "linear")
    if model == "linear":
        return VesselConnection(**spec)
    if model == "pr_single_phase_nozzle":
        return PRNozzleConnection(**spec)
    if model == "pr_hem_nozzle":
        return HEMNozzleConnection(**spec)
    raise ValueError(f"Unknown dynamic connection model: {model}")
